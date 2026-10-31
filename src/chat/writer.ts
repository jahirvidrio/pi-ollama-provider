/** The assistant turn as Pi sees it: block lifecycle, tool-call flush and the terminal events. */

import {
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Model,
  type TextContent,
  type ThinkingContent,
  type ToolCall,
} from '@earendil-works/pi-ai';
import { OllamaError } from '../errors.ts';
import type { OllamaChatChunk, OllamaToolCall } from '../wire.ts';
import { ollamaToolCallToPi } from './convert.ts';
import { type BufferedToolCall, mergeToolCall } from './toolcalls.ts';
import { usageOf } from './usage.ts';

/** Notice appended to the assistant text when the server sends a tool call Pi cannot run. */
const DROPPED_TOOL_CALL_NOTICE = '[dropped a malformed tool call from the server]';

/**
 * Ollama reports failures as `{"error": ...}` inside a 200 stream, so the provider text is
 * kept verbatim: Pi matches its context-overflow patterns against `errorMessage`.
 */
function errorText (error: unknown, aborted: boolean): string {
  if (aborted) return 'Request was aborted';
  if (error instanceof OllamaError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}

/**
 * Writes one assistant turn: it owns the open text and thinking blocks and the buffered tool
 * calls, so the stream loop only translates chunks and every event Pi receives comes from here.
 */
export class TurnWriter {
  private textBlock: TextContent | undefined;
  private thinkingBlock: ThinkingContent | undefined;
  /** Calls accumulate here by arrival position and are emitted together at `done`. */
  private readonly toolCalls = new Map<number, BufferedToolCall>();
  private toolCallCount = 0;

  constructor (
    private readonly stream: AssistantMessageEventStream,
    readonly output: AssistantMessage
  ) {}

  start (): void {
    this.stream.push({ type: 'start', partial: this.output });
  }

  /** Appends one thinking fragment, opening the block (and closing the text one) if needed. */
  thinking (piece: string): void {
    if (!this.thinkingBlock) {
      this.closeText();
      this.thinkingBlock = { type: 'thinking', thinking: '' };
      this.output.content.push(this.thinkingBlock);
      this.stream.push({
        type: 'thinking_start',
        contentIndex: this.output.content.length - 1,
        partial: this.output,
      });
    }
    this.thinkingBlock.thinking += piece;
    this.stream.push({
      type: 'thinking_delta',
      contentIndex: this.output.content.length - 1,
      delta: piece,
      partial: this.output,
    });
  }

  /** Appends one text fragment, opening the block (and closing the thinking one) if needed. */
  text (piece: string): void {
    const block = this.openText();
    block.text += piece;
    this.stream.push({
      type: 'text_delta',
      contentIndex: this.output.content.length - 1,
      delta: piece,
      partial: this.output,
    });
  }

  /** Buffers fragments by position; a name and its arguments may arrive in different chunks. */
  buffer (calls: readonly OllamaToolCall[]): void {
    calls.forEach((call, index) => {
      this.toolCalls.set(index, mergeToolCall(this.toolCalls.get(index), call));
    });
  }

  /** Flushes the buffered calls, computes the stop reason and ends the stream with `done`. */
  finish (finalChunk: OllamaChatChunk, model: Model<'ollama-chat'>): void {
    this.flushToolCalls();
    const hasToolCall = this.output.content.some((block) => block.type === 'toolCall');
    const reason: 'stop' | 'length' | 'toolUse' =
      finalChunk.done_reason === 'length' ? 'length' : hasToolCall ? 'toolUse' : 'stop';
    this.output.usage = usageOf(finalChunk, model);
    this.output.stopReason = reason;
    this.stream.push({ type: 'done', reason, message: this.output });
    this.stream.end();
  }

  /** Closes the open blocks and ends the stream with `error`; buffered calls are dropped. */
  fail (error: unknown, aborted: boolean): void {
    this.closeBlocks();
    this.output.stopReason = aborted ? 'aborted' : 'error';
    this.output.errorMessage = errorText(error, aborted);
    this.stream.push({ type: 'error', reason: this.output.stopReason, error: this.output });
    this.stream.end();
  }

  private openText (): TextContent {
    if (this.textBlock) return this.textBlock;
    this.closeThinking();
    this.textBlock = { type: 'text', text: '' };
    this.output.content.push(this.textBlock);
    this.stream.push({
      type: 'text_start',
      contentIndex: this.output.content.length - 1,
      partial: this.output,
    });
    return this.textBlock;
  }

  private closeText (): void {
    if (!this.textBlock) return;
    this.stream.push({
      type: 'text_end',
      contentIndex: this.output.content.indexOf(this.textBlock),
      content: this.textBlock.text,
      partial: this.output,
    });
    this.textBlock = undefined;
  }

  private closeThinking (): void {
    if (!this.thinkingBlock) return;
    this.stream.push({
      type: 'thinking_end',
      contentIndex: this.output.content.indexOf(this.thinkingBlock),
      content: this.thinkingBlock.thinking,
      partial: this.output,
    });
    this.thinkingBlock = undefined;
  }

  private closeBlocks (): void {
    this.closeThinking();
    this.closeText();
  }

  private flushToolCalls (): void {
    const runnable: ToolCall[] = [];
    let dropped = 0;
    for (const call of this.toolCalls.values()) {
      if (call.malformed || call.name.length === 0) {
        dropped += 1;
        continue;
      }
      runnable.push(
        ollamaToolCallToPi({ function: { name: call.name, arguments: call.arguments } }, this.toolCallCount)
      );
      this.toolCallCount += 1;
    }
    if (dropped > 0) {
      // Degrade visibly instead of failing the turn: the text above may already be the answer the
      // user is reading, and a call Pi cannot validate would throw mid-turn.
      const block = this.openText();
      const notice = Array.from({ length: dropped }, () => DROPPED_TOOL_CALL_NOTICE).join('\n');
      const delta = block.text.length > 0 ? `\n${notice}` : notice;
      block.text += delta;
      this.stream.push({
        type: 'text_delta',
        contentIndex: this.output.content.indexOf(block),
        delta,
        partial: this.output,
      });
    }
    this.closeBlocks();
    for (const toolCall of runnable) {
      this.output.content.push(toolCall);
      const contentIndex = this.output.content.length - 1;
      this.stream.push({ type: 'toolcall_start', contentIndex, partial: this.output });
      this.stream.push({ type: 'toolcall_end', contentIndex, toolCall, partial: this.output });
    }
  }
}
