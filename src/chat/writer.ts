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
import { type BufferedToolCall, emptyToolCall, fragmentName, mergeToolCall } from './toolcalls.ts';
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
  /** Calls keep their arrival order and are emitted together at `done`. */
  private readonly toolCalls: BufferedToolCall[] = [];
  /** Fragments repeating the server's `index` land in the same call. */
  private readonly toolCallsByIndex = new Map<number, BufferedToolCall>();
  /** Fragments without an `index` continue whatever their array position pointed at. */
  private readonly toolCallsByPosition = new Map<number, BufferedToolCall>();
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

  /**
   * Buffers fragments into the call they belong to. `index` wins when the server sends it;
   * otherwise a fragment continues whatever its array position pointed at, and one naming a
   * different call opens a new call on that position.
   */
  buffer (calls: readonly OllamaToolCall[]): void {
    calls.forEach((call, position) => mergeToolCall(this.slotFor(call, position), call));
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

  private slotFor (fragment: OllamaToolCall, position: number): BufferedToolCall {
    if (typeof fragment.index === 'number') {
      const known = this.toolCallsByIndex.get(fragment.index);
      if (known) return known;
      const created = this.addToolCall();
      this.toolCallsByIndex.set(fragment.index, created);
      return created;
    }
    const known = this.toolCallsByPosition.get(position);
    const name = fragmentName(fragment);
    if (known && (name === undefined || known.name.length === 0 || known.name === name)) return known;
    const created = this.addToolCall();
    this.toolCallsByPosition.set(position, created);
    return created;
  }

  private addToolCall (): BufferedToolCall {
    const created = emptyToolCall();
    this.toolCalls.push(created);
    return created;
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
    for (const buffered of this.toolCalls) {
      if (buffered.malformed || buffered.name.length === 0) {
        dropped += 1;
        continue;
      }
      const sent: OllamaToolCall = { function: { name: buffered.name, arguments: buffered.arguments } };
      if (buffered.id !== undefined) sent.id = buffered.id;
      runnable.push(ollamaToolCallToPi(sent, this.toolCallCount));
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
