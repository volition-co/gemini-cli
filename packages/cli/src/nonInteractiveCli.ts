/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Config,
  ToolCallRequestInfo,
  executeToolCall,
  ToolRegistry,
  shutdownTelemetry,
  isTelemetrySdkInitialized,
  GeminiEventType,
  ToolErrorType,
} from '@google/gemini-cli-core';
import { Content, Part, FunctionCall } from '@google/genai';

import { parseAndFormatApiError } from './ui/utils/errorParsing.js';

export async function runNonInteractive(
  config: Config,
  input: string,
  prompt_id: string,
): Promise<void> {
  await config.initialize();
  // Handle EPIPE errors when the output is piped to a command that closes early.
  process.stdout.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EPIPE') {
      // Exit gracefully if the pipe is closed.
      process.exit(0);
    }
  });

  const geminiClient = config.getGeminiClient();
  const toolRegistry: ToolRegistry = await config.getToolRegistry();
  const isJsonOutput = config.getJsonOutput();

  const abortController = new AbortController();
  let currentMessages: Content[] = [{ role: 'user', parts: [{ text: input }] }];
  let turnCount = 0;
  try {
    while (true) {
      turnCount++;
      if (
        config.getMaxSessionTurns() >= 0 &&
        turnCount > config.getMaxSessionTurns()
      ) {
        console.error(
          '\n Reached max session turns for this session. Increase the number of turns by specifying maxSessionTurns in settings.json.',
        );
        return;
      }
      const functionCalls: FunctionCall[] = [];

      const responseStream = geminiClient.sendMessageStream(
        currentMessages[0]?.parts || [],
        abortController.signal,
        prompt_id,
      );

      let responseText = '';
      for await (const event of responseStream) {
        if (abortController.signal.aborted) {
          console.error('Operation cancelled.');
          return;
        }

        if (event.type === GeminiEventType.Content) {
          if (isJsonOutput) {
            responseText += event.value;
          } else {
            process.stdout.write(event.value);
          }
        } else if (event.type === GeminiEventType.ToolCallRequest) {
          const toolCallRequest = event.value;
          const fc: FunctionCall = {
            name: toolCallRequest.name,
            args: toolCallRequest.args,
            id: toolCallRequest.callId,
          };
          functionCalls.push(fc);
        }
      }

      // Output assistant response in JSON format if enabled
      if (isJsonOutput && responseText) {
        console.log(
          JSON.stringify({
            type: 'assistant',
            content: responseText,
            timestamp: new Date().toISOString(),
          }),
        );
      }

      if (functionCalls.length > 0) {
        const toolResponseParts: Part[] = [];

        for (const fc of functionCalls) {
          const callId = fc.id ?? `${fc.name}-${Date.now()}`;
          const requestInfo: ToolCallRequestInfo = {
            callId,
            name: fc.name as string,
            args: (fc.args ?? {}) as Record<string, unknown>,
            isClientInitiated: false,
            prompt_id,
          };

          // Output tool call in JSON format if enabled
          if (isJsonOutput) {
            console.log(
              JSON.stringify({
                type: 'tool_call',
                name: fc.name,
                args: fc.args,
                callId: callId,
                timestamp: new Date().toISOString(),
              }),
            );
          }

          const toolResponse = await executeToolCall(
            config,
            requestInfo,
            toolRegistry,
            abortController.signal,
          );

          if (toolResponse.error) {
            if (isJsonOutput) {
              console.log(
                JSON.stringify({
                  type: 'tool_response',
                  name: fc.name,
                  callId: callId,
                  error: true,
                  message:
                    toolResponse.resultDisplay || toolResponse.error.message,
                  timestamp: new Date().toISOString(),
                }),
              );
            } else {
              console.error(
                `Error executing tool ${fc.name}: ${toolResponse.resultDisplay || toolResponse.error.message}`,
              );
            }
            if (toolResponse.errorType === ToolErrorType.UNHANDLED_EXCEPTION) {
              process.exit(1);
            }
          } else if (isJsonOutput) {
            // Output tool response in JSON format
            console.log(
              JSON.stringify({
                type: 'tool_response',
                name: fc.name,
                callId: callId,
                success: true,
                result: toolResponse.resultDisplay || '',
                timestamp: new Date().toISOString(),
              }),
            );
          }

          if (toolResponse.responseParts) {
            const parts = Array.isArray(toolResponse.responseParts)
              ? toolResponse.responseParts
              : [toolResponse.responseParts];
            for (const part of parts) {
              if (typeof part === 'string') {
                toolResponseParts.push({ text: part });
              } else if (part) {
                toolResponseParts.push(part);
              }
            }
          }
        }
        currentMessages = [{ role: 'user', parts: toolResponseParts }];
      } else {
        if (!isJsonOutput) {
          process.stdout.write('\n'); // Ensure a final newline
        }
        return;
      }
    }
  } catch (error) {
    console.error(
      parseAndFormatApiError(
        error,
        config.getContentGeneratorConfig()?.authType,
      ),
    );
    process.exit(1);
  } finally {
    if (isTelemetrySdkInitialized()) {
      await shutdownTelemetry();
    }
  }
}
