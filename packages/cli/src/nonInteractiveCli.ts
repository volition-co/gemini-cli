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
  Logger,
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
  const chat = await geminiClient.getChat();
  
  // Initialize logger for checkpoint saving
  const logger = new Logger(config.getSessionId());
  await logger.initialize();
  
  // Load from checkpoint if specified
  const continueTag = config.getContinueFromCheckpoint();
  if (continueTag) {
    const conversation = await logger.loadCheckpoint(continueTag);
    if (conversation.length === 0) {
      const errorMsg = `No saved checkpoint found with tag: ${continueTag}`;
      if (isJsonOutput) {
        console.log(
          JSON.stringify({
            type: 'error',
            message: errorMsg,
            timestamp: new Date().toISOString(),
          }),
        );
      } else {
        console.error(errorMsg);
      }
      process.exit(1);
    }
    
    // Restore the conversation history
    chat.clearHistory();
    for (const item of conversation) {
      chat.addHistory(item);
    }
    
    if (isJsonOutput) {
      console.log(
        JSON.stringify({
          type: 'checkpoint_loaded',
          tag: continueTag,
          timestamp: new Date().toISOString(),
        }),
      );
    }
  }
  
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
                callId,
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
                  callId,
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
                callId,
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
        
        // Save checkpoint before exiting
        await saveCheckpointAndOutput(chat, logger, isJsonOutput);
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
    
    // Save checkpoint even on error
    await saveCheckpointAndOutput(chat, logger, isJsonOutput);
    process.exit(1);
  } finally {
    if (isTelemetrySdkInitialized()) {
      await shutdownTelemetry();
    }
  }
}

async function saveCheckpointAndOutput(
  chat: { getHistory: () => Content[] },
  logger: Logger,
  isJsonOutput: boolean
): Promise<void> {
  try {
    const history = chat.getHistory();
    if (history.length > 0) {
      // Generate a unique tag based on timestamp
      const tag = `auto-${Date.now()}`;
      await logger.saveCheckpoint(history, tag);
      
      // Output checkpoint info in JSON format
      if (isJsonOutput) {
        console.log(
          JSON.stringify({
            type: 'checkpoint',
            tag,
            timestamp: new Date().toISOString(),
          }),
        );
      }
    }
  } catch (error) {
    // Log error but don't fail the process
    if (!isJsonOutput) {
      console.error('Failed to save checkpoint:', error);
    }
  }
}
