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
  Logger,
} from '@google/gemini-cli-core';
import {
  Content,
  Part,
  FunctionCall,
  GenerateContentResponse,
} from '@google/genai';

import { parseAndFormatApiError } from './ui/utils/errorParsing.js';

function getResponseText(response: GenerateContentResponse): string | null {
  if (response.candidates && response.candidates.length > 0) {
    const candidate = response.candidates[0];
    if (
      candidate.content &&
      candidate.content.parts &&
      candidate.content.parts.length > 0
    ) {
      // We are running in headless mode so we don't need to return thoughts to STDOUT.
      const thoughtPart = candidate.content.parts[0];
      if (thoughtPart?.thought) {
        return null;
      }
      return candidate.content.parts
        .filter((part) => part.text)
        .map((part) => part.text)
        .join('');
    }
  }
  return null;
}

export async function runNonInteractive(
  config: Config,
  input: string,
): Promise<void> {
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

  const chat = await geminiClient.getChat();
  const abortController = new AbortController();
  
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

  try {
    while (true) {
      const functionCalls: FunctionCall[] = [];

      const responseStream = await chat.sendMessageStream({
        message: currentMessages[0]?.parts || [], // Ensure parts are always provided
        config: {
          abortSignal: abortController.signal,
          tools: [
            { functionDeclarations: toolRegistry.getFunctionDeclarations() },
          ],
        },
      });

      let responseText = '';
      for await (const resp of responseStream) {
        if (abortController.signal.aborted) {
          console.error('Operation cancelled.');
          return;
        }
        const textPart = getResponseText(resp);
        if (textPart) {
          if (isJsonOutput) {
            responseText += textPart;
          } else {
            process.stdout.write(textPart);
          }
        }
        if (resp.functionCalls) {
          functionCalls.push(...resp.functionCalls);
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
            const isToolNotFound = toolResponse.error.message.includes(
              'not found in registry',
            );
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
            if (!isToolNotFound) {
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
        config.getContentGeneratorConfig().authType,
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
