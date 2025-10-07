/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Config, ToolCallRequestInfo } from '@google/gemini-cli-core';
import { isSlashCommand } from './ui/utils/commandUtils.js';
import type { LoadedSettings } from './config/settings.js';
import {
  executeToolCall,
  shutdownTelemetry,
  isTelemetrySdkInitialized,
  GeminiEventType,
  FatalInputError,
  promptIdContext,
  OutputFormat,
  JsonFormatter,
  uiTelemetryService,
  Logger,
} from '@google/gemini-cli-core';

import type { Content, Part } from '@google/genai';

import { handleSlashCommand } from './nonInteractiveCliCommands.js';
import { ConsolePatcher } from './ui/utils/ConsolePatcher.js';
import { handleAtCommand } from './ui/hooks/atCommandProcessor.js';
import {
  handleError,
  handleToolError,
  handleCancellationError,
  handleMaxTurnsExceededError,
} from './utils/errors.js';

export async function runNonInteractive(
  config: Config,
  settings: LoadedSettings,
  input: string,
  prompt_id: string,
): Promise<void> {
  return promptIdContext.run(prompt_id, async () => {
    const consolePatcher = new ConsolePatcher({
      stderr: true,
      debugMode: config.getDebugMode(),
    });

    try {
      consolePatcher.patch();
      // Handle EPIPE errors when the output is piped to a command that closes early.
      process.stdout.on('error', (err: NodeJS.ErrnoException) => {
        if (err.code === 'EPIPE') {
          // Exit gracefully if the pipe is closed.
          process.exit(0);
        }
      });

      const geminiClient = config.getGeminiClient();

      const abortController = new AbortController();
      const chat = await geminiClient.getChat();
      const logger = new Logger(config.getSessionId(), config.storage);
      await logger.initialize();

      const continueTag = config.getContinue();
      if (continueTag) {
        const conversation = await logger.loadCheckpoint(continueTag);

        // Handle missing checkpoint
        if (conversation.length === 0) {
          const errorMsg = `No conversation found with tag: ${continueTag}`;
          if (config.getOutputFormat() === OutputFormat.JSON) {
            const output = {
              type: 'error',
              message: errorMsg,
              timestamp: new Date().toISOString(),
            };
            process.stdout.write(JSON.stringify(output));
            process.stdout.write('\n');
          } else {
            console.error(errorMsg);
          }
          process.exit(1);
        }

        chat.clearHistory();

        for (const item of conversation) {
          chat.addHistory(item);
        }

        if (config.getOutputFormat() === OutputFormat.JSON) {
          const output = {
            type: 'checkpoint_loaded',
            tag: continueTag,
            timestamp: new Date().toISOString(),
          };
          process.stdout.write(JSON.stringify(output));
          process.stdout.write('\n');
        }
      }

      let query: Part[] | undefined;

      if (isSlashCommand(input)) {
        const slashCommandResult = await handleSlashCommand(
          input,
          abortController,
          config,
          settings,
        );
        // If a slash command is found and returns a prompt, use it.
        // Otherwise, slashCommandResult fall through to the default prompt
        // handling.
        if (slashCommandResult) {
          query = slashCommandResult as Part[];
        }
      }

      if (!query) {
        const { processedQuery, shouldProceed } = await handleAtCommand({
          query: input,
          config,
          addItem: (_item, _timestamp) => 0,
          onDebugMessage: () => {},
          messageId: Date.now(),
          signal: abortController.signal,
        });

        if (!shouldProceed || !processedQuery) {
          // An error occurred during @include processing (e.g., file not found).
          // The error message is already logged by handleAtCommand.
          throw new FatalInputError(
            'Exiting due to an error processing the @ command.',
          );
        }
        query = processedQuery as Part[];
      }

      let currentMessages: Content[] = [{ role: 'user', parts: query }];

      let turnCount = 0;
      while (true) {
        turnCount++;
        if (
          config.getMaxSessionTurns() >= 0 &&
          turnCount > config.getMaxSessionTurns()
        ) {
          handleMaxTurnsExceededError(config);
        }
        const toolCallRequests: ToolCallRequestInfo[] = [];

        const responseStream = geminiClient.sendMessageStream(
          currentMessages[0]?.parts || [],
          abortController.signal,
          prompt_id,
        );

        let responseText = '';
        for await (const event of responseStream) {
          if (abortController.signal.aborted) {
            handleCancellationError(config);
          }

          if (event.type === GeminiEventType.Content) {
            if (config.getOutputFormat() === OutputFormat.JSON) {
              responseText += event.value;
            } else {
              process.stdout.write(event.value);
            }
          } else if (event.type === GeminiEventType.ToolCallRequest) {
            toolCallRequests.push(event.value);
            if (config.getOutputFormat() === OutputFormat.JSON) {
              const output = {
                type: 'tool_call',
                name: event.value.name,
                args: event.value.args,
                callId: event.value.callId,
                timestamp: new Date().toISOString(),
              };
              process.stdout.write(JSON.stringify(output));
              process.stdout.write('\n');
            }
          }
        }

        if (toolCallRequests.length > 0) {
          const toolResponseParts: Part[] = [];
          for (const requestInfo of toolCallRequests) {
            const toolResponse = await executeToolCall(
              config,
              requestInfo,
              abortController.signal,
            );

            if (toolResponse.error) {
              handleToolError(
                requestInfo.name,
                toolResponse.error,
                config,
                toolResponse.errorType || 'TOOL_EXECUTION_ERROR',
                typeof toolResponse.resultDisplay === 'string'
                  ? toolResponse.resultDisplay
                  : undefined,
              );
              if (config.getOutputFormat() === OutputFormat.JSON) {
                const output = {
                  type: 'tool_response',
                  name: requestInfo.name,
                  callId: requestInfo.callId,
                  error: true,
                  message:
                    toolResponse.resultDisplay || toolResponse.error.message,
                  timestamp: new Date().toISOString(),
                };
                process.stdout.write(JSON.stringify(output));
                process.stdout.write('\n');
              }
            }

            if (toolResponse.responseParts) {
              toolResponseParts.push(...toolResponse.responseParts);
              if (config.getOutputFormat() === OutputFormat.JSON) {
                const output = {
                  type: 'tool_response',
                  callId: toolResponse.callId,
                  success: true,
                  result: toolResponse.resultDisplay || '',
                  timestamp: new Date().toISOString(),
                };
                process.stdout.write(JSON.stringify(output));
                process.stdout.write('\n');
              }
            }
          }
          currentMessages = [{ role: 'user', parts: toolResponseParts }];
        } else {
          if (config.getOutputFormat() === OutputFormat.JSON) {
            const formatter = new JsonFormatter();
            const stats = uiTelemetryService.getMetrics();
            process.stdout.write(formatter.format(responseText, stats));
            process.stdout.write('\n');
          } else {
            process.stdout.write('\n'); // Ensure a final newline
          }

          await saveCheckpoint(chat, logger, config.getOutputFormat());
          return;
        }
      }
    } catch (error) {
      handleError(error, config);
    } finally {
      consolePatcher.cleanup();
      if (isTelemetrySdkInitialized()) {
        await shutdownTelemetry(config);
      }
    }
  });
}

async function saveCheckpoint(
  chat: { getHistory: () => Content[] },
  logger: Logger,
  outputFormat: OutputFormat,
): Promise<void> {
  try {
    const history = chat.getHistory();
    if (history.length > 0) {
      const continueTag = `auto-${Date.now()}`;
      await logger.saveCheckpoint(history, continueTag);
      if (outputFormat === OutputFormat.JSON) {
        const output = {
          type: 'checkpoint',
          tag: continueTag,
          timestamp: new Date().toISOString(),
        };
        process.stdout.write(JSON.stringify(output));
        process.stdout.write('\n');
      }
    }
  } catch (error) {
    if (outputFormat !== OutputFormat.JSON) {
      console.error('Failed to save checkpoint: ', error);
    }
  }
}
