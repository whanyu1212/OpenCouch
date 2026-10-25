import { HttpAgent } from "@ag-ui/client";
import type { RunAgentInput } from "@ag-ui/core";

/**
 * An AG-UI agent whose requests carry only the newest message.
 *
 * The server keeps the conversation and reads nothing but the newest message,
 * so sending the rest would only make a long conversation too large to accept.
 */
export class NewestMessageAgent extends HttpAgent {
  /** Send these headers, such as a fresh sign-in token, with the next runs. */
  setAuthHeaders(headers: Record<string, string>): void {
    this.headers = headers;
  }

  protected requestInit(input: RunAgentInput): RequestInit {
    return super.requestInit({ ...input, messages: input.messages.slice(-1) });
  }
}
