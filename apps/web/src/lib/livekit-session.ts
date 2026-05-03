"use client";

import { AudioPresets, Room, TokenSource } from "livekit-client";
import {
  createLiveKitVoiceToken,
  type AssistantVoiceOption,
  type LiveKitVoiceTokenResponse,
  type TranscriptionLanguageOption,
  type VoiceMemoryMode,
} from "@/lib/api";

export const LIVEKIT_AUDIO_CAPTURE_DEFAULTS = {
  autoGainControl: false,
  channelCount: 1,
  echoCancellation: true,
  noiseSuppression: false,
  voiceIsolation: false,
} as const;

export function createOpenCouchVoiceRoom(): Room {
  return new Room({
    audioCaptureDefaults: LIVEKIT_AUDIO_CAPTURE_DEFAULTS,
    publishDefaults: {
      audioPreset: AudioPresets.speech,
      forceStereo: false,
    },
  });
}

export function createOpenCouchVoiceTokenSource(
  userId: string,
  threadId: string,
  transcriptionLanguage: TranscriptionLanguageOption,
  memoryMode: VoiceMemoryMode,
  assistantVoice: AssistantVoiceOption,
  onTokenCreated?: (token: LiveKitVoiceTokenResponse) => void
) {
  return TokenSource.custom(async () => {
    const token = await createLiveKitVoiceToken(
      userId,
      threadId,
      transcriptionLanguage,
      memoryMode,
      assistantVoice
    );
    onTokenCreated?.(token);
    return {
      serverUrl: token.server_url,
      participantToken: token.participant_token,
    };
  });
}
