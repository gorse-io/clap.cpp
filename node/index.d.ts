/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
export class CLAP {
  /** threads=0 selects the native default; negative/fractional values are rejected. */
  constructor(path: string, threads?: number);
  /** Returns an owned embedding. Text is tokenized internally. */
  encodeText(input: string | Int32Array | number[]): Float32Array;
  /** Mono PCM, automatically resampled. Defaults: 48000 Hz, cropOffset=0 samples. */
  encodeAudio(pcm: Float32Array, sampleRate?: number, cropOffset?: number): Float32Array;
  /** Decode WAV/MP3/FLAC by contents, downmix and resample internally. */
  encodeAudio(path: string, cropOffset?: number): Float32Array;
  /** Concurrent same-model inference off the JS thread. Inputs are snapshotted. */
  encodeTextAsync(input: string | Int32Array | number[]): Promise<Float32Array>;
  /** Concurrent mono PCM inference. Defaults: 48000 Hz, cropOffset=0 samples. */
  encodeAudioAsync(pcm: Float32Array, sampleRate?: number, cropOffset?: number): Promise<Float32Array>;
  /** Concurrent WAV/MP3/FLAC decoding and inference. */
  encodeAudioAsync(path: string, cropOffset?: number): Promise<Float32Array>;
  /** Idempotent close: rejects new work; inflight operations finish before native release. */
  close(): void;
}
