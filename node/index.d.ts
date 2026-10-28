/** Owned, flat time-major values: values.length === frames * dimension. */
export interface AudioFrames {
  values: Float32Array;
  frames: number;
  dimension: number;
}
export interface Audio {
  samples: Float32Array;
  sampleRate: number;
}
export interface ModelOptions {
  /** Zero selects the native thread default. */
  threads?: number;
  /** Separate audio mmproj GGUF for EmbeddingGemma2 audio encoding. */
  mmproj?: string;
}
/** Synchronous CPU inference. Each call blocks the calling JavaScript thread. */
export class Model {
  /** threads=0 selects the native default; negative/fractional values are rejected. */
  constructor(path: string, threadsOrOptions?: number | ModelOptions);
  readonly projectionDimension: number;
  readonly audioSampleRate: number;
  /** Native tokenizer; maxLength defaults to 512. Returns owned token IDs. */
  tokenize(text: string, maxLength?: number): Int32Array;
  text(input: string | Int32Array | number[]): Float32Array;
  /** Mono PCM. sampleRate defaults to 48000; cropOffset defaults to 0 samples. */
  audio(pcm: Float32Array, sampleRate?: number, cropOffset?: number): Float32Array;
  /** Available only for architectures exposing frame encodings. Default: 48000 Hz. */
  audioFrames(pcm: Float32Array, sampleRate?: number): AudioFrames;
  /** MERT-v1 all-layer mean features for CLaMP3. Default: 24000 Hz. */
  clamp3Features(pcm: Float32Array, sampleRate?: number): AudioFrames;
  /** Idempotent. All other methods/properties throw after close. */
  close(): void;
}
/** WAV/MP3/FLAC decoded by contents, downmixed to mono, original sample rate. */
export function decodeAudio(path: string): Audio;
/** Owned output. targetRate defaults to 48000; both rates must be positive integers. */
export function resampleAudio(pcm: Float32Array, sourceRate: number, targetRate?: number): Float32Array;
