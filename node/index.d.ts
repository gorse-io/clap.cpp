/** Synchronous CPU inference. Each call blocks the calling JavaScript thread. */
export class CLAP {
  /** threads=0 selects the native default; negative/fractional values are rejected. */
  constructor(path: string, threads?: number);
  /** Returns an owned embedding. Text is tokenized internally. */
  encodeText(input: string | Int32Array | number[]): Float32Array;
  /** Mono PCM, automatically resampled. Defaults: 48000 Hz, cropOffset=0 samples. */
  encodeAudio(pcm: Float32Array, sampleRate?: number, cropOffset?: number): Float32Array;
  /** Decode WAV/MP3/FLAC by contents, downmix and resample internally. */
  encodeAudio(path: string, cropOffset?: number): Float32Array;
  /** Idempotent early release; encoding methods throw after close. */
  close(): void;
}
