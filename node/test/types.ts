import { Model, decodeAudio, resampleAudio, AudioFrames } from '..';
const model = new Model('model.gguf', 1);
const ids: Int32Array = model.tokenize('hello');
const embedding: Float32Array = model.text(ids);
model.text('hello'); model.text([0, 2]);
const decoded = decodeAudio('clip.wav');
const pcm: Float32Array = resampleAudio(decoded.samples, decoded.sampleRate);
const frames: AudioFrames = model.audioFrames(pcm);
model.audio(pcm, 48000, 0); model.clamp3Features(pcm, 24000);
const dimension: number = model.projectionDimension;
const sampleRate: number = model.audioSampleRate;
model.close();
// @ts-expect-error read-only native property
model.projectionDimension = 2;
// @ts-expect-error PCM must have F32 storage
model.audio(new Float64Array(1));
// @ts-expect-error tokens cannot be strings
model.text(['hello']);
void [embedding, frames, dimension, sampleRate];
