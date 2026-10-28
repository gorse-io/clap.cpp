import { CLAP } from '..';
const model = new CLAP('model.gguf', 1);
const embedding: Float32Array = model.encodeText('hello');
model.encodeText([0, 2]);
model.encodeText(new Int32Array([0, 2]));
model.encodeAudio(new Float32Array(24000), 24000, 0);
model.encodeAudio('clip.wav');
model.encodeAudio('clip.flac', 0);
model.close();
// @ts-expect-error PCM must have F32 storage
model.encodeAudio(new Float64Array(1));
// @ts-expect-error tokens cannot be strings
model.encodeText(['hello']);
// @ts-expect-error old method is not public
model.text('hello');
// @ts-expect-error tokenizer is not public
model.tokenize('hello');
// @ts-expect-error model metadata is not public
model.projectionDimension;
// @ts-expect-error frame encodings are not public
model.audioFrames(new Float32Array(1));
// @ts-expect-error utility is not exported
import { decodeAudio } from '..';
// @ts-expect-error old class is not exported
import { Model } from '..';
void embedding;
