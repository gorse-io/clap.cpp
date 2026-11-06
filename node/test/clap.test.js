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
'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createWriteStream, existsSync, statSync } = require('node:fs');
const path = require('node:path');
const { pipeline } = require('node:stream/promises');
const { downloadFile } = require('@huggingface/hub');
const { CLAP } = require('..');

function cosine(a, b) {
  const dot = a.reduce((sum, x, i) => sum + x * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, x) => sum + x * x, 0));
  const normB = Math.sqrt(b.reduce((sum, x) => sum + x * x, 0));
  return dot / (normA * normB);
}

test('CLAP', async (t) => {
  const filename = 'larger_clap_general-q4_k_m.gguf';
  const modelPath = path.join(__dirname, '..', '..', filename);
  const file = await downloadFile({
    repo: 'gorse-io/larger_clap_general-GGUF',
    path: filename,
    xet: false,
  });
  if (!existsSync(modelPath) || statSync(modelPath).size !== file.size) {
    await pipeline(file.stream(), createWriteStream(modelPath));
  }

  const clap = new CLAP(modelPath, 1);
  t.after(() => clap.close());
  const audioEmbedding = clap.encodeAudio(path.join(__dirname, '..', '..', 'assets', 'dog.wav'), 0);
  const textEmbedding1 = clap.encodeText('Sound of a dog');
  const textEmbedding2 = clap.encodeText('Sound of vaccum cleaner');

  const distance1 = cosine(audioEmbedding, textEmbedding1);
  const distance2 = cosine(audioEmbedding, textEmbedding2);
  // softmax(cosine * exp(logit_scale_a)), using the learned scale from
  // laion/larger_clap_general, as in rust/tests/clap.rs.
  const logitScale = Math.exp(3.6549275);
  const logit1 = distance1 * logitScale;
  const logit2 = distance2 * logitScale;
  const maxLogit = Math.max(logit1, logit2);
  const exp1 = Math.exp(logit1 - maxLogit);
  const exp2 = Math.exp(logit2 - maxLogit);
  const score1 = exp1 / (exp1 + exp2);
  const score2 = exp2 / (exp1 + exp2);
  assert.ok(score1 > 0.99, `Expected dog score > 0.99, got ${score1}`);
  assert.ok(score2 < 0.01, `Expected vacuum cleaner score < 0.01, got ${score2}`);
});
