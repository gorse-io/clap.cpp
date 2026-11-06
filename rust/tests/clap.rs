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
use clap_cpp::CLAP;
use huggingface_hub::{HFClient, RepoDownloadFileParams};
use more_asserts::{assert_gt, assert_lt};
use std::path::PathBuf;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

#[tokio::test]
async fn clap() -> Result<(), Box<dyn std::error::Error>> {
    let hf = HFClient::new()?;
    let params = RepoDownloadFileParams::builder()
        .filename("larger_clap_general-q4_k_m.gguf")
        .local_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"))
        .build();
    hf.model("gorse-io", "larger_clap_general-GGUF")
        .download_file(&params)
        .await?;

    let mut clap = CLAP::new(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("larger_clap_general-q4_k_m.gguf"),
        1,
    )?;
    let audio_embedding = clap.encode_audio(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/dog.wav"),
        0,
    )?;
    let text_embedding_1 = clap.encode_text("Sound of a dog")?;
    let text_embedding_2 = clap.encode_text("Sound of vaccum cleaner")?;

    let distance_1 = cosine(&audio_embedding, &text_embedding_1);
    let distance_2 = cosine(&audio_embedding, &text_embedding_2);
    // softmax
    let score_1 = distance_1.exp() / (distance_1.exp() + distance_2.exp());
    let score_2 = distance_2.exp() / (distance_1.exp() + distance_2.exp());
    assert_gt!(score_1, 0.99);
    assert_lt!(score_2, 0.01);
    Ok(())
}
