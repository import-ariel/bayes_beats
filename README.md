# Bayes’ Beats: Generating Music from Image Vibes

---

## Abstract

*Bayes’ Beats is a generative model that creates short music clips based on the “vibe” of a randomly selected image. Using a LoRA-fine-tuned ACE-Step model, we translate text descriptions of images into coherent musical outputs. To control the diversity and coherence of generated music, we introduce a Bayesian prior over the model’s latent space using a variational inference layer. Our training leverages the 1-Million-Pairs-Image-Caption dataset for images and the MTG-Jamendo dataset for music, focusing on mood and emotion alignment. This approach enables controllable, expressive music generation from visual input.*

---

## Introduction & Motivation

- Overview of the problem space: why image-to-music?
- Motivation for using Bayesian Machine Learning in generative modeling.
- Use cases for mood-based music generation.

---

## Problem Statement

- Formal statement of mapping image "vibes" to music.
- Challenges: coherence, diversity, controllability, lyric alignment.


---

## Project Overview

### Architecture Summary

- High-level diagram and textual summary of project design.

### End-to-End Pipeline

- Description of input, feature extraction, model inference, and output generation.

---

## Datasets

### Image Dataset: 1-Million-Pairs-Image-Caption

- Description, size, relevance to project, sample structure.

### Music Dataset: MTG-Jamendo

- Description, tags/moods, why suitable for ACE-STEP, data structure.

### Data Preprocessing
We quickly realized our dataset was over 50GB; to avoid turning this into too much of a Big Data problem, we randomly sampled 284 songs from the dataset and uploaded them in a Google Cloud Storage (GC) bucket. 

The ACE-Step developers created a script that converted the data into the correct format for fine-tuning. However, our raw files were not compatible with the script's ingestion; for starters, **we needed labeled vibes from our songs**!

Thus, we did the following: 

*Song Selection and Sampling*
- Randomly sampled a fixed number of tracks (20 per genre) to ensure class balance and manageable dataset size
- Limited total dataset to 200 samples for memory efficiency and to fit Colab VRAM

*Audio Download and Standardization*
- Downloaded all sampled MP3 files from GCS to the local environment in parallel
- Cropped every audio file to a maximum duration of 5 seconds at a fixed sample rate (e.g., 48,000 Hz)
- Padded or truncated audio files as needed to maintain uniform length across the dataset

*Audio Feature Extraction and Embedding*
- Extracted rich audio features from each track (e.g., tempo, spectral centroid, MFCCs, chroma, RMS energy, zero-crossing rate) using `librosa`
- Computed Mel spectrograms as the main deep learning input
- Encoded each Mel spectrogram into a latent representation using a Deep Convolutional Autoencoder (DCAE)

*Prompt & Metadata Generation*
- Used OpenAI GPT to generate short, atmospheric "vibe" descriptions for each song based on extracted features
- Embedded these descriptions into fixed-length vectors using a pretrained SentenceTransformer model
- Recorded genre and filename metadata for downstream conditioning

*Dataset Curation and Filtering*
- Filtered out audio files that could not be processed, were corrupt, or exceeded the allowed duration
- Reorganized and reformatted metadata into the ACE-Step input format, including prompt and lyrics placeholder files for each MP3

*Final Dataset Preparation*
- Assembled all relevant data (audio latents, prompt embeddings, metadata) into a unified JSON and Pandas DataFrame
- Converted the processed dataset into Hugging Face `Dataset` format and saved to disk for training
- Conducted exploratory data analysis (genre counts, feature distributions, audio duration validation, t-SNE latent space visualization)

#### Double-click: "Vibe Generation"
Because ChatGPT 4o is trained on vast text corpora, we made an assumption that it would be able to generate descriptions of a song using: 
- tempo and rhythm to infer whether the track is energetic or relaxed.
- spectral and MFCC statistics to guess whether the track is bright, dark, warm, or cold.
- genre and extracted features to match with typical cultural or emotional associations.

Is this perfect? **Absolutely not.** In fact, the authors of ACE-Step recommended labelling songs through Qwen, Alibaba's LLM, which can ingest MP3 files. They even included a prompt they used to label their songs for training. 

In the future, this is a great approach; however, the developer working on the script was intimately familiar with the OpenAI API and had credits expiring at the end of the month :) For a production-ready system, other tagging mechanisms would be deployed and comparatively evaluated. 

#### The Final Dataset
Our final HuggingFace dataset looked like this: 
Dataset({
    features: ['keys', 'filename', 'tags', 'speaker_emb_path', 'norm_lyrics', 'recaption'],
    num_rows: 200
})

We fed this dataset, alongside the sampled audio files, to ACE-Step when fine-tuning the model. 

---

## Model Architecture

### ACE-STEP Foundation

Our foundational model is [ACE-STEP](https://github.com/ace-step/ACE-Step?tab=readme-ov-file#-features), a new model as of May 2025 that generates music from text inputs. It is a diffusion model but is complemented by a  deep-compression auto-encoder and linear transformer.

We fine-tuned this model on MTG-JAMENDO, a large dataset of annotated tracks that works has mood-specific labels. The goal of fine-tuning is to train ACE-STEP to be atuned to capturing "moods" in the songs it outputs. We use Low Rank Approximation methods, and only fine-tune the weights associated with the last generative layer.  

### LoRA Fine-Tuning Approach

Fine-tuning was a surprisingly difficult task, although this shouldn't have been surprising in hindsight:

| Module Name          | Type/Class         | Trainable Params | Mode  |
| -------------------- | ------------------ | ---------------- | ----- |
| `transformers`       | `PeftModel`        | 3.3B             | train |
| `dcae`               | `MusicDCAE`        | 259M             | eval  |
| `text_encoder_model` | `UMT5EncoderModel` | 281M             | eval  |
| `mert_model`         | `MERTModel`        | 315M             | eval  |
| `resampler_mert`     | `Resample`         | 0                | train |
| `hubert_model`       | `HubertModel`      | 94.4M            | eval  |
| `resampler_mhubert`  | `Resample`         | 0                | train |

16.4 M    Trainable params
3.3 B     Total params

To avoid re-training the 3.3B parameters, we chose a fine-tuning approach called LoRA, or "Low-Rank Adaption." Why?

- LoRA keeps the **pre-trained model weights frozen**.
- Instead of updating the full weight matrices (which are large and dense), LoRA **injects small trainable matrices** (called low-rank adapters) into certain layers (typically the attention layers) of the Transformer architecture.
- The update to a weight matrix \( W_0 \) is expressed as:
  \[
  W = W_0 + \Delta W = W_0 + BA
  \]
  Where \( A \) and \( B \) are low-rank matrices (with rank \( r \ll d \), where \( d \) is the original size).

During training and backpropagation: 
- Only the **small matrices \( A \) and \( B \) are updated during backpropagation**.
- The original weights \( W_0 \) are not updated (remain frozen).
- This reduces GPU memory usage since gradients and optimizer states are only kept for a tiny subset of parameters.

But we still faced huge issues when using this technique to fine-tune ACE-Step...

The first, and most time consuming issue, was **out of memory** errors. Because we were running this script on Google Colab, we only had access to 40 GB of VRAM. 
To account for these errors, we: 

1. Drastically reduced our dataset (took 5 second random samples from 200 songs)
2. Limited batch size to 1, and set accumulate_grad_batches to offset this
3. Only fine-tuned lightweight LoRA/adapter layers, keeping backbone models in eval mode to save VRAM.
4. Reduced the sequence length and feature dimensions in both the dataset and model configuration
5. Turned of all unnecssary logging and callbacks (this will bite us later)
6. Regularly checked nvidia-smi/Colab VRAM usage and cleared the cache (torch.cuda.empty_cache()) between runs when necessary.
7. Restarted the runtime as needed

Even with all of these measures, we still consumed ~ 38 GB/40 GB during fine-tuning!

Our troubles also didn't end here. We found the out-of-the-box tools ACE-Step's developers provided did not run smoothly. Issues we encountered and remediated included:

1. The Hugging Face dataset loader was tightly coupled to Google Cloud Storage; we rewrote data loading logic to support local files and small datasets.
2. The default data preprocessing pipeline assumed multi-channel audio with long sequences; we adjusted it to work with our short, mono/stereo 5-second clips.
3. The model’s collate function created unnecessarily large tensors; we modified the function to pad/stack only as needed, keeping memory use minimal.
4. Batch/sequence shape mismatches in the DCAE and downstream models triggered cryptic runtime errors; we debugged and aligned input/output shapes across the data pipeline and model forward passes.
5. The training script’s defaults expected multi-GPU environments; we adjusted configs to ensure stable single-GPU and CPU operation.
6. The plot/visualization callbacks were not compatible with Colab’s resource limits; we disabled them and moved any required plots to after training or local analysis.
7. Several model components were left in train mode by default (even when not being updated), consuming extra VRAM; we set unused modules to eval mode to conserve memory.
8. Loss/metric tracking used excessive memory with large history buffers; we limited logging intervals and kept metrics concise.
9. Frequent PyTorch and library version incompatibilities (especially with Lightning and transformers) caused subtle bugs; we pinned package versions and patched code for compatibility.

After 12+ hours of work, not including training time (that would be too easy!), our final model params were: 

- `learning_rate`: `1e-4`
- `num_workers`: `2`
- `accumulate_grad_batches`: `8`
- `devices`: `1`
- `precision`: `16`
- `max_steps`: `5000`
- `every_plot_step`: `1000000`
- `every_n_train_steps`: `500`
  
### Linear Transformer for Musical Coherence

- How this approach maintains alignment and coherence across generated music.

### Bayesian Prior over Latent Space

#### Variational Inference Layer

- Theoretical role and practical implementation.

#### Multivariate Gaussian Prior

- Initial assumption, adaptation using evidence from data.

---

## Mathematical Formulation

### Latent Variable Model

- Definition of observed and latent variables.

### Bayesian Priors and Regularization

- Explanation of prior choice and its impact on learning.

### Evidence Lower Bound (ELBO)

- Mathematical loss formulation.

### Variational Inference Derivation

- Equations, figures, and technical explanation of inference process.

---

## Implementation Details

### Software Stack & Libraries

- List of libraries, frameworks, and environments used.

### Training Procedure

- Data splits, batching, optimizer, and checkpointing.

### Hyperparameters

- Summary table or bullet list.

### Fine-Tuning with Jamendo

- Specifics on the music dataset integration.

### Music Generation Pipeline

- Detailed flow from image to music output.

### Visualization and Evaluation Tools

- Tools used for analysis and visualization.

---

## Challenges & Solutions

### Challenge 1: Ambiguous Image-to-Audio Mapping

- Solution: Bayesian regularization and variational inference.

### Challenge 2: Coherence vs. Diversity in Generated Music

- Solution: Model/architecture tuning and hyperparameter adjustment.

### Challenge 3: Data Alignment and Preprocessing

- Solution: Specific preprocessing and alignment strategies.

### Challenge 4: Model Training Stability

- Solution: Training tricks and stability enhancements.

### Other Challenges & Mitigations

- Any additional technical or organizational obstacles.

---

## Results

### Qualitative Results: Sample Generations

- Embedded links, sound files, or spectrograms.

### Quantitative Metrics

- Objective metrics and scores, if applicable.

### User Study (if applicable)

- Description and results.

### Ablation Studies

- Analysis of architecture/hyperparameter changes.

---

## Discussion

### Interpretation of Results

- Insights from results and their meaning.

### Limitations

- Data/model/generalizability limitations.

### Future Work

- Extensions, alternative approaches, scalability.

---

## Conclusion

- Final summary and impact.

---

## References

Hu, E., Shen, Y., Wallis, P., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://github.com/microsoft/LoRA)

---

## Appendix

### Figures & Technical Diagrams

- All diagrams and technical illustrations.

### Additional Mathematical Derivations

- Extended derivations as needed.

### Instructions for Reproducibility

- Environment setup, running code, reproducing results.

