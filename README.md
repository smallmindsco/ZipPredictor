# ZipPredictor: Learning to Predict Zip File Outputs from Model Weights

## Overview

ZipPredictor (Model K) is a neural architecture designed to learn from a source model M (stored as `.safetensors`) and produce zip file outputs that, when unzipped, match what model M would generate for a given text prompt.

## Problem Analysis

### The Challenge
Given:
- A source model M (e.g., a text-to-image model, code generator, etc.)
- M's weights in `.safetensors` format

Goal: Train model K such that:
```
K(prompt) → zip_file
unzip(zip_file) ≈ M(prompt)
```

### Key Difficulties
1. **Binary output**: Zip files are compressed binary data
2. **Variable length**: Output size varies dramatically with content
3. **Structural constraints**: Zip format has headers, CRCs, compression
4. **Semantic gap**: Text input → complex binary output

## Architecture Options

### Option A: Two-Stage Generator (Recommended)
```
Prompt → [Encoder] → [Content Generator] → [Structured Output] → [Zip Encoder] → Zip File
```
- First generate the logical content
- Then serialize and compress

### Option B: Direct Byte Generation
```
Prompt → [Encoder] → [Autoregressive Decoder] → Raw Bytes → Zip File
```
- Generate zip bytes directly (very hard)

### Option C: Latent Space Approach
```
Prompt → [Encoder] → [Latent Predictor] → [Zip VAE Decoder] → Zip File
```
- Learn a compressed representation of zip files

## Recommended Architecture: Two-Stage Generator

See `model.py` for the complete implementation.

## Installation

```bash
pip install torch safetensors transformers einops
```

## Usage

```python
from model import ZipPredictorModel
from trainer import ZipPredictorTrainer
from data import create_training_data

# 1. Load source model M and generate training data
training_data = create_training_data(
    model_path="path/to/model.safetensors",
    prompts=["prompt1", "prompt2", ...],
    output_dir="training_data/"
)

# 2. Initialize ZipPredictor (model K)
model_k = ZipPredictorModel(
    vocab_size=32000,
    d_model=1024,
    n_heads=16,
    n_layers=12,
    max_content_length=65536,
)

# 3. Train
trainer = ZipPredictorTrainer(model_k)
trainer.train(training_data, epochs=100)

# 4. Inference
zip_bytes = model_k.generate("your prompt here")
with open("output.zip", "wb") as f:
    f.write(zip_bytes)
```

## File Structure

```
zip_predictor/
├── README.md           # This file
├── model.py            # Core model architecture
├── trainer.py          # Training loop and losses
├── data.py             # Data loading and preprocessing
├── zip_utils.py        # Zip file encoding/decoding utilities
└── inference.py        # Inference and generation code
```
