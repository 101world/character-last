# FLUX.1-dev Kohya Training Worker

This repository is a custom RunPod Serverless worker specifically optimized for FLUX.1-dev LoRA training and inference using Kohya's sd-scripts (sd3 branch).

## Features

- **FLUX.1-dev Optimized**: Uses the sd3 branch of Kohya's sd-scripts with proper FLUX.1 parameters
- **Complete Model Setup**: Includes all required models (FLUX.1-dev, CLIP-L, T5-XXL, AE)
- **Optimized Training**: Pre-configured with recommended FLUX.1 training parameters
- **High-Quality Inference**: FLUX.1-dev inference with proper guidance and sampling
- **GPU Optimized**: CUDA 12.4 with PyTorch 2.6.0 for maximum performance

## Model Requirements

The worker automatically downloads and includes:

- **FLUX.1-dev**: Main model (`flux1-dev.safetensors`)
- **CLIP-L**: Text encoder for CLIP-Large (`clip_l.safetensors`)
- **T5-XXL**: Text encoder for T5-XXL (`t5xxl_fp16.safetensors`)
- **AE**: AutoEncoder for FLUX.1 (`ae.safetensors`)

## Training Parameters (FLUX.1-dev Optimized)

- **Guidance Scale**: `1.0` (disabled for training, as recommended)
- **Timestep Sampling**: `flux_shift` (recommended for FLUX.1)
- **Model Prediction Type**: `raw` (recommended for FLUX.1)
- **Discrete Flow Shift**: `3.1582` (for flux_shift sampling)
- **Network Module**: `networks.lora_flux` (FLUX-specific LoRA)
- **Learning Rate**: `1e-4` with constant scheduler

## Input Format

### Training
```json
{
  "input": {
    "mode": "train",
    "train_data": "/workspace/data",
    "output_dir": "/workspace/output"
  }
}
```

### Inference
```json
{
  "input": {
    "mode": "infer",
    "prompt": "a cinematic portrait of a character in a dark blue denim jacket, photorealistic, high detail, 8k"
  }
}
```

## Key Features

- **FLUX.1-dev Specific**: Uses `flux_train_network.py` with FLUX-optimized parameters
- **Memory Efficient**: Gradient checkpointing and text encoder caching enabled
- **Mixed Precision**: FP16 training for optimal performance
- **LoRA Training**: 16-rank LoRA with proper FLUX.1 architecture support
- **Batch Processing**: Configurable batch sizes for different GPU memory

## Technical Details

### PyTorch Version
- **PyTorch**: 2.6.0 (required for FLUX.1 support)
- **CUDA**: 12.4
- **Python**: 3.10+

### Kohya Branch
- **Branch**: `sd3` (FLUX.1 and SD3/SD3.5 support)
- **Script**: `flux_train_network.py` for training
- **Inference**: `flux_minimal_inference.py` for generation

## Deploying to RunPod

1. Push this repo to GitHub
2. Create new Serverless GPU Worker on RunPod
3. Select GPU with at least 16GB VRAM (recommended: 24GB+)
4. Connect your GitHub repo → build → deploy

## Usage Examples

### Training a LoRA
```javascript
const response = await fetch("https://api.runpod.ai/v2/YOUR-ENDPOINT/run", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
  },
  body: JSON.stringify({
    input: {
      mode: "train",
      train_data: "/workspace/data",
      output_dir: "/workspace/output"
    }
  })
})
```

### Generating Images
```javascript
const response = await fetch("https://api.runpod.ai/v2/YOUR-ENDPOINT/run", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
  },
  body: JSON.stringify({
    input: {
      mode: "infer",
      prompt: "masterpiece, best quality, 1girl, in white dress, detailed face, beautiful eyes"
    }
  })
})
```

## Memory Optimization

For different GPU memory sizes:

- **24GB VRAM**: Default settings work optimally
- **16GB VRAM**: Set batch size to 1, use `--blocks_to_swap`
- **12GB VRAM**: Use `--blocks_to_swap 16` and 8bit AdamW
- **10GB VRAM**: Use `--blocks_to_swap 22`, consider FP8 format for T5XXL
- **8GB VRAM**: Use `--blocks_to_swap 28`, FP8 format recommended

## Related Links

- [Kohya sd-scripts (sd3 branch)](https://github.com/kohya-ss/sd-scripts/tree/sd3)
- [FLUX.1 Training Documentation](https://github.com/kohya-ss/sd-scripts/blob/sd3/docs/flux_train_network.md)
- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/overview)
- [FLUX.1-dev Model](https://huggingface.co/black-forest-labs/FLUX.1-dev)