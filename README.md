# FLUX.1-dev Kohya Training Worker

This repository is a custom RunPod Serverless worker specifically optimized for FLUX.1-dev LoRA training and inference using Kohya's sd-scripts (sd3 branch).

## Features

- **FLUX.1-dev Optimized**: Uses the sd3 branch of Kohya's sd-scripts with proper FLUX.1 parameters
- **Complete Model Setup**: Includes all required models (FLUX.1-dev, CLIP-L, T5-XXL, AE)
- **Advanced Captioning**: BLIP and CLIP-based automatic captioning for character training
- **Cloudflare R2 Integration**: Seamless integration with your web app's Cloudflare R2 storage
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

### Training with Captioning and Cloudflare R2

```json
{
  "input": {
    "mode": "train",
    "train_data": "/workspace/data",
    "output_dir": "/workspace/output",

    // Captioning parameters
    "use_captioning": true,
    "caption_method": "blip",  // "blip", "clip", or "existing"
    "caption_extension": ".txt",
    "max_caption_length": 75,

    // Cloudflare R2 parameters (for your web app integration)
    "use_r2": true,
    "r2_access_key": "your_r2_access_key",
    "r2_secret_key": "your_r2_secret_key",
    "r2_endpoint": "https://your-account-id.r2.cloudflarestorage.com",
    "r2_bucket": "your-bucket-name",
    "r2_prefix": "character-training-data/",

    // Training hyperparameters
    "learning_rate": "1e-4",
    "max_train_steps": "1000",
    "train_batch_size": "1",
    "network_dim": "16",
    "save_every_n_steps": "500"
  }
}
```

### Training Parameters

**Captioning Options:**
- `use_captioning`: Enable/disable automatic captioning (boolean, default: true)
- `caption_method`: Captioning method - "blip", "clip", or "existing" (string, default: "blip")
- `caption_extension`: File extension for captions (string, default: ".txt")
- `max_caption_length`: Maximum caption length for BLIP (number, default: 75)

**Cloudflare R2 Integration:**
- `use_r2`: Enable Cloudflare R2 dataset download (boolean, default: false)
- `r2_access_key`: Your R2 access key (string)
- `r2_secret_key`: Your R2 secret key (string)
- `r2_endpoint`: R2 endpoint URL (string)
- `r2_bucket`: R2 bucket name (string)
- `r2_prefix`: Prefix/path in R2 bucket (string, default: "")

**Training Hyperparameters:**
- `learning_rate`: Learning rate (string, default: "1e-4")
- `max_train_steps`: Maximum training steps (string, default: "1000")
- `train_batch_size`: Batch size (string, default: "1")
- `network_dim`: LoRA network dimension (string, default: "16")
- `save_every_n_steps`: Save checkpoint frequency (string, default: "500")

### Inference
```json
{
  "input": {
    "mode": "infer",
    "prompt": "a cinematic portrait of a character in a dark blue denim jacket, photorealistic, high detail, 8k"
  }
}
```

## Captioning Methods

The worker supports multiple captioning approaches for character training:

### BLIP Captioning (Recommended)
- Uses Salesforce's BLIP large model for detailed image descriptions
- Generates high-quality, descriptive captions automatically
- Best for character training with varied poses and expressions

### CLIP Captioning
- Uses OpenAI's CLIP model for semantic understanding
- Faster processing but less detailed captions
- Good for consistent character features

### Existing Captions
- Use pre-existing caption files in your dataset
- No automatic captioning performed
- Best when you have custom, curated captions

## Cloudflare R2 Integration

The worker can automatically download your character training datasets from Cloudflare R2:

1. **Upload Dataset**: Upload your character images to R2 bucket
2. **Configure Access**: Provide R2 credentials in the training request
3. **Automatic Download**: Worker downloads and processes your dataset
4. **Captioning**: Generates captions if needed
5. **Training**: Trains LoRA on your character data
6. **Results**: Saves trained model back to specified location

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

### Training a Character LoRA with Captioning and R2

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
      output_dir: "/workspace/output",

      // Enable automatic captioning
      use_captioning: true,
      caption_method: "blip",
      caption_extension: ".txt",
      max_caption_length: 75,

      // Cloudflare R2 integration for your web app
      use_r2: true,
      r2_access_key: "your_r2_access_key",
      r2_secret_key: "your_r2_secret_key",
      r2_endpoint: "https://your-account-id.r2.cloudflarestorage.com",
      r2_bucket: "character-datasets",
      r2_prefix: "my-character/photos/",

      // Custom training parameters
      learning_rate: "2e-4",
      max_train_steps: "2000",
      train_batch_size: "2",
      network_dim: "32",
      save_every_n_steps: "250"
    }
  })
})
```

### Training with Existing Captions

```javascript
{
  input: {
    mode: "train",
    train_data: "/workspace/data",
    output_dir: "/workspace/output",

    // Use existing captions
    use_captioning: true,
    caption_method: "existing",
    caption_extension: ".txt",

    // R2 integration
    use_r2: true,
    r2_access_key: "your_r2_access_key",
    r2_secret_key: "your_r2_secret_key",
    r2_endpoint: "https://your-account-id.r2.cloudflarestorage.com",
    r2_bucket: "character-datasets",
    r2_prefix: "my-character/"
  }
}
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