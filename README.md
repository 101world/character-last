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

## Character Training Parameters (FluxGym Style)

**Character Name:**
- `character_name`: Name of the character (used for output file naming)
- Example: `"anya_forger"`, `"luffy"`, `"character_x"`

**Character Trigger Word:**
- `character_trigger`: Unique word to activate character features
- Example: `"anya person"`, `"luffy style"`, `"charx"`
- **Important**: This gets added to ALL training image captions

**Captioning Process:**
1. Upload character images to Cloudflare R2
2. Worker downloads images automatically
3. Generates captions: `"{character_trigger}, {blip_caption}"`
4. Example: `"anya person, a young girl with pink hair"`

**Training Workflow:**
1. **Input**: Character images + trigger word
2. **Captioning**: Auto-generates captions with trigger word
3. **Training**: LoRA learns character features from trigger word
4. **Output**: `character_name_lora.safetensors`

**Sample Generation During Training:**
- Uses trigger word in sample prompts to show progress
- Example: `"anya person in a school uniform"`
- Helps monitor LoRA learning progress

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
- `CLOUDFLARE_R2_ACCESS_KEY_ID`: Your R2 access key (string)
- `CLOUDFLARE_R2_SECRET_ACCESS_KEY`: Your R2 secret key (string)
- `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare account ID (string)
- `R2_BUCKET_NAME`: R2 bucket name (string)
- `r2_prefix`: Prefix/path in R2 bucket (string, default: "kohya/Dataset/riya_bhatu_v1/Character/")

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

1. **Create Serverless Worker**:
   - Go to RunPod → Serverless → Create Worker
   - Select GPU with at least 24GB VRAM (RTX 3090/4090 or A100 recommended)

2. **Connect Repository**:
   - Repository: `https://github.com/101world/character-last`
   - Branch: `main`
   - Build Command: `pip install -r requirements.txt`

3. **Test R2 Integration** (Optional):
   ```bash
   python test_r2_integration.py
   ```
   This will verify your Cloudflare R2 credentials and upload/download functionality.

4. **Build the Docker Image**:
   ```bash
   # Option 1: Use the build script (recommended)
   ./build.sh

   # Option 2: Build manually with multi-stage optimization
   docker build -t flux-kohya-worker .

   # Option 3: Build with specific CUDA architecture (for older GPUs)
   docker build --build-arg CUDA_ARCHITECTURES=75 -t flux-kohya-worker .
   ```
   **Note**: The multi-stage build may take 20-30 minutes. The builder stage downloads models and dependencies, while the runtime stage creates a minimal production image.

## Multi-Stage Build Architecture

The Dockerfile uses a modern multi-stage build approach:

### Builder Stage (`nvidia/cuda:12.4.0-devel-ubuntu20.04`)
- Downloads all FLUX.1-dev models (20GB+)
- Installs build dependencies and Python packages
- Clones Kohya and FluxGym repositories
- Creates optimized virtual environment

### Runtime Stage (`nvidia/cuda:12.4.0-runtime-ubuntu20.04`)
- Minimal Ubuntu 20.04 with CUDA runtime
- Only essential runtime dependencies
- Pre-installed models and repositories from builder
- Optimized for production deployment

### Benefits:
- **Faster builds**: Model downloads cached in builder stage
- **Smaller images**: Runtime image excludes build tools
- **Better caching**: Dependencies isolated from source code changes
- **Production ready**: Follows ML container best practices

## Troubleshooting Build Issues

### Common Build Problems

**Multi-stage build failures:**
```bash
# Clear Docker cache and rebuild
docker system prune -a
docker build --no-cache -t flux-kohya-worker .

# Check build logs for specific stage failures
docker build --progress=plain -t flux-kohya-worker .
```

**Model download failures:**
- The builder stage includes progress indicators for downloads
- If downloads fail, check internet connection and retry
- Models are cached after first successful download

**CUDA compatibility issues:**
- Ensure your system has NVIDIA drivers compatible with CUDA 12.4
- For RunPod deployment, use GPUs with Compute Capability 7.5+
- Check GPU compatibility: `nvidia-smi --query-gpu=compute_cap --format=csv`

**Disk space issues:**
```bash
# Check available space (need at least 50GB free)
df -h

# Clear Docker cache and system
docker system prune -a
docker volume prune -f
```

**Memory issues during build:**
```bash
# Increase Docker memory limit if building locally
# Or use RunPod's cloud build environment
```

### Testing the Build Locally

```bash
# Build the image
docker build -t flux-kohya-worker .

# Test CUDA availability
docker run --rm --gpus all flux-kohya-worker python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Test model loading
docker run --rm --gpus all flux-kohya-worker python3 -c "import torch; from safetensors import safe_open; print('Models accessible')"
```

### Build Optimization Tips

- **Use BuildKit**: Enable Docker BuildKit for faster builds
  ```bash
  export DOCKER_BUILDKIT=1
  docker build -t flux-kohya-worker .
  ```

- **Layer Caching**: The multi-stage build maximizes Docker layer caching
- **Parallel Downloads**: Models download in parallel where possible
- **Dependency Optimization**: Python packages installed with `--no-cache-dir`

### Health Checks

The container includes built-in health checks:
- CUDA availability verification
- Model file integrity checks
- Memory and GPU utilization monitoring

Monitor health status:
```bash
docker ps
# Look for "healthy" status in STATUS column
```
   - Branch: `main`
   - Handler: `handler.py`
   - Docker Context: `/`

3. **Set Environment Variables**:
   ```
   CLOUDFLARE_ACCOUNT_ID=your_account_id
   CLOUDFLARE_R2_ACCESS_KEY_ID=your_access_key
   CLOUDFLARE_R2_SECRET_ACCESS_KEY=your_secret_key
   R2_BUCKET_NAME=your_bucket_name
   HUGGINGFACE_TOKEN=your_huggingface_token
   ```

4. **Configure Timeouts** (Critical for FLUX training):
   - **Execution Timeout**: 7200 seconds (2 hours) - for long training jobs
   - **Startup Timeout**: 600 seconds (10 minutes) - for model loading
   - **HTTP Timeout**: 7200 seconds (2 hours) - for API calls

5. **Deploy**: Monitor the build process (30-60 minutes for initial setup)

### Training Time Estimates:
- **Small Dataset** (10-50 images): 30-60 minutes
- **Medium Dataset** (50-200 images): 1-3 hours
- **Large Dataset** (200+ images): 3-6+ hours

**⚠️ Important**: Set execution timeout longer than your expected training time!

## Usage Examples

### Training a Character LoRA (FluxGym Style)

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

      // Character information (FluxGym style)
      character_name: "anya_forger",
      character_trigger: "anya person",

      // Cloudflare R2 integration
      use_r2: true,
      CLOUDFLARE_R2_ACCESS_KEY_ID: "ef926435442c79cb22a8397939f3f878",
      CLOUDFLARE_R2_SECRET_ACCESS_KEY: "da8c672469940a0b338d86c65b386fc7fe933549706e3aff10ce6d570ec82eb3",
      CLOUDFLARE_ACCOUNT_ID: "ced616f33f6492fd708a8e897b61b953",
      R2_BUCKET_NAME: "the-social-twin-storage",
      r2_prefix: "character_images/anya/",

      // Automatic captioning with trigger word
      use_captioning: true,
      caption_method: "blip",

      // Training parameters
      learning_rate: "1e-4",
      max_train_steps: "1000",
      network_dim: "16"
    }
  })
})
```

**What happens:**
1. Downloads character images from R2
2. Generates captions: `"anya person, [BLIP description]"`
3. Trains LoRA that learns to respond to `"anya person"`
4. Saves as `anya_forger_lora.safetensors`

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
    CLOUDFLARE_R2_ACCESS_KEY_ID: "ef926435442c79cb22a8397939f3f878",
    CLOUDFLARE_R2_SECRET_ACCESS_KEY: "da8c672469940a0b338d86c65b386fc7fe933549706e3aff10ce6d570ec82eb3",
    CLOUDFLARE_ACCOUNT_ID: "ced616f33f6492fd708a8e897b61b953",
    R2_BUCKET_NAME: "the-social-twin-storage",
    r2_prefix: "kohya/Dataset/riya_bhatu_v1/Character/"
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
      prompt: "masterpiece, best quality, 1girl, in white dress, detailed face, beautiful eyes",

      // Optional: Upload generated image to R2
      use_r2: true,
      CLOUDFLARE_R2_ACCESS_KEY_ID: "ef926435442c79cb22a8397939f3f878",
      CLOUDFLARE_R2_SECRET_ACCESS_KEY: "da8c672469940a0b338d86c65b386fc7fe933549706e3aff10ce6d570ec82eb3",
      CLOUDFLARE_ACCOUNT_ID: "ced616f33f6492fd708a8e897b61b953",
      R2_BUCKET_NAME: "the-social-twin-storage",
      r2_prefix: "kohya/Dataset/riya_bhatu_v1/Character/"
    }
  })
})
```

## Accessing Your Outputs

When using Cloudflare R2 integration, your trained models and generated images are automatically uploaded to your R2 bucket:

### Trained LoRA Models
- **Location**: `kohya/Dataset/riya_bhatu_v1/Character/trained_models/`
- **Files**: 
  - `lora.safetensors` - Your trained LoRA weights
  - `lora.safetensors.toml` - Model configuration
  - `last.safetensors` - Final checkpoint
- **Access URL**: `https://ced616f33f6492fd708a8e897b61b953.r2.cloudflarestorage.com/kohya/Dataset/riya_bhatu_v1/Character/trained_models/lora.safetensors`

### Generated Images
- **Location**: `kohya/Dataset/riya_bhatu_v1/Character/generated_images/`
- **Files**: `generated_[timestamp].png` - Timestamped generated images
- **Access URL**: `https://ced616f33f6492fd708a8e897b61b953.r2.cloudflarestorage.com/kohya/Dataset/riya_bhatu_v1/Character/generated_images/`

### Response Format
The worker returns R2 URLs in the response:
```json
{
  "status": "training complete",
  "lora_path": "/workspace/output/lora.safetensors",
  "r2_url": "https://ced616f33f6492fd708a8e897b61b953.r2.cloudflarestorage.com/kohya/Dataset/riya_bhatu_v1/Character/trained_models/lora.safetensors"
}
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