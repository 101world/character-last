import runpod
import subprocess
import os
import glob
import time
import boto3
import json
import logging
import sys
from huggingface_hub import snapshot_download, login
from botocore.client import Config
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import open_clip
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def health_check():
    """Perform startup health checks for models and dependencies"""
    logger.info("Performing startup health checks...")

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("CUDA not available, falling back to CPU")

    # Check model files
    required_models = [
        "/workspace/models/flux1-dev.safetensors",
        "/workspace/models/clip_l.safetensors",
        "/workspace/models/t5xxl_fp16.safetensors",
        "/workspace/models/ae.safetensors"
    ]

    missing_models = []
    for model_path in required_models:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
        else:
            file_size = os.path.getsize(model_path) / (1024 * 1024 * 1024)  # GB
            logger.info(f"Model found: {model_path} ({file_size:.2f} GB)")

    if missing_models:
        logger.error(f"Missing required models: {missing_models}")
        return False

    # Check Kohya installation
    kohya_path = "/workspace/kohya"
    if not os.path.exists(kohya_path):
        logger.error("Kohya sd-scripts not found")
        return False

    flux_train_script = "/workspace/kohya/flux_train_network.py"
    if not os.path.exists(flux_train_script):
        logger.error("FLUX training script not found")
        return False

    logger.info("All health checks passed!")
    return True

def setup_cloudflare_r2(access_key, secret_key, endpoint_url, bucket_name):
    """Setup Cloudflare R2 client for dataset access with error handling"""
    try:
        if not all([access_key, secret_key, endpoint_url, bucket_name]):
            raise ValueError("Missing required Cloudflare R2 credentials")

        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            config=Config(signature_version='s3v4')
        )

        # Test connection
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Successfully connected to R2 bucket: {bucket_name}")
        return s3_client, bucket_name

    except Exception as e:
        logger.error(f"Failed to setup Cloudflare R2 client: {e}")
        raise

def download_dataset_from_r2(s3_client, bucket_name, r2_prefix, local_dir):
    """Download dataset from Cloudflare R2 with error handling and progress tracking"""
    try:
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f"Starting download from R2 prefix: {r2_prefix}")

        total_files = 0
        downloaded_files = 0

        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=r2_prefix):
            for obj in page.get('Contents', []):
                total_files += 1

        logger.info(f"Found {total_files} files to download")

        for page in paginator.paginate(Bucket=bucket_name, Prefix=r2_prefix):
            for obj in page.get('Contents', []):
                try:
                    local_path = os.path.join(local_dir, obj['Key'].replace(r2_prefix, '').lstrip('/'))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    s3_client.download_file(bucket_name, obj['Key'], local_path)
                    downloaded_files += 1

                    file_size = obj['Size'] / (1024 * 1024)  # MB
                    logger.info(f"Downloaded ({downloaded_files}/{total_files}): {obj['Key']} ({file_size:.2f} MB)")

                except Exception as e:
                    logger.error(f"Failed to download {obj['Key']}: {e}")
                    continue

        logger.info(f"Successfully downloaded {downloaded_files}/{total_files} files")

    except Exception as e:
        logger.error(f"Failed to download dataset from R2: {e}")
        raise

def monitor_resources():
    """Monitor GPU memory and system resources"""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        logger.info(f"GPU Memory - Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB, Total: {gpu_memory_total:.2f} GB")
    else:
        logger.info("GPU not available for monitoring")

def track_training_progress(output_dir, max_steps):
    """Track training progress by monitoring output files"""
    import time
    start_time = time.time()
    
    while True:
        try:
            # Check for checkpoint files
            checkpoints = glob.glob(os.path.join(output_dir, "*.safetensors"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                checkpoint_size = os.path.getsize(latest_checkpoint) / (1024 * 1024)  # MB
                logger.info(f"Latest checkpoint: {os.path.basename(latest_checkpoint)} ({checkpoint_size:.2f} MB)")
            
            # Check for log files
            log_files = glob.glob(os.path.join(output_dir, "*.log"))
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        logger.info(f"Training log: {lines[-1].strip()}")
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Error tracking training progress: {e}")
            break

def validate_training_params(input_data):
    """Validate training parameters and set defaults"""
    required_params = ['mode']
    for param in required_params:
        if param not in input_data:
            raise ValueError(f"Missing required parameter: {param}")

    # Set defaults
    defaults = {
        'train_data': '/workspace/data',
        'output_dir': '/workspace/output',
        'learning_rate': '1e-4',
        'max_train_steps': '1000',
        'train_batch_size': '1',
        'network_dim': '16',
        'save_every_n_steps': '500',
        'use_captioning': True,
        'caption_method': 'blip',
        'caption_extension': '.txt',
        'max_caption_length': 75
    }

    for key, default_value in defaults.items():
        if key not in input_data:
            input_data[key] = default_value
            logger.info(f"Using default value for {key}: {default_value}")

    return input_data

def generate_captions_blip(image_dir, caption_extension=".txt", max_length=75, character_trigger=None):
    """Generate captions using BLIP model with character trigger word"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)
                caption_path = os.path.splitext(image_path)[0] + caption_extension

                if os.path.exists(caption_path):
                    continue  # Skip if caption already exists

                try:
                    image = Image.open(image_path).convert('RGB')
                    inputs = processor(image, return_tensors="pt").to(device)

                    with torch.no_grad():
                        output = model.generate(**inputs, max_length=max_length)

                    # Get base caption from BLIP
                    base_caption = processor.decode(output[0], skip_special_tokens=True)

                    # Add character trigger to caption (FluxGym style)
                    if character_trigger:
                        caption = f"{character_trigger}, {base_caption}"
                    else:
                        caption = base_caption

                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)

                    print(f"Generated caption: {caption}")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

def generate_captions_clip(image_dir, caption_extension=".txt", character_trigger=None):
    """Generate captions using CLIP model with character trigger word"""
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)
                caption_path = os.path.splitext(image_path)[0] + caption_extension

                if os.path.exists(caption_path):
                    continue

                try:
                    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_features = model.encode_image(image)
                        image_features /= image_features.norm(dim=-1, keepdim=True)

                    # Create caption with character trigger (FluxGym style)
                    if character_trigger:
                        caption = f"{character_trigger}, a high quality character image"
                    else:
                        caption = "a high quality character image"

                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)

                    print(f"Generated caption: {caption}")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

def upload_training_results_to_r2(s3_client, bucket_name, output_dir, r2_prefix):
    """Upload training results to Cloudflare R2"""
    results_prefix = f"{r2_prefix.rstrip('/')}/training_results/"

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Create R2 key by replacing local path with results prefix
            r2_key = results_prefix + os.path.relpath(local_path, output_dir)

            try:
                s3_client.upload_file(local_path, bucket_name, r2_key)
                logger.info(f"Uploaded training result: {r2_key}")
            except Exception as e:
                logger.error(f"Failed to upload {local_path}: {e}")

def handler(event):
    """Main handler function with comprehensive error handling"""
    try:
        logger.info("Starting FLUX.1-dev training worker")
        logger.info(f"Event: {json.dumps(event, indent=2)}")

        # Perform health check
        if not health_check():
            return {"error": "Health check failed", "status": "failed"}

        input_data = event.get("input", {})
        mode = input_data.get("mode", "train")   # "train" or "infer"

        # Validate parameters
        try:
            input_data = validate_training_params(input_data)
        except ValueError as e:
            logger.error(f"Parameter validation failed: {e}")
            return {"error": str(e), "status": "failed"}

        logger.info(f"Processing mode: {mode}")
        monitor_resources()

        # Login to HuggingFace if token provided
        hf_token = input_data.get("HUGGINGFACE_TOKEN")
        if hf_token:
            login(hf_token)
            logger.info("Logged in to HuggingFace successfully")

        # FLUX.1-dev model paths (downloaded in Dockerfile)
        flux_model_path = "/workspace/models/flux1-dev.safetensors"
        clip_l_path = "/workspace/models/clip_l.safetensors"
        t5xxl_path = "/workspace/models/t5xxl_fp16.safetensors"
        ae_path = "/workspace/models/ae.safetensors"

        if mode == "train":
            # Training parameters
            train_data = input_data.get("train_data", "/workspace/data")
            output_dir = input_data.get("output_dir", "/workspace/output")
            os.makedirs(output_dir, exist_ok=True)

            # Character training parameters (FluxGym style)
            character_name = input_data.get("character_name", "")
            character_trigger = input_data.get("character_trigger", "")

            # Captioning parameters
            use_captioning = input_data.get("use_captioning", True)
            caption_extension = input_data.get("caption_extension", ".txt")
            caption_method = input_data.get("caption_method", "blip")  # "blip", "clip", or "existing"
            max_caption_length = input_data.get("max_caption_length", 75)

            # Cloudflare R2 parameters - using exact environment variable names
            use_r2 = input_data.get("use_r2", False)
            r2_access_key = input_data.get("CLOUDFLARE_R2_ACCESS_KEY_ID")
            r2_secret_key = input_data.get("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
            r2_account_id = input_data.get("CLOUDFLARE_ACCOUNT_ID")
            r2_bucket = input_data.get("R2_BUCKET_NAME")
            r2_prefix = input_data.get("r2_prefix", "kohya/Dataset/riya_bhatu_v1/Character/")
            
            # Construct R2 endpoint from account ID
            if r2_account_id:
                r2_endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"

            # Training hyperparameters
            learning_rate = input_data.get("learning_rate", "1e-4")
            max_train_steps = input_data.get("max_train_steps", "1000")
            train_batch_size = input_data.get("train_batch_size", "1")
            network_dim = input_data.get("network_dim", "16")
            save_every_n_steps = input_data.get("save_every_n_steps", "500")

            # Download dataset from Cloudflare R2 if specified
            if use_r2 and r2_access_key and r2_secret_key and r2_endpoint and r2_bucket:
                logger.info("Downloading dataset from Cloudflare R2...")
                s3_client, bucket_name = setup_cloudflare_r2(
                    r2_access_key, r2_secret_key, r2_endpoint, r2_bucket
                )
                download_dataset_from_r2(s3_client, bucket_name, r2_prefix, train_data)

            # Generate captions if requested
            if use_captioning and caption_method != "existing":
                logger.info(f"Generating captions using {caption_method} method...")
                if caption_method == "blip":
                    generate_captions_blip(train_data, caption_extension, max_caption_length, character_trigger)
                elif caption_method == "clip":
                    generate_captions_clip(train_data, caption_extension, character_trigger)

            # Validate character parameters (FluxGym style)
            if not character_trigger:
                logger.warning("No character trigger word provided. Consider adding one for better training results.")
            if character_trigger:
                logger.info(f"Using character trigger: '{character_trigger}'")

            # Set output name based on character (FluxGym style)
            if character_name:
                output_name = f"{character_name.replace(' ', '_')}_lora"
            else:
                output_name = "character_lora"

            # Use FLUX.1 training script with proper parameters
            cmd = [
                "python3", "/workspace/kohya/flux_train_network.py",
                "--pretrained_model_name_or_path", flux_model_path,
                "--clip_l", clip_l_path,
                "--t5xxl", t5xxl_path,
                "--ae", ae_path,
                "--train_data_dir", train_data,
                "--output_dir", output_dir,
                "--output_name", output_name,
                "--network_module", "networks.lora_flux",
                f"--network_dim", network_dim,
                "--network_alpha", "1",
                f"--learning_rate", learning_rate,
                "--lr_scheduler", "constant",
                f"--train_batch_size", train_batch_size,
                f"--max_train_steps", max_train_steps,
                f"--save_every_n_steps", save_every_n_steps,
                "--mixed_precision", "fp16",
                "--optimizer_type", "AdamW8bit",
                "--max_data_loader_n_workers", "0",
                "--seed", "42",
                "--guidance_scale", "1.0",  # Disable guidance for FLUX.1-dev training
                "--timestep_sampling", "flux_shift",  # Recommended for FLUX.1
                "--model_prediction_type", "raw",  # Recommended for FLUX.1
                "--discrete_flow_shift", "3.1582",  # For flux_shift sampling
                "--gradient_checkpointing",
                "--cache_text_encoder_outputs",
                "--cache_latents",
                "--save_model_as", "safetensors"
            ]

            # Add captioning parameters if captions exist
            if use_captioning:
                cmd.extend([
                    f"--caption_extension", caption_extension,
                    "--shuffle_caption",
                    "--keep_tokens", "1"
                ])

            logger.info(f"Running training command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Training failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return {"error": "Training failed", "stdout": result.stdout, "stderr": result.stderr, "status": "failed"}
            
            logger.info("Training completed successfully")
            
            # Upload trained model to R2 if credentials provided
            if use_r2 and r2_access_key and r2_secret_key and r2_endpoint and r2_bucket:
                upload_training_results_to_r2(s3_client, bucket_name, output_dir, r2_prefix)
            
            return {"status": "training complete", "output": output_dir}

        elif mode == "infer":
            prompt = input_data.get("prompt", "a character portrait")
            output_path = "/workspace/output.png"

            # Character parameters for inference (FluxGym style)
            character_trigger = input_data.get("character_trigger", "")

            # Integrate character trigger into prompt (FluxGym style)
            if character_trigger:
                enhanced_prompt = f"{character_trigger}, {prompt}"
                logger.info(f"Enhanced prompt: {enhanced_prompt}")
            else:
                enhanced_prompt = prompt

            # Cloudflare R2 parameters for inference
            use_r2 = input_data.get("use_r2", False)
            r2_access_key = input_data.get("CLOUDFLARE_R2_ACCESS_KEY_ID")
            r2_secret_key = input_data.get("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
            r2_account_id = input_data.get("CLOUDFLARE_ACCOUNT_ID")
            r2_bucket = input_data.get("R2_BUCKET_NAME")
            r2_prefix = input_data.get("r2_prefix", "kohya/Dataset/riya_bhatu_v1/Character/")
            
            # Setup R2 client for inference
            s3_client = None
            bucket_name = None
            if use_r2 and r2_access_key and r2_secret_key and r2_account_id and r2_bucket:
                r2_endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"
                s3_client, bucket_name = setup_cloudflare_r2(
                    r2_access_key, r2_secret_key, r2_endpoint, r2_bucket
                )

            # Use Kohya's FLUX inference script
            cmd = [
                "python3", "/workspace/kohya/flux_minimal_inference.py",
                "--ckpt_path", flux_model_path,
                "--clip_l", clip_l_path,
                "--t5xxl", t5xxl_path,
                "--ae", ae_path,
                "--prompt", enhanced_prompt,
                "--width", "1024",
                "--height", "1024",
                "--guidance_scale", "3.5",  # Default guidance for FLUX.1 inference
                "--num_inference_steps", "20",
                "--output_path", output_path
            ]
            logger.info(f"Running inference command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Inference failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return {"error": "Inference failed", "stdout": result.stdout, "stderr": result.stderr, "status": "failed"}
            
            logger.info("Inference completed successfully")
            
            # Upload generated image to R2 if credentials provided
            if s3_client and bucket_name:
                inference_prefix = f"{r2_prefix.rstrip('/')}/generated_images/"
                image_r2_key = inference_prefix + f"generated_{int(time.time())}.png"
                try:
                    s3_client.upload_file(output_path, bucket_name, image_r2_key)
                    logger.info(f"Uploaded generated image: {image_r2_key}")
                    return {"status": "inference complete", "image": output_path, "r2_url": f"https://{r2_account_id}.r2.cloudflarestorage.com/{image_r2_key}"}
                except Exception as e:
                    logger.error(f"Failed to upload image: {e}")
                    return {"status": "inference complete", "image": output_path}

            return {"status": "inference complete", "image": output_path}

        return {"error": "invalid mode"}

    except Exception as e:
        logger.error(f"Handler execution failed: {e}", exc_info=True)
        return {"error": str(e), "status": "failed"}

runpod.serverless.start({"handler": handler})
