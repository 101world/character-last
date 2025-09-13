import runpod
import subprocess
import os
import glob
import time
import boto3
import json
import logging
import sys
from huggingface_hub import hf_hub_download, login
from botocore.client import Config
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import open_clip

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ----------------- Model Downloader -----------------
def download_models_at_runtime():
    """Download required FLUX/Kohya models if missing"""
    models_dir = "/workspace/models"
    os.makedirs(models_dir, exist_ok=True)

    required_models = [
        ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
        ("black-forest-labs/FLUX.1-dev", "ae.safetensors"),
        ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
        ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
    ]

    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        logger.error("HF_TOKEN or HUGGINGFACE_TOKEN not set")
        return False

    for repo_id, filename in required_models:
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            logger.info(f"Downloading {filename}...")
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=models_dir,
                            local_dir_use_symlinks=False, token=token)
            logger.info(f"Downloaded {filename}")
    return True

# ----------------- Cloudflare R2 -----------------
def setup_cloudflare_r2(access_key, secret_key, endpoint_url, bucket_name):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        config=Config(signature_version='s3v4')
    )
    s3_client.head_bucket(Bucket=bucket_name)
    return s3_client, bucket_name

def upload_results_to_r2(s3_client, bucket_name, output_dir, r2_prefix):
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            local_path = os.path.join(root, file)
            r2_key = r2_prefix.rstrip('/') + '/' + os.path.relpath(local_path, output_dir)
            s3_client.upload_file(local_path, bucket_name, r2_key)
            logger.info(f"Uploaded: {r2_key}")

# ----------------- Health Check -----------------
def health_check():
    if not download_models_at_runtime():
        return False
    if not torch.cuda.is_available():
        logger.warning("CUDA not available")
    return True

# ----------------- Main Handler -----------------
def handler(event):
    try:
        logger.info("Starting FLUX training worker")
        if not health_check():
            return {"status": "failed", "error": "health check failed"}

        input_data = event.get("input", {})
        mode = input_data.get("mode", "train")

        # Paths
        flux_model = "/workspace/models/flux1-dev.safetensors"
        clip_l = "/workspace/models/clip_l.safetensors"
        t5xxl = "/workspace/models/t5xxl_fp16.safetensors"
        ae = "/workspace/models/ae.safetensors"

        # Training
        if mode == "train":
            train_data = input_data.get("train_data", "/workspace/data")
            output_dir = input_data.get("output_dir", "/workspace/output")
            os.makedirs(output_dir, exist_ok=True)
            character_name = input_data.get("character_name", "character")
            output_name = f"{character_name.replace(' ','_')}_lora"

            # Cloudflare R2
            use_r2 = input_data.get("use_r2", False)
            if use_r2:
                r2_access_key = input_data.get("CLOUDFLARE_R2_ACCESS_KEY_ID")
                r2_secret_key = input_data.get("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
                r2_account_id = input_data.get("CLOUDFLARE_ACCOUNT_ID")
                r2_bucket = input_data.get("R2_BUCKET_NAME")
                r2_prefix = input_data.get("r2_prefix","lora_results/")
                r2_endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"
                s3_client, bucket_name = setup_cloudflare_r2(
                    r2_access_key, r2_secret_key, r2_endpoint, r2_bucket
                )

            # FLUX training command
            cmd = [
                "python3", "/workspace/kohya/flux_train_network.py",
                "--pretrained_model_name_or_path", flux_model,
                "--clip_l", clip_l,
                "--t5xxl", t5xxl,
                "--ae", ae,
                "--train_data_dir", train_data,
                "--output_dir", output_dir,
                "--output_name", output_name,
                "--network_module", "networks.lora_flux",
                "--network_dim", "16",
                "--learning_rate", "1e-4",
                "--train_batch_size", "1",
                "--max_train_steps", "1000",
                "--save_every_n_steps", "500",
                "--mixed_precision", "fp16",
                "--optimizer_type", "AdamW8bit",
                "--gradient_checkpointing",
                "--cache_text_encoder_outputs",
                "--save_model_as", "safetensors"
            ]

            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(result.stderr)
                return {"status": "failed", "error": "training failed"}

            logger.info("Training complete")

            # Upload results to R2
            if use_r2:
                upload_results_to_r2(s3_client, bucket_name, output_dir, r2_prefix)

            return {"status": "success", "output_dir": output_dir}

        return {"status": "failed", "error": "invalid mode"}

    except Exception as e:
        logger.error(f"Handler failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}

# Start serverless
runpod.serverless.start({"handler": handler})
