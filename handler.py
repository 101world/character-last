import runpod
import os
import subprocess
import boto3
from botocore.client import Config
import logging
from huggingface_hub import snapshot_download, login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_r2(access_key, secret_key, account_id, bucket_name):
    if not all([access_key, secret_key, account_id, bucket_name]):
        raise ValueError("Missing R2 credentials")
    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4")
    )
    client.head_bucket(Bucket=bucket_name)
    logger.info(f"Connected to R2 bucket: {bucket_name}")
    return client, endpoint

def upload_to_r2(client, bucket_name, local_dir, r2_prefix="lora_outputs/"):
    uploaded_files = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.endswith(".safetensors"):
                local_path = os.path.join(root, file)
                r2_key = os.path.join(r2_prefix, file)
                client.upload_file(local_path, bucket_name, r2_key)
                logger.info(f"Uploaded {file} -> {r2_key}")
                uploaded_files.append(r2_key)
    return uploaded_files

def download_flux_models(hf_token=None):
    """Download FLUX.1-dev models if they don't exist"""
    logger.info("Checking for FLUX models...")

    models_dir = "/workspace/models"
    os.makedirs(models_dir, exist_ok=True)

    # FLUX.1-dev model path
    flux_model_path = os.path.join(models_dir, "flux.safetensors")

    if os.path.exists(flux_model_path):
        logger.info("FLUX model already exists, skipping download")
        return flux_model_path

    # Need to download model
    logger.info("FLUX model not found, downloading from HuggingFace...")

    if not hf_token:
        raise ValueError("HF_TOKEN required for downloading FLUX.1-dev model")

    try:
        # Login to HuggingFace
        login(hf_token)
        logger.info("Logged in to HuggingFace successfully")

        # Download FLUX.1-dev model
        logger.info("Downloading FLUX.1-dev model...")
        downloaded_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            token=hf_token
        )

        # Find the safetensors file
        for root, dirs, files in os.walk(downloaded_path):
            for file in files:
                if file.endswith(".safetensors") and "flux" in file.lower():
                    flux_model_path = os.path.join(root, file)
                    logger.info(f"Found FLUX model: {flux_model_path}")
                    return flux_model_path

        raise FileNotFoundError("Could not find FLUX safetensors file after download")

    except Exception as e:
        logger.error(f"Failed to download FLUX model: {e}")
        raise

def handler(job):
    try:
        data = job.get("input", {})
        model_path = data.get("model_path", "/workspace/models/flux.safetensors")
        train_dir = data.get("train_data", "/workspace/data")
        output_dir = data.get("output_dir", "/workspace/output")
        steps = str(data.get("steps", 1000))
        hf_token = data.get("HF_TOKEN")
        os.makedirs(output_dir, exist_ok=True)

        # Download FLUX model if it doesn't exist
        if not os.path.exists(model_path):
            logger.info("Model not found locally, downloading from HuggingFace...")
            model_path = download_flux_models(hf_token)
        else:
            logger.info(f"Using existing model: {model_path}")

        # R2 config
        use_r2 = data.get("use_r2", False)
        r2_access = data.get("CLOUDFLARE_R2_ACCESS_KEY_ID")
        r2_secret = data.get("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
        r2_account = data.get("CLOUDFLARE_ACCOUNT_ID")
        r2_bucket = data.get("R2_BUCKET_NAME")
        r2_prefix = data.get("r2_prefix", "lora_outputs/")

        # Download FLUX model if not exists
        hf_token = data.get("HF_TOKEN")
        if hf_token:
            download_flux_models(hf_token)

        # Training command
        cmd = [
            "python3", "/workspace/kohya/flux_train_network.py",
            f"--pretrained_model_name_or_path={model_path}",
            f"--train_data_dir={train_dir}",
            f"--output_dir={output_dir}",
            "--network_module=networks.lora_flux",
            f"--max_train_steps={steps}",
            "--mixed_precision", "fp16",
            "--optimizer_type", "AdamW8bit",
            "--gradient_checkpointing"
        ]
        logger.info(f"Running training: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        loras = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
        logger.info(f"Training complete. Found {len(loras)} LoRAs.")

        r2_urls = []
        if use_r2 and r2_access and r2_secret and r2_account and r2_bucket:
            client, endpoint = setup_r2(r2_access, r2_secret, r2_account, r2_bucket)
            uploaded_files = upload_to_r2(client, r2_bucket, output_dir, r2_prefix)
            r2_urls = [f"{endpoint}/{file}" for file in uploaded_files]

        return {"status": "training_complete", "loras": loras, "r2_urls": r2_urls}

    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"status": "failed", "error": str(e)}

runpod.serverless.start({"handler": handler})
