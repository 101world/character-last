import runpod
import os
import subprocess
import boto3
from botocore.client import Config
import logging

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

def handler(job):
    try:
        data = job.get("input", {})
        model_path = data.get("model_path", "/workspace/models/flux.safetensors")
        train_dir = data.get("train_data", "/workspace/data")
        output_dir = data.get("output_dir", "/workspace/output")
        steps = str(data.get("steps", 1000))
        os.makedirs(output_dir, exist_ok=True)

        # R2 config
        use_r2 = data.get("use_r2", False)
        r2_access = data.get("CLOUDFLARE_R2_ACCESS_KEY_ID")
        r2_secret = data.get("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
        r2_account = data.get("CLOUDFLARE_ACCOUNT_ID")
        r2_bucket = data.get("R2_BUCKET_NAME")
        r2_prefix = data.get("r2_prefix", "lora_outputs/")

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
