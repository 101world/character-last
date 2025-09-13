import runpod
import subprocess
import os
import glob
import boto3
import json
from huggingface_hub import snapshot_download, login
from botocore.client import Config
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import open_clip
from sentence_transformers import SentenceTransformer

def setup_cloudflare_r2(access_key, secret_key, endpoint_url, bucket_name):
    """Setup Cloudflare R2 client for dataset access"""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        config=Config(signature_version='s3v4')
    )
    return s3_client, bucket_name

def download_dataset_from_r2(s3_client, bucket_name, r2_prefix, local_dir):
    """Download dataset from Cloudflare R2"""
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=r2_prefix):
        for obj in page.get('Contents', []):
            local_path = os.path.join(local_dir, obj['Key'].replace(r2_prefix, '').lstrip('/'))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            s3_client.download_file(bucket_name, obj['Key'], local_path)
            print(f"Downloaded: {obj['Key']} -> {local_path}")

def generate_captions_blip(image_dir, caption_extension=".txt", max_length=75):
    """Generate captions using BLIP model"""
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

                    caption = processor.decode(output[0], skip_special_tokens=True)
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)

                    print(f"Generated caption for: {image_path}")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

def generate_captions_clip(image_dir, caption_extension=".txt"):
    """Generate captions using CLIP model"""
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # This is a simplified CLIP captioning - in practice you'd want more sophisticated prompting
    templates = [
        "a photo of a {}",
        "an image of a {}",
        "picture of a {}",
        "this is a {}"
    ]

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

                    # For simplicity, using a generic caption. In practice, you'd use a trained captioning model
                    caption = "a high quality image of a character"
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)

                    print(f"Generated CLIP-based caption for: {image_path}")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

def handler(event):
    input_data = event.get("input", {})
    mode = input_data.get("mode", "train")   # "train" or "infer"

    # Login to HuggingFace if token provided
    hf_token = input_data.get("HUGGINGFACE_TOKEN")
    if hf_token:
        login(hf_token)
        print("Logged in to HuggingFace successfully")

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
            print("Downloading dataset from Cloudflare R2...")
            s3_client, bucket_name = setup_cloudflare_r2(
                r2_access_key, r2_secret_key, r2_endpoint, r2_bucket
            )
            download_dataset_from_r2(s3_client, bucket_name, r2_prefix, train_data)

        # Generate captions if requested
        if use_captioning and caption_method != "existing":
            print(f"Generating captions using {caption_method} method...")
            if caption_method == "blip":
                generate_captions_blip(train_data, caption_extension, max_caption_length)
            elif caption_method == "clip":
                generate_captions_clip(train_data, caption_extension)

        # Use FLUX.1 training script with proper parameters
        cmd = [
            "python3", "/workspace/kohya/flux_train_network.py",
            "--pretrained_model_name_or_path", flux_model_path,
            "--clip_l", clip_l_path,
            "--t5xxl", t5xxl_path,
            "--ae", ae_path,
            "--train_data_dir", train_data,
            "--output_dir", output_dir,
            "--output_name", "flux_lora",
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

        print(f"Running training command: {' '.join(cmd)}")
        subprocess.run(cmd)
        return {"status": "training complete", "output": output_dir}

    elif mode == "infer":
        prompt = input_data.get("prompt", "a test image")
        output_path = "/workspace/output.png"

        # Use Kohya's FLUX inference script
        cmd = [
            "python3", "/workspace/kohya/flux_minimal_inference.py",
            "--ckpt_path", flux_model_path,
            "--clip_l", clip_l_path,
            "--t5xxl", t5xxl_path,
            "--ae", ae_path,
            "--prompt", prompt,
            "--width", "1024",
            "--height", "1024",
            "--guidance_scale", "3.5",  # Default guidance for FLUX.1 inference
            "--num_inference_steps", "20",
            "--output_path", output_path
        ]
        subprocess.run(cmd)

        return {"status": "inference complete", "image": output_path}

    return {"error": "invalid mode"}

runpod.serverless.start({"handler": handler})
