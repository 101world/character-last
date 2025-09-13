#!/usr/bin/env python3
"""
Model downloader script for FLUX.1-dev models with Hugging Face authentication
"""
import os
import sys
from huggingface_hub import hf_hub_download
import argparse

def download_model(repo_id, filename, local_dir):
    """Download a model file from Hugging Face"""
    try:
        print(f"Downloading {filename} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ {filename} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download FLUX models')
    parser.add_argument('--token', help='Hugging Face token')
    parser.add_argument('--local-dir', default='/workspace/models', help='Local directory to save models')
    args = parser.parse_args()

    # Set token if provided
    if args.token:
        os.environ['HF_TOKEN'] = args.token
        print("Using provided Hugging Face token")
    elif os.getenv('HF_TOKEN'):
        print("Using HF_TOKEN environment variable")
    else:
        print("Warning: No Hugging Face token provided. Downloads may fail for private models.")

    # Create local directory
    os.makedirs(args.local_dir, exist_ok=True)

    # Models to download
    models = [
        ('black-forest-labs/FLUX.1-dev', 'flux1-dev.safetensors'),
        ('black-forest-labs/FLUX.1-dev', 'ae.safetensors'),
        ('comfyanonymous/flux_text_encoders', 'clip_l.safetensors'),
        ('comfyanonymous/flux_text_encoders', 't5xxl_fp16.safetensors'),
    ]

    success_count = 0
    for repo_id, filename in models:
        if download_model(repo_id, filename, args.local_dir):
            success_count += 1

    if success_count == len(models):
        print(f"\n✓ All {len(models)} models downloaded successfully!")
        return 0
    else:
        print(f"\n✗ Only {success_count}/{len(models)} models downloaded successfully")
        return 1

if __name__ == '__main__':
    sys.exit(main())