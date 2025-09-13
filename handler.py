import runpod
import subprocess
import os
import glob
from huggingface_hub import snapshot_download

def handler(event):
    input_data = event.get("input", {})
    mode = input_data.get("mode", "train")   # "train" or "infer"

    # FLUX.1-dev model paths (downloaded in Dockerfile)
    flux_model_path = "/workspace/models/flux1-dev.safetensors"
    clip_l_path = "/workspace/models/clip_l.safetensors"
    t5xxl_path = "/workspace/models/t5xxl_fp16.safetensors"
    ae_path = "/workspace/models/ae.safetensors"

    if mode == "train":
        train_data = input_data.get("train_data", "/workspace/data")
        output_dir = input_data.get("output_dir", "/workspace/output")
        os.makedirs(output_dir, exist_ok=True)

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
            "--network_dim", "16",  # LoRA rank
            "--network_alpha", "1",
            "--learning_rate", "1e-4",
            "--lr_scheduler", "constant",
            "--train_batch_size", "1",
            "--max_train_steps", "1000",
            "--save_every_n_steps", "500",
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
