#!/usr/bin/env python3
"""
Test script for FLUX.1-dev training worker serverless endpoint
"""

import json
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the handler function
from handler import handler

def test_character_training():
    """Test character LoRA training workflow"""
    test_event = {
        "input": {
            "mode": "train",
            "character_name": "Test Character",
            "character_trigger": "testchar",
            "use_r2": False,  # Skip R2 for testing
            "use_captioning": True,
            "caption_method": "blip",
            "max_caption_length": 50,
            "learning_rate": "1e-4",
            "max_train_steps": "10",  # Very short for testing
            "network_dim": "8",  # Smaller for testing
            "save_every_n_steps": "5"
        }
    }

    print("🧪 Testing FLUX.1-dev Character Training Handler...")
    print(f"📝 Test Input: {json.dumps(test_event, indent=2)}")

    try:
        result = handler(test_event)
        print(f"✅ Handler Response: {json.dumps(result, indent=2)}")

        if result.get("status") == "training complete":
            print("🎉 Character training test PASSED!")
            return True
        else:
            print(f"❌ Character training test FAILED: {result}")
            return False

    except Exception as e:
        print(f"💥 Handler test FAILED with exception: {e}")
        return False

def test_inference():
    """Test inference workflow"""
    test_event = {
        "input": {
            "mode": "infer",
            "prompt": "a portrait photo",
            "character_trigger": "testchar",
            "use_r2": False  # Skip R2 for testing
        }
    }

    print("\n🧪 Testing FLUX.1-dev Inference Handler...")
    print(f"📝 Test Input: {json.dumps(test_event, indent=2)}")

    try:
        result = handler(test_event)
        print(f"✅ Handler Response: {json.dumps(result, indent=2)}")

        if result.get("status") == "inference complete":
            print("🎉 Inference test PASSED!")
            return True
        else:
            print(f"❌ Inference test FAILED: {result}")
            return False

    except Exception as e:
        print(f"💥 Inference test FAILED with exception: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FLUX.1-dev Training Worker Test Suite")
    print("=" * 50)

    # Run tests
    training_passed = test_character_training()
    inference_passed = test_inference()

    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Character Training: {'✅ PASSED' if training_passed else '❌ FAILED'}")
    print(f"Inference: {'✅ PASSED' if inference_passed else '❌ FAILED'}")

    if training_passed and inference_passed:
        print("\n🎯 ALL TESTS PASSED! Serverless endpoint is ready! 🎯")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")
        sys.exit(1)