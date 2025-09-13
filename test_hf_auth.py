#!/usr/bin/env python3
"""
Test script to verify Hugging Face authentication and model access
This tests the runtime download functionality
"""
import os
import sys
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

def test_hf_authentication():
    """Test Hugging Face authentication"""
    print("Testing Hugging Face authentication...")

    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("❌ No HF_TOKEN or HUGGINGFACE_TOKEN environment variable found")
        print("Set it with: export HF_TOKEN=your_token_here")
        print("Or: export HUGGINGFACE_TOKEN=your_token_here")
        return False

    try:
        api = HfApi(token=token)
        user = api.whoami()
        print(f"✅ Authentication successful for user: {user['name']}")
        return True
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            print("❌ Invalid Hugging Face token")
            print("Get a new token from: https://huggingface.co/settings/tokens")
        else:
            print(f"❌ Authentication failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_access():
    """Test access to required models"""
    print("\nTesting model access...")

    models = [
        'black-forest-labs/FLUX.1-dev',
        'comfyanonymous/flux_text_encoders'
    ]

    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    api = HfApi(token=token) if token else HfApi()

    for model_id in models:
        try:
            # Try to get model info
            model_info = api.model_info(model_id)
            print(f"✅ Can access {model_id}")
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                print(f"❌ No access to {model_id} - may need to accept terms")
                print(f"Visit: https://huggingface.co/{model_id}")
            elif e.response.status_code == 403:
                print(f"❌ Forbidden access to {model_id}")
            else:
                print(f"❌ Error accessing {model_id}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error accessing {model_id}: {e}")
            return False

    return True

def main():
    print("FLUX.1-dev Runtime Model Download Test")
    print("=" * 50)

    auth_ok = test_hf_authentication()
    if not auth_ok:
        print("\n❌ Authentication test failed")
        print("This will cause runtime download failures")
        return 1

    model_ok = test_model_access()
    if not model_ok:
        print("\n❌ Model access test failed")
        print("Runtime downloads will fail")
        return 1

    print("\n✅ All tests passed!")
    print("Runtime model downloads should work correctly")
    print("\nFor RunPod deployment:")
    print("1. Set HF_TOKEN environment variable in your RunPod endpoint")
    print("2. First request will download models (~11GB, 10-15 minutes)")
    print("3. Subsequent requests will use cached models")
    return 0

if __name__ == '__main__':
    sys.exit(main())