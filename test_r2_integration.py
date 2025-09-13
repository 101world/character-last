#!/usr/bin/env python3
"""
Test script for FLUX Kohya Worker R2 Integration
Tests the Cloudflare R2 upload/download functionality
"""

import os
import boto3
from botocore.client import Config
import json

def setup_cloudflare_r2(access_key, secret_key, endpoint, bucket_name):
    """Setup Cloudflare R2 client"""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=Config(signature_version='s3v4')
    )
    return s3_client, bucket_name

def test_r2_connection():
    """Test R2 connection and basic operations"""
    # Get credentials from environment variables
    r2_access_key = os.getenv('CLOUDFLARE_R2_ACCESS_KEY_ID')
    r2_secret_key = os.getenv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
    r2_account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
    r2_bucket = os.getenv('R2_BUCKET_NAME')

    if not all([r2_access_key, r2_secret_key, r2_account_id, r2_bucket]):
        print("❌ Missing R2 environment variables. Please set:")
        print("  CLOUDFLARE_R2_ACCESS_KEY_ID")
        print("  CLOUDFLARE_R2_SECRET_ACCESS_KEY")
        print("  CLOUDFLARE_ACCOUNT_ID")
        print("  R2_BUCKET_NAME")
        return False

    r2_endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"

    print("Testing Cloudflare R2 connection...")
    print(f"Endpoint: {r2_endpoint}")
    print(f"Bucket: {r2_bucket}")

    try:
        s3_client, bucket_name = setup_cloudflare_r2(
            r2_access_key, r2_secret_key, r2_endpoint, r2_bucket
        )

        # Test bucket access
        response = s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Successfully connected to bucket: {bucket_name}")

        # Test listing objects
        prefix = "kohya/Dataset/riya_bhatu_v1/Character/"
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            print(f"✓ Found {len(response['Contents'])} objects in {prefix}")
            for obj in response['Contents'][:5]:  # Show first 5
                print(f"  - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print(f"✓ Bucket accessible, no objects found in {prefix}")

        return True

    except Exception as e:
        print(f"✗ R2 connection failed: {e}")
        return False

def test_upload_download():
    """Test upload and download functionality"""
    print("\nTesting upload/download functionality...")

    # Create a test file
    test_file = "/tmp/test_upload.txt"
    test_content = "This is a test file for R2 upload functionality"

    with open(test_file, 'w') as f:
        f.write(test_content)

    # Setup R2 client
    r2_access_key = "ef926435442c79cb22a8397939f3f878"
    r2_secret_key = "da8c672469940a0b338d86c65b386fc7fe933549706e3aff10ce6d570ec82eb3"
    r2_account_id = "ced616f33f6492fd708a8e897b61b953"
    r2_bucket = "the-social-twin-storage"
    r2_endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"

    try:
        s3_client, bucket_name = setup_cloudflare_r2(
            r2_access_key, r2_secret_key, r2_endpoint, r2_bucket
        )

        # Upload test file
        test_key = "test_upload.txt"
        s3_client.upload_file(test_file, bucket_name, test_key)
        print(f"✓ Successfully uploaded test file: {test_key}")

        # Download test file
        download_file = "/tmp/test_download.txt"
        s3_client.download_file(bucket_name, test_key, download_file)

        # Verify content
        with open(download_file, 'r') as f:
            downloaded_content = f.read()

        if downloaded_content == test_content:
            print("✓ Successfully downloaded and verified test file")
        else:
            print("✗ Downloaded content doesn't match original")

        # Clean up
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print("✓ Cleaned up test file")

        # Clean up local files
        os.remove(test_file)
        os.remove(download_file)

        return True

    except Exception as e:
        print(f"✗ Upload/download test failed: {e}")
        return False

if __name__ == "__main__":
    print("FLUX Kohya Worker R2 Integration Test")
    print("=" * 50)

    # Test connection
    connection_ok = test_r2_connection()

    if connection_ok:
        # Test upload/download
        upload_ok = test_upload_download()

        if upload_ok:
            print("\n🎉 All R2 integration tests passed!")
        else:
            print("\n❌ Upload/download tests failed")
    else:
        print("\n❌ Connection tests failed")