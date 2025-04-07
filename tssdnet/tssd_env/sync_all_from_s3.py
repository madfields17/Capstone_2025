import subprocess
import os

def sync_entire_bucket(bucket: str, local_dir: str = "./s3_data"):
    s3_path = f"s3://{bucket}"
    local_dir = local_dir.rstrip("/")
    os.makedirs(local_dir, exist_ok=True)

    print(f"🔄 Syncing ENTIRE bucket from {s3_path} → {local_dir}...")
    try:
        subprocess.run(["aws", "s3", "sync", s3_path, local_dir], check=True)
        print("✅ Sync complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Sync failed with error: {e}")
        exit(1)

if __name__ == "__main__":
    BUCKET = "audio-deepfake-detection-data"
    LOCAL_DIR = "./Standardized_full_data"  # All your code already expects this base path

    sync_entire_bucket(BUCKET, LOCAL_DIR)
