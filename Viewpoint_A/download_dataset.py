# download the dataset into dir
import json
import os
import subprocess
import sys
from urllib.parse import urlparse

# Read API key from environment for safety.
api_key = os.getenv("MDC_API_KEY")
if not api_key:
    print("Missing API key. Set environment variable: MDC_API_KEY")
    sys.exit(1)

# Dataset download endpoint from Mozilla Data Collective
dataset_download_endpoint = (
    "https://mozilladatacollective.com/api/datasets/cmwdapwry02jnmh07dyo46mot/download"
)

# Destination directory + filename
save_dir = "./"
output_filename = "Common Voice Scripted Speech 25.0 - English.tar.gz"

# Step 1: Get presigned download URL
response = subprocess.run([
    "curl", "-fsSL", "-X", "POST", dataset_download_endpoint,
    "-H", f"Authorization: Bearer {api_key}",
    "-H", "Accept: application/json",
    "-H", "Content-Type: application/json",
], capture_output=True, text=True)

if response.returncode != 0:
    print("Failed to request presigned URL.")
    if response.stderr.strip():
        print(response.stderr.strip())
    sys.exit(1)

raw = response.stdout.strip()
if not raw:
    print("Dataset API returned an empty response; cannot parse JSON.")
    sys.exit(1)

try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("Dataset API did not return valid JSON.")
    print(f"Response starts with: {raw[:300]}")
    sys.exit(1)

download_url = data.get("downloadUrl") or data.get("download_url") or data.get("url")
if not download_url:
    print("No download URL found in API response.")
    print(f"Response keys: {list(data.keys())}")
    sys.exit(1)

print("Got download URL, starting download...")

# Step 2: Download to directory
os.makedirs(save_dir, exist_ok=True)
filename = output_filename or os.path.basename(urlparse(download_url).path) or "dataset.bin"
output_path = os.path.join(save_dir, filename)

if os.path.exists(output_path):
    existing_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Existing partial file found: {existing_size_gb:.2f} GB. Resuming...")

download_result = subprocess.run([
    "curl",
    "-fL",
    "-C", "-",                  # resume partial download
    "--retry", "8",             # retry transient failures
    "--retry-all-errors",
    "--retry-delay", "5",
    "--speed-time", "30",       # fail/retry on stalled transfer
    "--speed-limit", "1024",
    "-o", output_path,
    download_url
])

if download_result.returncode != 0:
    print("Download failed.")
    sys.exit(1)

print("Download complete!")
print(f"Saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")
