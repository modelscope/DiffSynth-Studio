import os
from huggingface_hub import hf_hub_download, snapshot_download

# Defined download directory
DOWNLOAD_ROOT = "./models/"
os.makedirs(DOWNLOAD_ROOT, exist_ok=True)

print(f"Downloading models to: {DOWNLOAD_ROOT}")

# Wan-AI/Wan2.1-VACE-14B
print("\n[1/4] Downloading Wan2.1-VACE-14B (DiT/VACE)...")
# Note: This model usually has multiple shards. We download all related safetensors and index.
snapshot_download(
    repo_id="Wan-AI/Wan2.1-VACE-14B",
    allow_patterns=["diffusion_pytorch_model*.safetensors", "*.json","*.pth"], 
    local_dir=os.path.join(DOWNLOAD_ROOT, "Wan-AI/Wan2.1-VACE-14B"),
    local_dir_use_symlinks=False 
)

print(f"\nAll downloads completed! Check {DOWNLOAD_ROOT}")
