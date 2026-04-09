import sys
from safetensors import safe_open

file_path = "/eva_data1/lynn/Video_Stylization/2_DiT/DiffSynth-Studio/models/train/Wan2.1-VACE-1.3B_lora_depth/step-200.safetensors"

print(f"Inspecting file: {file_path}")

try:
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        
        print(f"Total keys found: {len(keys)}")
        
        lora_keys = [k for k in keys if "lora" in k.lower()]
        context_embedder_keys = [k for k in keys if "context_embedder" in k.lower() or "patch_embedding" in k.lower()]
        
        print("\n--- LoRA Weights ---")
        if lora_keys:
            print(f"Found {len(lora_keys)} LoRA keys.")
            print("First 5 LoRA keys:")
            for k in lora_keys[:5]:
                print(f"  - {k} (Shape: {f.get_tensor(k).shape})")
        else:
            print("No keys containing 'lora' found.")
            
        print("\n--- Context Embedder / Patch Embedding Weights ---")
        if context_embedder_keys:
             print(f"Found {len(context_embedder_keys)} Context Embedder keys.")
             for k in context_embedder_keys:
                print(f"  - {k} (Shape: {f.get_tensor(k).shape})")
        else:
             print("No keys containing 'context_embedder' or 'patch_embedding' found.")

except Exception as e:
    print(f"Error opening file: {e}")
