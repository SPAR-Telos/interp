"""Upload decoder probe model to HuggingFace repository.

This script uploads the trained decoder probe model to the project-telos/interp
repository under a new folder called 'decoder_reasoning_probe'.

Usage:
    python scripts/upload_decoder_to_hf.py
"""

from pathlib import Path
from huggingface_hub import HfApi

def upload_decoder_probe():
    """Upload decoder probe model to HuggingFace."""
    
    # Configuration
    repo_id = "project-telos/interp"
    model_path = Path("models/decoder_probe_layer15_full.pt")
    target_path = "decoder_reasoning_probe/decoder_probe_layer15_full.pt"
    
    print(f"\n{'='*80}")
    print("Uploading Decoder Probe to HuggingFace")
    print(f"{'='*80}")
    print(f"  Repository: {repo_id}")
    print(f"  Local file: {model_path}")
    print(f"  Target path: {target_path}")
    print(f"  File size: {model_path.stat().st_size / (1024**3):.2f} GB")
    print(f"{'='*80}\n")
    
    # Check if file exists
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        return
    
    # Initialize HuggingFace API
    api = HfApi()
    
    try:
        print("📤 Uploading model to HuggingFace...")
        print("   (This may take several minutes for large files)\n")
        
        # Upload file
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add decoder reasoning probe (layer 15, full training)"
        )
        
        print(f"\n✅ Successfully uploaded decoder probe!")
        print(f"\n📍 View at: https://huggingface.co/{repo_id}/tree/main/decoder_reasoning_probe")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error uploading file: {e}")
        print("\nMake sure you are logged in to HuggingFace:")
        print("  huggingface-cli login")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    upload_decoder_probe()
