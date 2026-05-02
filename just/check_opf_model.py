import os
import sys
from pathlib import Path


def main():
    checkpoint = os.environ.get("OPF_CHECKPOINT")
    if checkpoint:
        print(f"  ✓ Using OPF_CHECKPOINT: {checkpoint}")
        if not Path(checkpoint).exists():
            print(f"ERROR: OPF_CHECKPOINT path does not exist: {checkpoint}")
            sys.exit(1)
        if not (Path(checkpoint) / "config.json").is_file():
            print(f"ERROR: OPF_CHECKPOINT path is incomplete: missing config.json")
            sys.exit(1)
        return

    default_path = Path.home() / ".opf" / "privacy_filter"
    if default_path.exists() and (default_path / "config.json").is_file():
        print(f"  ✓ Model found at default path: {default_path}")
        return

    print("  Model not found. Downloading from HuggingFace (openai/privacy-filter)...")
    print("  This may take a while on first run...")
    try:
        from opf._common.checkpoint_download import ensure_default_checkpoint

        path = ensure_default_checkpoint()
        print(f"  ✓ Model downloaded to: {path}")
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        print("  You can manually set OPF_CHECKPOINT to a local model path.")
        sys.exit(1)


if __name__ == "__main__":
    main()
