"""Download checkpoints from Modal storage to local model/checkpoint directory."""
# modal run src/download_checkpoint.py::download # download latest
# modal run src/download_checkpoint.py::list # list checkpoints
# modal run src/download_checkpoint.py::download --checkpoint-name checkpoint_step_1000.pt
from pathlib import Path
import modal
import os
# Reference the same app and volume from train.py
app = modal.App("chess-training")
checkpoints_vol = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)

# Simple image for downloading
image = modal.Image.debian_slim()


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoints_vol},
)
def list_checkpoints() -> list[str]:
    """List available checkpoints in Modal storage."""
    checkpoint_dir = "/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path)
            checkpoints.append({
                'name': file,
                'size_mb': file_size / 1024 / 1024,
                'path': file_path
            })

    return sorted(checkpoints, key=lambda x: x['name'])


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoints_vol},
)
def get_checkpoint_data(checkpoint_name: str) -> bytes:
    """Get checkpoint file data from Modal storage."""
    modal_path = f"/checkpoints/{checkpoint_name}"

    if not os.path.exists(modal_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found in Modal storage")

    # Read the checkpoint file
    with open(modal_path, 'rb') as f:
        checkpoint_data = f.read()

    return checkpoint_data


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoints_vol},
)
def delete_checkpoint(checkpoint_name: str) -> dict:
    """Delete a specific checkpoint from Modal storage."""
    modal_path = f"/checkpoints/{checkpoint_name}"

    if not os.path.exists(modal_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found in Modal storage")

    # Get size before deleting
    file_size_mb = os.path.getsize(modal_path) / 1024 / 1024

    # Delete the checkpoint file
    os.remove(modal_path)
    
    # Commit changes to volume
    checkpoints_vol.commit()

    return {
        'name': checkpoint_name,
        'size_mb': file_size_mb,
        'deleted': True
    }


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoints_vol},
)
def delete_all_checkpoints() -> dict:
    """Delete all checkpoints from Modal storage."""
    checkpoint_dir = "/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return {'deleted_count': 0, 'total_size_mb': 0}

    deleted_count = 0
    total_size = 0

    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            os.remove(file_path)
            deleted_count += 1

    # Commit changes to volume
    checkpoints_vol.commit()

    return {
        'deleted_count': deleted_count,
        'total_size_mb': total_size / 1024 / 1024
    }


@app.local_entrypoint()
def list():
    """List all available checkpoints.

    Usage:
        modal run src/download_checkpoint.py::list
    """
    print("📂 Listing checkpoints in Modal storage...\n")

    checkpoints = list_checkpoints.remote()

    if not checkpoints:
        print("❌ No checkpoints found in Modal storage.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s):\n")
    for ckpt in checkpoints:
        print(f"  📄 {ckpt['name']}")
        print(f"     Size: {ckpt['size_mb']:.2f} MB\n")


@app.local_entrypoint()
def download(checkpoint_name: str = None):
    """Download a checkpoint from Modal to local model/checkpoint directory.

    Usage:
        modal run src/download_checkpoint.py::download --checkpoint-name checkpoint_step_1000.pt

    If no checkpoint name is provided, downloads the latest checkpoint.
    """
    # Get the model/checkpoint directory (relative to this script)
    script_dir = Path(__file__).parent
    checkpoint_dir = script_dir / "model" / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # If no checkpoint specified, get the latest one
    if checkpoint_name is None:
        print("📋 No checkpoint specified, finding latest...")
        checkpoints = list_checkpoints.remote()

        if not checkpoints:
            print("❌ No checkpoints found in Modal storage.")
            return

        # Get the checkpoint with the highest step number
        latest = max(checkpoints, key=lambda x: x['name'])
        checkpoint_name = latest['name']
        print(f"✅ Latest checkpoint: {checkpoint_name}\n")

    print(f"📥 Downloading {checkpoint_name} from Modal...")

    # Download the checkpoint
    try:
        checkpoint_data = get_checkpoint_data.remote(checkpoint_name)
    except Exception as e:
        print(f"❌ Error downloading checkpoint: {e}")
        return

    # Save to local file in model/checkpoint
    output_path = checkpoint_dir / checkpoint_name
    with open(output_path, 'wb') as f:
        f.write(checkpoint_data)

    print(f"✅ Checkpoint saved to {output_path}")
    print(f"   Size: {len(checkpoint_data) / 1024 / 1024:.2f} MB")

    return str(output_path)


@app.local_entrypoint()
def download_all():
    """Download all checkpoints from Modal to local model/checkpoint directory.

    Usage:
        modal run src/download_checkpoint.py::download_all
    """
    # Get the model/checkpoint directory
    script_dir = Path(__file__).parent
    checkpoint_dir = script_dir / "model" / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("📋 Listing all checkpoints...\n")
    checkpoints = list_checkpoints.remote()

    if not checkpoints:
        print("❌ No checkpoints found in Modal storage.")
        return

    print(f"📥 Downloading {len(checkpoints)} checkpoint(s)...\n")

    total_size = 0
    for i, ckpt in enumerate(checkpoints, 1):
        checkpoint_name = ckpt['name']
        print(f"[{i}/{len(checkpoints)}] Downloading {checkpoint_name}...")

        try:
            checkpoint_data = get_checkpoint_data.remote(checkpoint_name)

            # Save to local file
            output_path = checkpoint_dir / checkpoint_name
            with open(output_path, 'wb') as f:
                f.write(checkpoint_data)

            size_mb = len(checkpoint_data) / 1024 / 1024
            total_size += size_mb
            print(f"            ✅ Saved ({size_mb:.2f} MB)\n")

        except Exception as e:
            print(f"            ❌ Error: {e}\n")
            continue

    print(f"🏁 Download complete!")
    print(f"   Total size: {total_size:.2f} MB")
    print(f"   Location: {checkpoint_dir}")


@app.local_entrypoint()
def delete(checkpoint_name: str):
    """Delete a specific checkpoint from Modal storage.

    Usage:
        modal run src/download_checkpoint.py::delete --checkpoint-name checkpoint_step_1000.pt
    """
    print(f"🗑️  Deleting {checkpoint_name} from Modal storage...")

    try:
        result = delete_checkpoint.remote(checkpoint_name)
        print(f"✅ Deleted {result['name']}")
        print(f"   Freed: {result['size_mb']:.2f} MB")
    except FileNotFoundError:
        print(f"❌ Checkpoint {checkpoint_name} not found in Modal storage.")
    except Exception as e:
        print(f"❌ Error deleting checkpoint: {e}")


@app.local_entrypoint()
def delete_all():
    """Delete ALL checkpoints from Modal storage.

    Usage:
        modal run src/download_checkpoint.py::delete_all

    WARNING: This will delete all checkpoints permanently!
    """
    print("⚠️  WARNING: This will delete ALL checkpoints from Modal storage!")
    print("📋 Listing checkpoints to be deleted...\n")

    checkpoints = list_checkpoints.remote()

    if not checkpoints:
        print("❌ No checkpoints found in Modal storage.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) to delete:\n")
    for ckpt in checkpoints:
        print(f"  📄 {ckpt['name']} ({ckpt['size_mb']:.2f} MB)")

    print("\n🗑️  Deleting all checkpoints...")

    try:
        result = delete_all_checkpoints.remote()
        print(f"\n✅ Deleted {result['deleted_count']} checkpoint(s)")
        print(f"   Freed: {result['total_size_mb']:.2f} MB")
    except Exception as e:
        print(f"\n❌ Error deleting checkpoints: {e}")


if __name__ == "__main__":
    # Default behavior when run directly
    print("Use one of the following commands:")
    print("\n📋 List:")
    print("  modal run src/download_checkpoint.py::list")
    print("\n📥 Download:")
    print("  modal run src/download_checkpoint.py::download --checkpoint-name <name>")
    print("  modal run src/download_checkpoint.py::download  (downloads latest)")
    print("  modal run src/download_checkpoint.py::download_all")
    print("\n🗑️  Delete:")
    print("  modal run src/download_checkpoint.py::delete --checkpoint-name <name>")
    print("  modal run src/download_checkpoint.py::delete_all  (⚠️  deletes ALL checkpoints!)")
