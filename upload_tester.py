import modal
import os

app_name = "downloader"
app = modal.App(app_name)

volume = modal.Volume.from_name(
    "openwebtext",
    create_if_missing=True,
)

base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pydantic==2.9.1",
    "transformers",
)

@app.function(
    image=base_image,
    volumes={"/openwebtext/": volume},
    timeout=3600,
)
def upload_to_huggingface(repo_id: str, private: bool = True) -> str:
    """Upload model weights from Modal volume to HuggingFace."""
    from huggingface_hub import HfApi, create_repo

    # Get HF token
    # hf_token = os.environ["HF_TOKEN"]
    hf_token = "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"
    if not hf_token:
        raise ValueError("HuggingFace token not found in environment")

    # Initialize HF API
    api = HfApi(token=hf_token)

    # Create or verify repo
    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            token=hf_token,
            exist_ok=True
        )
        print(f"✓ Repository {repo_id} ready")
    except Exception as e:
        print(f"Note on repo creation: {e}")

    # Upload model files
    print(f":arrow_up:  Uploading files to {repo_id}")

    with open("/openwebtext/test.txt", "w+") as fopen:
        fopen.write("basedddddddddd")

    file_name = "ckpt-5625.pt"
    api.upload_file(
        path_or_fileobj=f"/openwebtext/test.txt",
        path_in_repo=f"test.txt",
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
        # commit_message=f"Uploading {file_name}"
    )

@app.local_entrypoint()
def main():
    upload_to_huggingface.remote("abecedarianc/downloader")