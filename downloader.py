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
    # url = api.upload_folder(
    #     folder_path=str(model_path),
    #     repo_id=repo_id,
    #     token=hf_token,
    #     ignore_patterns=[".*"],
    #     commit_message=f"Upload model from run {run_name}"
    # )
    model_name = "baseline-large"
    file_name = "ckpt-5625.pt"
    api.upload_file(
        path_or_fileobj=f"/openwebtext/{file_name}",
        path_in_repo=f"{model_name}-{file_name}",
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
        # commit_message=f"Uploading {file_name}"
    )

@app.local_entrypoint()
def main():
    upload_to_huggingface.remote("abecedarianc/downloader")