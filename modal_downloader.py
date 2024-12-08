import modal

app_name = "downloader"
app = modal.App(app_name)

volume = modal.Volume.from_name(
    "openwebtext",
    create_if_missing=True,
)

base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pydantic==2.9.1",

)

@app.function(
    image=base_image,
    mounts=[
        modal.Mount.from_local_dir(
            ".",
            remote_path="/source/.",
        ),
    ],
    volumes={"/openwebtext/": volume},
    timeout=60000,
)
def runner():


@app.local_entrypoint()
def main():
    runner.remote()