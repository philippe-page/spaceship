"""The models a page can choose to steer, and getting their weights onto this machine. Nothing is downloaded until a page
asks for a model by name; weights go to the Hugging Face cache, where later runs find them."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tqdm.auto import tqdm

# Each model is pinned to a revision, so what is downloaded is what was tested. `size` is the download in bytes, for its progress.
MODELS = [
    {
        "id": "lfm3-1.7b",
        "name": "LFM3 1.7B",
        "repo": "mlx-community/LFM3-1.7B-OptiQ-4bit",
        "revision": "82d0e45972a9343db2ab5534b257b2d6fb3edd46",
        "size": 1_360_000_000,
    },
    {
        "id": "qwen3.5-2b",
        "name": "Qwen3.5 2B",
        "repo": "mlx-community/Qwen3.5-2B-4bit",
        "revision": "674aaa7240b91e8012fcad5d791b7dfe5ba90207",
        "size": 1_750_000_000,
    },
]

# The files mlx-lm loads a model from, less any code a repository carries: both models' code is already here or in mlx-lm.
WEIGHT_FILES = ["*.json", "model*.safetensors", "tokenizer.model", "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja"]


def model_of(model_id: str):
    return next((m for m in MODELS if m["id"] == model_id), None)


def local_path(model: dict):
    """The model's folder if all its weights are on this machine, else None. A download that was cut short leaves the folder
    without some of them."""
    try:
        folder = Path(snapshot_download(model["repo"], revision=model["revision"], allow_patterns=WEIGHT_FILES, local_files_only=True))
    except LocalEntryNotFoundError:
        return None
    index = folder / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    return str(folder) if (folder / "config.json").exists() and all((folder / shard).exists() for shard in shards) else None


def download(model: dict, progress: dict):
    """Fetches the model's weights, keeping `progress["done"]`, the bytes so far, current; returns their folder."""

    class Progress(tqdm):
        """huggingface_hub reports to progress bars. It counts bytes received and bytes written on separate ones; whichever
        is further along is the progress."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "disable": True})
            self.counts_bytes = kwargs.get("unit") == "B"

        def update(self, n=1):
            self.n += n or 0
            if self.counts_bytes:
                progress["done"] = max(progress["done"], self.n)

    return snapshot_download(model["repo"], revision=model["revision"], allow_patterns=WEIGHT_FILES, tqdm_class=Progress)
