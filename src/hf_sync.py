"""Sync the git-ignored local datasets (data/local: Alpaca minute bars, auction prices, samples) with a
PRIVATE Hugging Face dataset repo, so a fresh session does not re-download them from Alpaca.

    python src/hf_sync.py pull      # download whatever the repo has into data/local (run first in a new session)
    python src/hf_sync.py push      # upload data/local (only files that changed; creates the repo if needed)

Needs HF_TOKEN (write access) in the environment. Repo id: HF_DATA_REPO (default <your user>/nooptoo-market-data).
Monthly partitions are immutable once a month closes, so pushes after the first only send new months.
"""
import os
import sys

from huggingface_hub import HfApi, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(ROOT, "data", "local")
SKIP = (".ratelimit", ".pkl")          # lock file and derived caches are rebuilt locally


def repo_id(api):
    rid = os.environ.get("HF_DATA_REPO")
    if rid:
        return rid
    return f"{api.whoami()['name']}/nooptoo-market-data"


def push():
    api = HfApi(token=os.environ["HF_TOKEN"])
    rid = repo_id(api)
    api.create_repo(rid, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=LOCAL, repo_id=rid, repo_type="dataset",
                      ignore_patterns=[f"*{s}" for s in SKIP], commit_message="sync data/local")
    print("pushed to", rid)


def pull():
    api = HfApi(token=os.environ["HF_TOKEN"])
    rid = repo_id(api)
    snapshot_download(rid, repo_type="dataset", local_dir=LOCAL, token=os.environ["HF_TOKEN"])
    print("pulled", rid, "into", LOCAL)


if __name__ == "__main__":
    {"push": push, "pull": pull}[sys.argv[1] if len(sys.argv) > 1 else "pull"]()
