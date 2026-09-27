"""Sync the git-ignored local datasets (data/local: Alpaca minute bars, auction prices, samples) with a
PRIVATE Hugging Face dataset repo, so a fresh session does not re-download them from Alpaca.

    python src/hf_sync.py pull      # download whatever the repo has into data/local (run first in a new session)
    python src/hf_sync.py pull models   # only the LightGBM models (what the paper runner needs)
    python src/hf_sync.py push      # upload data/local (only files that changed; creates the repo if needed)

Needs HF_TOKEN (write access) in the environment. Repo id: HF_DATA_REPO (default <your user>/nooptoo-market-data).
Monthly partitions are immutable once a month closes, so pushes after the first only send new months.
The LightGBM models in data/models (needed by study3_timing.py and paper_overnight.py) are stored in the repo
under models/; pull copies them back into data/models.
"""
import os
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(ROOT, "data", "local")
MODELS = os.path.join(ROOT, "data", "models")
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
    if os.path.isdir(MODELS):
        api.upload_folder(folder_path=MODELS, path_in_repo="models", repo_id=rid, repo_type="dataset",
                          commit_message="sync data/models")
    print("pushed to", rid)


def pull(only=None):
    api = HfApi(token=os.environ["HF_TOKEN"])
    rid = repo_id(api)
    snapshot_download(rid, repo_type="dataset", local_dir=LOCAL, token=os.environ["HF_TOKEN"],
                      allow_patterns=[f"{only}/*"] if only else None)
    src = os.path.join(LOCAL, "models")
    if os.path.isdir(src):
        os.makedirs(MODELS, exist_ok=True)
        for f in os.listdir(src):
            shutil.copy2(os.path.join(src, f), os.path.join(MODELS, f))
    print("pulled", rid, "into", LOCAL)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "pull"
    if what == "pull":
        pull(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        push()
