"""HuggingFace Hub authentication helper."""
from __future__ import annotations
import os


def hf_login(token: str | None = None) -> None:
    """Login to HuggingFace Hub.

    Resolution order:
      1. ``token`` argument
      2. ``HF_TOKEN`` environment variable
      3. Cached token from a previous ``huggingface-cli login``
      4. Interactive browser login
    """
    from huggingface_hub import login

    resolved = token or os.environ.get("HF_TOKEN")
    if not resolved:
        try:
            from huggingface_hub import HfFolder
            resolved = HfFolder.get_token()
        except Exception:
            pass

    if resolved:
        login(token=resolved, add_to_git_credential=False)
        print("✓ Logged in to HuggingFace Hub (token)")
    else:
        print("No HF_TOKEN found — starting interactive login...")
        login()
