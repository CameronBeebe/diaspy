"""Fail closed: offline experiment tests never use provider credentials or sockets."""
import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import pytest


@pytest.fixture(autouse=True)
def offline_only(monkeypatch, socket_disabled):
    for key in list(os.environ):
        if key.endswith("API_KEY") or key.endswith("API_TOKEN"):
            monkeypatch.delenv(key, raising=False)
