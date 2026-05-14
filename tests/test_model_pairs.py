"""Validates docker/proxy/model_pairs.json — OpenRouter inference allowlist.

The proxy njs runtime isn't exercised in this Python CI; this test checks the
data file is well-formed (typos, duplicates, empty list).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PAIRS_PATH = Path(__file__).resolve().parent.parent / "docker" / "proxy" / "model_pairs.json"


@pytest.fixture(scope="module")
def allowed_models() -> list[str]:
    doc = json.loads(PAIRS_PATH.read_text())
    return doc["allowed_models"]


def test_model_pairs_file_parses(allowed_models: list[str]) -> None:
    assert isinstance(allowed_models, list)
    assert len(allowed_models) > 0


def test_each_model_id_non_empty_slug(allowed_models: list[str]) -> None:
    for m in allowed_models:
        assert isinstance(m, str) and m.strip() == m and m
        assert "/" in m, m


def test_no_duplicate_model_ids(allowed_models: list[str]) -> None:
    assert len(allowed_models) == len(set(allowed_models))


def test_openrouter_slug_shape(allowed_models: list[str]) -> None:
    """OpenRouter slugs are typically lowercase org/model (no Chutes -TEE suffix)."""
    for m in allowed_models:
        assert m == m.lower(), m
