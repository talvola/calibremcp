"""Unit tests for the cookbook tagger module — taxonomy, prompt, and
the per-book content builder. The actual LLM call is exercised in the
orchestrator tests via a fake client; this file covers the pieces that
are deterministic without an API."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibre_mcp.cleanup import cookbook_tagger as ct
from calibre_mcp.cleanup.cookbook_tagger import (
    ALL_TAGS,
    CONFIDENCE_MAP,
    BookContext,
    CookbookTags,
    CuisineTag,
    DietaryTag,
    TechniqueTag,
    _build_user_content,
    _media_type_for,
)

# ---------------------------------------------------------------------------
# Taxonomy invariants
# ---------------------------------------------------------------------------


def test_taxonomy_sizes() -> None:
    """If we add or remove labels, this test breaks deliberately so we
    notice. Erik's guidance + strategy memo target ~30 facets; current
    is 16+10+6 = 32."""
    assert len(CuisineTag.__args__) == 16  # type: ignore[attr-defined]
    assert len(TechniqueTag.__args__) == 10  # type: ignore[attr-defined]
    assert len(DietaryTag.__args__) == 6  # type: ignore[attr-defined]


def test_specific_asian_cuisines_present() -> None:
    """Erik's late-feedback fix: 'Asian' alone is too generic; specific
    Asian cuisines (Japanese, Chinese, Korean, Vietnamese, Thai, Filipino,
    Indian) must all be selectable so the model doesn't fall back to the
    umbrella for clearly-single-cuisine books."""
    cuisines = set(CuisineTag.__args__)  # type: ignore[attr-defined]
    for required in ("Japanese", "Chinese", "Korean", "Vietnamese", "Thai", "Filipino", "Indian"):
        assert required in cuisines, f"missing {required!r} in cuisine taxonomy"
    assert "Asian" in cuisines  # umbrella stays for Burmese / Malaysian / etc.


def test_all_tags_is_union_of_three_facets() -> None:
    expected = (
        set(CuisineTag.__args__)  # type: ignore[attr-defined]
        | set(TechniqueTag.__args__)  # type: ignore[attr-defined]
        | set(DietaryTag.__args__)  # type: ignore[attr-defined]
    )
    assert expected == ALL_TAGS


def test_confidence_map_covers_all_levels() -> None:
    assert set(CONFIDENCE_MAP) == {"high", "medium", "low"}
    # Monotonic decreasing.
    assert CONFIDENCE_MAP["high"] > CONFIDENCE_MAP["medium"] > CONFIDENCE_MAP["low"]
    # All in [0, 1] for the propose-queue's confidence column.
    for v in CONFIDENCE_MAP.values():
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# CookbookTags schema validation
# ---------------------------------------------------------------------------


def test_cookbook_tags_accepts_valid() -> None:
    tags = CookbookTags(cuisine=["Italian"], technique=["Baking"], dietary=[], confidence="high")
    assert tags.cuisine == ["Italian"]


def test_cookbook_tags_rejects_invalid_label() -> None:
    """A label outside the taxonomy must raise — this is the schema-level
    constraint enforcement we depend on instead of post-processing."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CookbookTags(cuisine=["Klingon"], technique=[], dietary=[], confidence="high")


def test_cookbook_tags_rejects_invalid_confidence() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CookbookTags(cuisine=[], technique=[], dietary=[], confidence="absolute")


def test_cookbook_tags_allows_all_empty_lists() -> None:
    """Hawaiian-fusion-style honest-empty case must validate."""
    tags = CookbookTags(cuisine=[], technique=[], dietary=[], confidence="medium")
    assert all(not lst for lst in (tags.cuisine, tags.technique, tags.dietary))


# ---------------------------------------------------------------------------
# _build_user_content — image + text assembly
# ---------------------------------------------------------------------------


def _make_jpeg(path: Path, n_bytes: int = 100) -> Path:
    """Minimal JPEG-shaped bytes — header + filler. Just needs to be
    readable, not actually decode."""
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * (n_bytes - 4))
    return path


def test_build_user_content_with_cover(tmp_path: Path) -> None:
    cover = _make_jpeg(tmp_path / "cover.jpg")
    book = BookContext(
        book_id=1, title="Test", authors=("Author A",),
        description="A test description", cover_path=cover,
    )
    blocks = _build_user_content(book)
    # First block is the image, second is text — better attention pattern.
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["media_type"] == "image/jpeg"
    assert blocks[0]["source"]["type"] == "base64"
    assert blocks[1]["type"] == "text"
    assert "Test" in blocks[1]["text"]
    assert "Author A" in blocks[1]["text"]


def test_build_user_content_without_cover(tmp_path: Path) -> None:
    book = BookContext(
        book_id=1, title="No Cover Book", authors=("X",),
        description="desc", cover_path=None,
    )
    blocks = _build_user_content(book)
    assert len(blocks) == 1  # text only
    assert blocks[0]["type"] == "text"


def test_build_user_content_falls_back_when_cover_missing(tmp_path: Path) -> None:
    """cover_path points somewhere but file isn't there — text-only fallback
    rather than crashing."""
    book = BookContext(
        book_id=1, title="X", authors=(), description=None,
        cover_path=tmp_path / "nope.jpg",  # not created
    )
    blocks = _build_user_content(book)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"


def test_build_user_content_truncates_long_description(tmp_path: Path) -> None:
    book = BookContext(
        book_id=1, title="X", authors=(), cover_path=None,
        description="x" * 5000,
    )
    blocks = _build_user_content(book)
    text = blocks[0]["text"]
    # 1500 cap + ellipsis. Bound the rendered text length to confirm cap.
    assert len(text) < 2000
    assert text.endswith("…")


def test_build_user_content_handles_missing_authors(tmp_path: Path) -> None:
    book = BookContext(book_id=1, title="X", authors=(), description="d", cover_path=None)
    text = _build_user_content(book)[0]["text"]
    assert "(unknown)" in text


def test_build_user_content_handles_missing_description(tmp_path: Path) -> None:
    book = BookContext(book_id=1, title="X", authors=("a",), description=None, cover_path=None)
    text = _build_user_content(book)[0]["text"]
    assert "(no description available)" in text


# ---------------------------------------------------------------------------
# _media_type_for — extension → MIME mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("cover.jpg", "image/jpeg"),
        ("cover.JPG", "image/jpeg"),
        ("cover.jpeg", "image/jpeg"),
        ("cover.png", "image/png"),
        ("cover.PNG", "image/png"),
        ("cover.gif", "image/gif"),
        ("cover.webp", "image/webp"),
        ("cover.unknown", "image/jpeg"),  # default — Calibre uses .jpg overwhelmingly
    ],
)
def test_media_type_for(name: str, expected: str) -> None:
    assert _media_type_for(Path(name)) == expected


# ---------------------------------------------------------------------------
# tag_book — graceful API-error handling
# ---------------------------------------------------------------------------


def test_tag_book_returns_none_on_api_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Network blip / rate limit / 5xx — orchestrator should be able to
    log + continue rather than abort the whole 1,238-book batch."""
    import anthropic

    class FailingMessages:
        def parse(self, **_kwargs):
            raise anthropic.APIError("boom", request=None, body=None)

    class FakeClient:
        def __init__(self) -> None:
            self.messages = FailingMessages()

    book = BookContext(book_id=1, title="X", authors=(), description="d", cover_path=None)
    result = ct.tag_book(FakeClient(), book)
    assert result is None
