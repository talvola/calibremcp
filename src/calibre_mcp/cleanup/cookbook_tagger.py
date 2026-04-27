"""LLM-driven subgenre tagging for cookbooks.

Sends per-book metadata (title + authors + description) plus the cover image
to Claude Sonnet 4.6 with vision, gets back a structured set of cuisine /
technique / dietary tags constrained to a curated taxonomy.

Why one module: taxonomy + system prompt + LLM call are tightly coupled —
changing the taxonomy means re-evaluating the prompt, and the Pydantic model
is the source of truth for both. Splitting them adds friction without
isolating anything that varies independently.

The taxonomy is a closed set of Title-Case labels (Pydantic ``Literal``
unions). The API constrains the model to these via JSON-schema enum
generated automatically from the Literal types — no string-similarity
post-processing needed.

Persistence is *not* this module's job. Callers (the orchestrator) take
the returned ``CookbookTags`` and emit one ``tags.add`` proposal per tag
into the propose-queue, with ``source='cookbook_llm'`` for filterability.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

# Load API key from project-local .env.local at import time. Bash on this
# system gates ~/.bashrc on interactive shells, so non-interactive `uv run`
# invocations don't see the export. .env.local sits next to the repo
# pyproject.toml, is gitignored (line 64 of .gitignore), and is loaded
# only if it exists — production callers can also set ANTHROPIC_API_KEY
# directly in their environment.
_ENV_LOCAL = Path(__file__).resolve().parents[3] / ".env.local"
if _ENV_LOCAL.exists():
    load_dotenv(_ENV_LOCAL, override=False)

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
SOURCE = "cookbook_llm"


# ---------------------------------------------------------------------------
# Taxonomy — the closed set of facet labels Claude is allowed to pick from.
# ---------------------------------------------------------------------------

# Cuisine: 15 labels covering the bulk of cookbook cuisine signal.
# Empty list is allowed for generic / multi-cuisine books — the prompt
# explicitly tells the model not to force a tag.
#
# Asian breakdown (Erik's feedback): bare "Asian" is too generic — most
# Asian cookbooks are clearly one cuisine. Reserve the umbrella label
# for genuine multi-Asian fusion books and prefer the specific label
# whenever possible.
CuisineTag = Literal[
    "Italian",
    "French",
    "Asian",            # umbrella ONLY for genuine multi-Asian fusion
    "Japanese",
    "Chinese",
    "Korean",
    "Vietnamese",
    "Thai",
    "Filipino",
    "Indian",
    "Mexican",
    "Mediterranean",
    "Middle Eastern",
    "American",
    "Latin American",
]

# Technique / format: distinguishing methods. Generic ``Cooking`` is NOT
# in the set — every cookbook is "cooking". Tag only when the book's
# identity is built around the technique.
TechniqueTag = Literal[
    "Baking",
    "Pastry",
    "Grilling/BBQ",
    "Slow Cooker",
    "Pressure Cooker",
    "Bread",
    "Cocktails",        # already a top-level genre tag in Erik's library;
                         # repeating here lets cookbook books co-tag
    "Preservation",
    "Fermentation",
    "One-Pot",
]

# Dietary: only flagged when the book's clear identity is the dietary
# focus (a vegan cookbook tags Vegan; a general cookbook with a vegan
# chapter does not).
DietaryTag = Literal[
    "Vegetarian",
    "Vegan",
    "Gluten-Free",
    "Keto/Low-Carb",
    "Paleo",
    "Kid-Friendly",
]

# Total facet labels: 12 + 10 + 6 = 28. Within the strategy memo's ~30
# target.

ALL_TAGS: frozenset[str] = (
    frozenset(CuisineTag.__args__)  # type: ignore[attr-defined]
    | frozenset(TechniqueTag.__args__)  # type: ignore[attr-defined]
    | frozenset(DietaryTag.__args__)  # type: ignore[attr-defined]
)


# ---------------------------------------------------------------------------
# Response shape — Claude's structured output
# ---------------------------------------------------------------------------


class CookbookTags(BaseModel):
    """Structured tag output. Empty lists are valid when no facet fits."""

    cuisine: list[CuisineTag]
    technique: list[TechniqueTag]
    dietary: list[DietaryTag]
    confidence: Literal["high", "medium", "low"]


# Tag-confidence → propose-queue confidence numeric. The propose queue's
# confidence is a 0–1 float used elsewhere for sorting and bulk-approval
# thresholds; this mapping makes LLM tags comparable to other sources.
CONFIDENCE_MAP: dict[str, float] = {"high": 0.9, "medium": 0.7, "low": 0.5}


@dataclass(frozen=True, slots=True)
class BookContext:
    """The metadata the tagger needs for one book. The orchestrator builds
    these from Calibre rows + the on-disk cover."""

    book_id: int
    title: str
    authors: tuple[str, ...]
    description: str | None
    cover_path: Path | None  # None when no cover exists on disk


# ---------------------------------------------------------------------------
# System prompt — cached, paid for once across the whole 1,238-book run.
# ---------------------------------------------------------------------------

# Designed for prompt-caching: stable across every call, no timestamps or
# per-book interpolation. Keep edits minimal between runs to preserve
# cache hits.
SYSTEM_PROMPT = """You're tagging cookbooks for a personal-library discovery system.

The taxonomy is closed and small — these are the only labels you may use:

  CUISINE: Italian, French, Asian, Japanese, Chinese, Korean, Vietnamese,
           Thai, Filipino, Indian, Mexican, Mediterranean, Middle Eastern,
           American, Latin American
  TECHNIQUE: Baking, Pastry, Grilling/BBQ, Slow Cooker, Pressure Cooker,
             Bread, Cocktails, Preservation, Fermentation, One-Pot
  DIETARY: Vegetarian, Vegan, Gluten-Free, Keto/Low-Carb, Paleo,
           Kid-Friendly

The 'Asian' umbrella is a LAST RESORT — almost every Asian cookbook is
one specific cuisine and should get the specific label. Use the umbrella
only for genuine multi-Asian fusion books that span several of the
specific cuisines (e.g. a pan-Asian noodle book, a Chinese-Thai-Japanese
home cookbook). If the book is single-cuisine (Japanese, Chinese, Korean,
Vietnamese, Thai, Filipino, Indian, etc.), use that specific label and
do NOT also add 'Asian'. If the cuisine is single but not in the
taxonomy (e.g. Burmese, Sri Lankan), then 'Asian' is the closest fit.

'American' covers the modern American home-cooking tradition (Joy of
Cooking, J. Kenji López-Alt, BBQ books, Tex-Mex, regional US — Southern,
New England, etc.). Use it sparingly — many "American" books are really
generic catch-alls and should have empty cuisine instead.

'Latin American' covers Brazilian, Argentinian, Peruvian, Cuban, etc. —
NOT Mexican (that has its own label).

Apply tags ONLY from the provided enums in the response schema. Empty lists
are valid and preferred when no tag fits well — over-tagging is worse than
under-tagging here.

Input you receive per book:
- The cover image (if available — strong cuisine signal via food
  photography style, typography, color palette).
- Title, author(s), description.

Use the cover image as a primary signal alongside the text. For cuisine
specifically, food photography style and book design often indicate
cuisine more reliably than the title alone.

Tagging rules:

CUISINE (0-2 tags): Pick the dominant cuisine(s). Leave empty if the book
is multi-cuisine, generic ("the home cook"), or a technique-focused book
not tied to a single cuisine. When in doubt between a specific cuisine
and 'Asian' (umbrella), prefer the specific one if it's clearly that
cuisine; use 'Asian' only when multiple Asian cuisines are blended.

TECHNIQUE (0-3 tags): Tag distinguishing methods/formats — Baking,
Grilling/BBQ, Pressure Cooker, etc. Do NOT tag generic "cooking" — every
cookbook is cooking. Tag a technique only when the book's identity is
built around it (a bread book → Bread; a general cookbook with a bread
chapter → no Bread tag). Cocktails is included for mixology books.

DIETARY (0-2 tags): Only flag when the book's clear identity IS the
dietary focus. A vegan cookbook → Vegan. A general cookbook with vegan
recipes mixed in → no Vegan tag. Kid-Friendly only for cookbooks
explicitly marketed for/about cooking with or for children.

CONFIDENCE: 'high' if the cuisine/technique is unambiguous from cover
+ title together. 'medium' if you're inferring from one signal alone or
the book straddles categories. 'low' if you're guessing — and consider
returning empty lists instead of low-confidence tags.

Examples — calibrate against these:

* "Pasta by Hand: A Collection of Italy's Regional Hand-Shaped Pasta"
  → cuisine=['Italian'], technique=[], dietary=[], confidence='high'
  (specific cuisine clear from title; pasta is implied by cuisine, not
   a distinguishing technique)

* "Mastering the Art of French Cooking" by Julia Child
  → cuisine=['French'], technique=[], dietary=[], confidence='high'

* "The Wok: Recipes and Techniques" — a book covering Chinese, Thai,
  Vietnamese, and Korean wok cooking
  → cuisine=['Asian'], technique=[], dietary=[], confidence='high'
  (genuine multi-Asian fusion; do NOT add four separate cuisine tags
   for each — the umbrella is the right call for true blends only)

* "Japanese Soul Cooking: Ramen, Tonkatsu, Tempura"
  → cuisine=['Japanese'], technique=[], dietary=[], confidence='high'
  (specific Asian cuisine; do NOT add 'Asian' alongside)

* "Vietnamese Home Cooking" by Charles Phan
  → cuisine=['Vietnamese'], technique=[], dietary=[], confidence='high'
  (specific Asian cuisine — Vietnamese is in the taxonomy, use it)

* "Pok Pok: Food and Stories from the Streets, Homes, and Roadside
  Restaurants of Thailand"
  → cuisine=['Thai'], technique=[], dietary=[], confidence='high'

* "I Am a Filipino: And This Is How We Cook"
  → cuisine=['Filipino'], technique=[], dietary=[], confidence='high'

* "The Mexican Home Kitchen"
  → cuisine=['Mexican'], technique=[], dietary=[], confidence='high'

* "Joy of Cooking" — a comprehensive general American home cookbook
  → cuisine=[], technique=[], dietary=[], confidence='medium'
  (truly multi-cuisine catch-all; an empty cuisine list is correct)

* "The Mediterranean Diet for Beginners"
  → cuisine=['Mediterranean'], technique=[], dietary=[], confidence='high'

* "Maangchi's Real Korean Cooking"
  → cuisine=['Korean'], technique=[], dietary=[], confidence='high'

* "Indian-ish: Recipes and Antics from a Modern American Family"
  → cuisine=['Indian', 'American'], technique=[], dietary=[], confidence='medium'
  (deliberate fusion; both cuisines are central to the book's identity)

* "Tartine Bread" by Chad Robertson — a bread-focused book
  → cuisine=[], technique=['Baking', 'Bread'], dietary=[], confidence='high'

* "Franklin Barbecue: A Meat-Smoking Manifesto"
  → cuisine=['American'], technique=['Grilling/BBQ'], dietary=[],
    confidence='high'

* "Slow Cooker Revolution" by America's Test Kitchen
  → cuisine=[], technique=['Slow Cooker'], dietary=[], confidence='high'

* "The Essential Instant Pot Cookbook"
  → cuisine=[], technique=['Pressure Cooker'], dietary=[], confidence='high'

* "The PDT Cocktail Book"
  → cuisine=[], technique=['Cocktails'], dietary=[], confidence='high'

* "The Noma Guide to Fermentation"
  → cuisine=[], technique=['Preservation', 'Fermentation'], dietary=[],
    confidence='high'

* "Plenty" by Yotam Ottolenghi (vegetarian-focused but not strictly so)
  → cuisine=['Mediterranean', 'Middle Eastern'], technique=[],
    dietary=['Vegetarian'], confidence='medium'
  (vegetarian is the book's identity even without explicit "vegetarian"
   in the title — judgment call from cover and description)

* "Thug Kitchen: The Official Cookbook" (vegan)
  → cuisine=[], technique=[], dietary=['Vegan'], confidence='high'

* "Gluten-Free on a Shoestring"
  → cuisine=[], technique=[], dietary=['Gluten-Free'], confidence='high'

* "The Complete Ketogenic Diet Cookbook"
  → cuisine=[], technique=[], dietary=['Keto/Low-Carb'], confidence='high'

* "Bouchon Bakery" by Thomas Keller (French pastry)
  → cuisine=['French'], technique=['Baking', 'Pastry'], dietary=[],
    confidence='high'

* "The Salt Fix" — a nutrition/health book about sodium that includes
  some recipes
  → cuisine=[], technique=[], dietary=[], confidence='medium'
  (not really a cookbook by identity; empty is honest)

* "Roald Dahl's Revolting Recipes" — recipes for/with children based on
  the books
  → cuisine=[], technique=[], dietary=['Kid-Friendly'], confidence='high'

* "The Food Lab: Better Home Cooking Through Science" by J. Kenji
  López-Alt — a general technique-focused American cookbook
  → cuisine=['American'], technique=[], dietary=[], confidence='medium'
  (American by default identity; no single technique dominates the book)

* "Mi Cocina: Recipes and Rapture from My Kitchen in Mexico"
  → cuisine=['Mexican'], technique=[], dietary=[], confidence='high'

* "The Forager Chef's Book of Flora" — wild-foods cookbook
  → cuisine=[], technique=['Preservation'], dietary=[], confidence='medium'
  (technique-driven, no specific cuisine; preservation is central if
   pickling/drying/curing are throughline themes — leave empty if not)

* "Death & Co: Modern Classic Cocktails"
  → cuisine=[], technique=['Cocktails'], dietary=[], confidence='high'

* "Six Seasons: A New Way with Vegetables" — vegetable-forward but not
  strictly vegetarian
  → cuisine=[], technique=[], dietary=[], confidence='medium'
  (vegetable-forward ≠ vegetarian as identity; empty is correct)

* "Veganomicon: The Ultimate Vegan Cookbook"
  → cuisine=[], technique=[], dietary=['Vegan'], confidence='high'

* "The Whole30 Cookbook" — paleo-adjacent strict-elimination diet
  → cuisine=[], technique=[], dietary=['Paleo'], confidence='high'
  (Paleo is the closest taxonomy label; use it for paleo-aligned diets)

* A cookbook focused on Cantonese / Sichuan / specifically-Chinese cooking
  → cuisine=['Chinese'], technique=[], dietary=[], confidence='high'

* "Burma: Rivers of Flavor" by Naomi Duguid (Burmese cooking — not in
  taxonomy)
  → cuisine=['Asian'], technique=[], dietary=[], confidence='medium'
  (Burmese isn't in the taxonomy; 'Asian' is the only available
   umbrella label, used here as a LAST resort for unrepresented Asian
   cuisines)

* "Aloha Kitchen: Recipes from Hawai'i" — Hawaiian fusion with
  Japanese / Filipino / Polynesian / American influences
  → cuisine=[], technique=[], dietary=[], confidence='medium'
  (Hawaiian fusion isn't really one of the listed cuisines; empty is
   honest. Don't force 'American' or 'Asian' here — Hawaiian is its
   own thing.)

The pattern: when in doubt, prefer empty over guessing. The user can
always add tags later, but bulk-removing wrong tags is more painful.

Final reminder: respond with the structured JSON schema only. Each list
field can be empty. The confidence field is required. Do not add
prose, explanations, or extra fields."""


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------


def _build_user_content(book: BookContext) -> list[dict]:
    """Assemble the per-book user-message content blocks. Image first
    (better attention pattern for vision); text second."""
    blocks: list[dict] = []

    if book.cover_path is not None and book.cover_path.exists():
        try:
            img_bytes = book.cover_path.read_bytes()
        except OSError as exc:  # cover unreadable — fall back to text-only
            log.warning("book %d: cover read failed (%s); text-only", book.book_id, exc)
        else:
            media_type = _media_type_for(book.cover_path)
            img_b64 = base64.standard_b64encode(img_bytes).decode("ascii")
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": img_b64},
            })

    desc = (book.description or "").strip()
    # Cap description to keep the per-book input bounded; cookbook
    # descriptions over ~1500 chars are usually marketing copy with
    # diminishing signal-to-noise.
    if len(desc) > 1500:
        desc = desc[:1500].rstrip() + "…"

    text = (
        f"Title: {book.title}\n"
        f"Authors: {', '.join(book.authors) if book.authors else '(unknown)'}\n"
        f"Description: {desc or '(no description available)'}"
    )
    blocks.append({"type": "text", "text": text})
    return blocks


def _media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")  # Calibre overwhelmingly uses .jpg


def tag_book(
    client: anthropic.Anthropic, book: BookContext, *, max_tokens: int = 1024,
) -> CookbookTags | None:
    """Call Claude Sonnet 4.6 with vision. Returns None on API error so
    the orchestrator can log and continue rather than abort the whole
    batch."""
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # 1h TTL: a sequential 1,238-book run at ~1-2s/call
                    # is 20-40 min; 5-min default would expire mid-run
                    # and force re-writes. 1h covers the whole pass.
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            ],
            messages=[{"role": "user", "content": _build_user_content(book)}],
            output_format=CookbookTags,
        )
    except anthropic.APIError:
        log.exception("book %d: API error", book.book_id)
        return None

    return response.parsed_output
