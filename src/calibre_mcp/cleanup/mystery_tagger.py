"""LLM-driven subgenre tagging for Mystery books.

Sends per-book metadata (title + authors + description) to Claude Haiku,
gets back a structured set of subgenre / character tags constrained to a
curated taxonomy.  No vision — description is the dominant signal for genre.

Persistence is not this module's job.  The orchestrator (mystery.py) takes
the returned ``MysteryTags`` and emits one ``tags.add`` proposal per tag
into the propose-queue, source='mystery_llm'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

_ENV_LOCAL = Path(__file__).resolve().parents[3] / ".env.local"
if _ENV_LOCAL.exists():
    load_dotenv(_ENV_LOCAL, override=False)

log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
SOURCE = "mystery_llm"


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

# Subgenre: the primary classification dimension.  0–2 tags per book.
# When the book straddles two equally, both are valid; but don't force a tag
# if the book is a generic mystery without a strong subgenre identity.
SubgenreTag = Literal[
    "Cozy Mystery",          # amateur sleuth, light tone, no graphic violence;
                              # hobby/village/small-town setting; often humorous
    "Police Procedural",     # professional law enforcement, investigative process
                              # as the book's engine
    "Noir",                  # cynical protagonist, moral ambiguity, gritty urban
                              # setting; dark atmosphere is the defining quality
    "Historical Mystery",    # set in a clearly defined historical period
                              # (roughly pre-1960); period setting is central,
                              # not just incidental
    "Paranormal Mystery",    # supernatural element is integral to the mystery
                              # (ghosts, psychic detectives, vampires etc.);
                              # NOT for books that are merely atmospheric
    "Legal Thriller",        # courtroom / legal system as the arena; lawyer or
                              # DA protagonist; verdict is the resolution
    "Psychological Thriller", # unreliable narrator, psychological manipulation,
                              # suspense from inside the protagonist's head;
                              # the threat is mental as much as physical
    "True Crime",            # non-fiction accounts of real crimes and investigations
    "Locked Room Mystery",   # impossible-crime subgenre; the puzzle is HOW,
                              # not just WHO
]

# Character dimension: flag when the sleuth / investigator is female.
# Apply whenever the lead detective / amateur sleuth / protagonist who
# drives the investigation is a woman — regardless of subgenre.
CharacterTag = Literal[
    "Women Sleuths",
]

ALL_TAGS: frozenset[str] = (
    frozenset(SubgenreTag.__args__)   # type: ignore[attr-defined]
    | frozenset(CharacterTag.__args__)  # type: ignore[attr-defined]
)

CONFIDENCE_MAP: dict[str, float] = {"high": 0.9, "medium": 0.7, "low": 0.5}


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class MysteryTags(BaseModel):
    """Structured tag output. Empty lists are valid."""
    subgenre: list[SubgenreTag]
    character: list[CharacterTag]
    confidence: Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Book context
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BookContext:
    book_id: int
    title: str
    authors: tuple[str, ...]
    description: str | None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are tagging mystery and crime novels for a personal-library discovery system.
The taxonomy is closed — use ONLY the labels listed below.

SUBGENRE (0-2 tags): pick the dominant subgenre(s). Leave empty for books
that are generic mystery/crime without a strong subgenre identity.

  Cozy Mystery       — amateur sleuth, light tone, no graphic violence;
                       hobby/village/small-town; often humorous or whimsical.
                       Examples: Agatha Raisin, Hamish Macbeth, the Cat Who...
                       series, any "bakery mystery" or "knitting mystery."

  Police Procedural  — professional law enforcement as the engine; the
                       investigation process (forensics, procedure, hierarchy)
                       is central. Examples: Inspector Morse, Rebus, Wambaugh,
                       Tana French, Michael Connelly's Bosch novels.

  Noir               — cynical or morally compromised protagonist; dark,
                       gritty, often urban atmosphere; moral ambiguity is
                       the defining texture. Examples: Raymond Chandler,
                       James Ellroy, Megan Abbott, neo-noir like In the Woods
                       (Du Maurier-ish dread, not cozy).

  Historical Mystery — set in a clearly defined historical period, roughly
                       pre-1960, where period detail is central (not merely
                       incidental). Examples: Cadfael, Matthew Shardlake,
                       Bernie Gunther, Lord Peter Wimsey, Maisie Dobbs,
                       Amelia Peabody.

  Paranormal Mystery — supernatural element is INTEGRAL to the mystery
                       (ghosts as witnesses, psychic detectives, vampire
                       detectives, magic as investigative tool). Mere gothic
                       atmosphere or a haunted house setting does NOT qualify
                       unless the supernatural actively drives the plot.
                       Examples: Rivers of London, Dresden Files, Greywalker,
                       Sookie Stackhouse mysteries.

  Legal Thriller     — courtroom / legal system as the primary arena;
                       lawyer, prosecutor, or public defender as protagonist;
                       verdict or legal outcome is the resolution.
                       Examples: John Grisham, Scott Turow, Perry Mason,
                       Michael Connelly's Bosch-adjacent Lincoln Lawyer.

  Psychological Thriller — suspense derived from inside the protagonist's
                       (or antagonist's) mind; unreliable narrator, paranoia,
                       gaslighting, or psychological manipulation as the
                       central mechanism. Examples: Gone Girl, The Girl on the
                       Train, Behind Closed Doors, You (Kepnes).

  True Crime         — non-fiction: real crimes, real investigations, real
                       perpetrators. Examples: In Cold Blood, I'll Be Gone
                       in the Dark, Say Nothing, Midnight in the Garden of
                       Good and Evil. Fictional crime novels are NOT True Crime.

  Locked Room Mystery — impossible-crime subgenre; the puzzle is HOW a
                       crime was committed in a sealed space, not merely who
                       did it. Examples: classic John Dickson Carr, Ellery
                       Queen, recent Jaap de Haas, some Agatha Christie.

CHARACTER (0-1 tags):

  Women Sleuths      — the lead detective / amateur sleuth / protagonist who
                       DRIVES the investigation is a woman. Apply regardless
                       of subgenre. Examples: Kinsey Millhone (Sue Grafton),
                       V.I. Warshawski (Sara Paretsky), Vera Stanhope (Ann
                       Cleeves), Mma Ramotswe (Alexander McCall Smith),
                       Agatha Raisin, Harriet Vane, Maisie Dobbs.
                       Do NOT apply if the female character is a secondary
                       investigator, a victim, or shares lead billing equally
                       with a male character.

CLASSIFICATION RULES

  • Cozy Mystery vs Police Procedural: a female amateur detective in a
    village is Cozy; a female detective on the force is Police Procedural.
    Both can combine with Women Sleuths.

  • Cozy Mystery vs Paranormal Mystery: a cozy with a ghost helper is BOTH;
    a paranormal urban-fantasy detective is Paranormal Mystery only (usually
    too dark for Cozy).

  • Historical Mystery vs other subgenres: Historical Mystery stacks freely
    with others (e.g. Historical Mystery + Police Procedural for a 1930s
    Scotland Yard novel).

  • Psychological Thriller vs Noir: Noir = atmosphere and moral ambiguity
    from a hard-boiled detective's perspective. Psychological Thriller =
    suspense from unreliable narration or mental manipulation, often without
    a traditional detective. If both apply, tag both.

  • True Crime: strictly non-fiction. A novel "based on true events" is NOT
    True Crime — tag its subgenre instead.

  • When uncertain between two subgenres: pick the stronger one. Tag both
    only when the book genuinely straddles them equally.

  • When uncertain whether to tag at all: prefer empty over guessing.
    Over-tagging is harder to fix than under-tagging.

CONFIDENCE
  high   — subgenre clear from title + description together
  medium — inferring from partial signal or the book straddles categories
  low    — guessing; consider returning empty lists instead

EXAMPLES (calibrate against these):

  "Murder at the Vicarage" by Agatha Christie — Miss Marple investigates
  → subgenre=['Cozy Mystery'], character=['Women Sleuths'], confidence='high'

  "The Big Sleep" by Raymond Chandler — Philip Marlowe
  → subgenre=['Noir'], character=[], confidence='high'

  "The Name of the Rose" by Umberto Eco — medieval monastery murders
  → subgenre=['Historical Mystery', 'Locked Room Mystery'], character=[],
    confidence='high'

  "Gone Girl" by Gillian Flynn — unreliable narrators, marriage thriller
  → subgenre=['Psychological Thriller'], character=[], confidence='high'
  (Amy is not the investigator; Nick is; neither maps to Women Sleuths)

  "The Lincoln Lawyer" by Michael Connelly — defense attorney Mickey Haller
  → subgenre=['Legal Thriller'], character=[], confidence='high'

  "The Firm" by John Grisham
  → subgenre=['Legal Thriller'], character=[], confidence='high'

  "In Cold Blood" by Truman Capote
  → subgenre=['True Crime'], character=[], confidence='high'

  "Maisie Dobbs" by Jacqueline Winspear — 1930s female private investigator
  → subgenre=['Historical Mystery'], character=['Women Sleuths'],
    confidence='high'

  "Rivers of London" by Ben Aaronovitch — police constable + magic
  → subgenre=['Police Procedural', 'Paranormal Mystery'], character=[],
    confidence='high'

  "A is for Alibi" by Sue Grafton — Kinsey Millhone PI novels
  → subgenre=['Noir'], character=['Women Sleuths'], confidence='high'
  (Kinsey is hard-boiled, not cozy — not in a village with a hobby)

  "The Sweetness at the Bottom of the Pie" — 11-year-old girl solves a
  murder in 1950s English village
  → subgenre=['Historical Mystery', 'Cozy Mystery'],
    character=['Women Sleuths'], confidence='high'

  "Tana French's In the Woods" — Dublin Murder Squad
  → subgenre=['Police Procedural'], character=[], confidence='high'
  (dark and literary but the engine is a police investigation, not Noir's
   morally-compromised lone detective)

  "I'll Be Gone in the Dark" by Michelle McNamara — Golden State Killer
  → subgenre=['True Crime'], character=[], confidence='high'

  "The Girl with the Dragon Tattoo" by Stieg Larsson
  → subgenre=['Police Procedural'], character=['Women Sleuths'],
    confidence='medium'
  (Lisbeth is a co-lead investigator; tag Women Sleuths when she is
   clearly co-driving the investigation, which she is)

  "Rebecca" by Daphne du Maurier — gothic suspense, unnamed narrator
  → subgenre=['Psychological Thriller'], character=[], confidence='medium'
  (the female narrator is the protagonist but not a sleuth in the
   traditional sense — omit Women Sleuths when no investigation is driven)

  "And Then There Were None" by Agatha Christie — impossible situation,
  no outside detective
  → subgenre=['Locked Room Mystery'], character=[], confidence='high'

  A generic private-eye thriller with no strong subgenre signal
  → subgenre=[], character=[], confidence='medium'
  (empty is honest when no label fits strongly)

Respond using the structured schema only. Each list may be empty."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _build_user_content(book: BookContext) -> str:
    desc = (book.description or "").strip()
    if len(desc) > 1500:
        desc = desc[:1500].rstrip() + "…"
    return (
        f"Title: {book.title}\n"
        f"Authors: {', '.join(book.authors) if book.authors else '(unknown)'}\n"
        f"Description: {desc or '(no description available)'}"
    )


def tag_book(
    client: anthropic.Anthropic,
    book: BookContext,
    *,
    max_tokens: int = 512,
) -> MysteryTags | None:
    """Call Haiku (text-only). Returns None on API or parse error."""
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            ],
            messages=[{"role": "user", "content": _build_user_content(book)}],
            output_format=MysteryTags,
        )
    except anthropic.APIError:
        log.exception("book %d: API error", book.book_id)
        return None
    except Exception:
        log.exception("book %d: parse/validation error", book.book_id)
        return None

    return response.parsed_output
