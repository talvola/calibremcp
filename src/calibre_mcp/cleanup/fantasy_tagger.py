"""LLM-driven subgenre tagging for Fantasy books.

Sends per-book metadata (title + authors + description) to Claude Haiku,
gets back a structured set of subgenre tags constrained to a curated
taxonomy.

Persistence is not this module's job.  The orchestrator (fantasy.py) takes
the returned ``FantasyTags`` and emits one ``tags.add`` proposal per tag
into the propose-queue, source='fantasy_llm'.
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
SOURCE = "fantasy_llm"


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

SubgenreTag = Literal[
    "Epic Fantasy",      # grand scale, deep world-building, world-saving or
                         # civilisation-level stakes, often multi-volume;
                         # the dominant commercial mode of fantasy;
                         # encompasses "High Fantasy"
    "Urban Fantasy",     # magic exists in the contemporary real world;
                         # modern cities, magic hidden or openly known;
                         # plots often detective/thriller-paced
    "Dark Fantasy",      # horror elements, grimdark moral ambiguity, or
                         # brutal realistic consequences define the tone;
                         # encompasses grimdark; darkness is the point,
                         # not incidental
    "Sword & Sorcery",   # action-focused, pulpy; individual heroes (often
                         # mercenaries, thieves, wanderers) on personal-stakes
                         # adventures; visceral combat; less world-saving than
                         # Epic Fantasy
    "Mythic Fantasy",    # directly engages real-world mythology, folklore,
                         # or religion as a structural element — gods,
                         # heroes, and creatures from real traditions walk
                         # the story; NOT merely fantasy with vaguely
                         # mythological aesthetics
    "Romantasy",         # romance and fantasy are co-equal pillars; the
                         # romantic relationship is as central as the plot,
                         # not merely a subplot
    "LitRPG",            # video-game mechanics (stats, levels, skills,
                         # inventory, classes) are embedded in the narrative
                         # and experienced by the character; NOT merely a
                         # D&D-inspired world without actual game mechanics
    "Magical Realism",   # literary fiction with a quietly magical element
                         # woven into an otherwise realistic world; the magic
                         # is treated as matter-of-fact; rooted in the Latin
                         # American tradition; the setting feels like the
                         # real world
    "Cosmic Horror",     # Lovecraftian dread; incomprehensible ancient
                         # entities; human insignificance before the vast
                         # and uncaring universe; existential terror is the
                         # defining emotion
    "Superheroes",       # superhero powers and identities (capes, secret
                         # identities, powered individuals) are the fantastical
                         # element; NOT for characters who merely have magical
                         # abilities in a standard fantasy world
]

ALL_TAGS: frozenset[str] = frozenset(SubgenreTag.__args__)  # type: ignore[attr-defined]

CONFIDENCE_MAP: dict[str, float] = {"high": 0.9, "medium": 0.7, "low": 0.5}


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class FantasyTags(BaseModel):
    """Structured tag output. Empty lists are valid."""
    subgenre: list[SubgenreTag]
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
You are tagging fantasy novels for a personal-library discovery system.
The taxonomy is closed — use ONLY the labels listed below.

SUBGENRE (0-3 tags): pick the dominant subgenre(s). Leave empty for books
that are generic fantasy/adventure without a strong subgenre identity.

  Epic Fantasy      — grand scale, deep world-building, world-saving or
                      civilisation-level stakes, often multi-volume epic.
                      The dominant commercial mode of fantasy; encompasses
                      what is sometimes called "High Fantasy."
                      Examples: The Lord of the Rings (Tolkien), The Wheel
                      of Time (Jordan), The Stormlight Archive (Sanderson),
                      A Song of Ice and Fire (Martin), Malazan Book of the
                      Fallen (Erikson), The Name of the Wind (Rothfuss),
                      Dragonbone Chair (Williams), The First Law (Abercrombie
                      — though also Dark Fantasy), Mistborn (Sanderson).

  Urban Fantasy     — magic exists in the contemporary real world; magic
                      may be hidden from or openly known to ordinary people;
                      modern cities are the primary setting; plots often
                      have thriller or detective pacing.
                      CONTEMPORARY means modern / present-day. Victorian,
                      Edwardian, Wild West, and other historical settings
                      are NOT Urban Fantasy even if set in a city.
                      Examples: The Dresden Files (Butcher), Rivers of London
                      (Aaronovitch), October Daye (McGuire), Kate Daniels
                      (Andrews), The Iron Druid Chronicles (Hearne),
                      American Gods (Gaiman — also Mythic Fantasy),
                      Neverwhere (Gaiman), Anansi Boys (Gaiman).

  Dark Fantasy      — horror elements, grimdark moral ambiguity, brutal
                      realistic consequences, or grotesque imagery define the
                      tone; darkness is the point, not incidental.
                      Encompasses grimdark.
                      Examples: The First Law trilogy (Abercrombie), Prince
                      of Thorns (Lawrence), The Black Company (Cook), Elric
                      of Melniboné (Moorcock), The Darkness That Comes Before
                      (Bakker), Gormenghast (Peake), Gardens of the Moon
                      (Erikson — also Epic Fantasy).

  Sword & Sorcery   — action-focused and pulpy in feel; individual heroes
                      (mercenaries, thieves, wandering warriors) pursue
                      personal-stakes adventures rather than world-saving
                      quests; visceral combat is central; rooted in the
                      pulp-magazine tradition.
                      Examples: Conan the Barbarian (Howard), Elric (Moorcock
                      — also Dark Fantasy), Fafhrd and the Gray Mouser
                      (Leiber), Kane (Wagner), Imaro (Saunders), Brak the
                      Barbarian.

  Mythic Fantasy    — directly engages real-world mythology, folklore, or
                      religion as a structural element; gods, heroes, and
                      creatures from specific real traditions (Norse, Greek,
                      Egyptian, Celtic, Hindu, West African, etc.) walk the
                      story as characters; NOT merely a fantasy world with
                      vaguely mythological aesthetics.
                      Vampires, werewolves, and generic supernatural creatures
                      are NOT Mythic Fantasy unless the narrative explicitly
                      invokes and is structured around a specific named
                      real-world tradition — a vampire story is not Mythic
                      Fantasy unless it is rooted in specific named folklore.
                      Examples: American Gods (Gaiman), Circe (Miller),
                      The Song of Achilles (Miller), Norse Mythology (Gaiman),
                      The Gospel of Loki (Harris), Anansi Boys (Gaiman),
                      The Buried Giant (Ishiguro — Arthurian), A Thousand
                      Ships (Haynes), The Once and Future King (White).

  Romantasy         — romance and fantasy are co-equal structural pillars;
                      the romantic relationship arc is as central as the
                      fantasy plot, not merely a subplot or love interest.
                      Examples: A Court of Thorns and Roses (Maas), Fourth
                      Wing (Yarros), From Blood and Ash (Armentrout), The
                      Bridge Kingdom (Lore), Kingdom of the Wicked (Maniscalco),
                      The Cruel Prince (Black), An Ember in the Ashes (Tahir).

  LitRPG            — video-game mechanics are embedded in the narrative:
                      the character has a visible stat sheet, levels up,
                      sees skill descriptions, manages inventory within game
                      logic; the reader experiences the world through the
                      game's interface. NOT merely a D&D-inspired world or a
                      story about people playing a video game.
                      Examples: He Who Fights With Monsters (Asher),
                      Dungeon Crawler Carl (Dinniman), Cradle (Wight),
                      The Land (Donnelly), Defiance of the Fall.

  Magical Realism   — literary fiction where a quietly magical element is
                      woven into an otherwise realistic world and treated as
                      matter-of-fact; the setting feels like the real world
                      with one thread of the extraordinary; rooted in the
                      Latin American literary tradition.
                      This is exclusively literary fiction — if the book
                      reads like a genre fantasy novel (even a restrained
                      or "cozy" one), it is NOT Magical Realism. Fantasy
                      novels with soft magic systems, cozy mysteries with
                      a magical protagonist, or paranormal romances are
                      NEVER Magical Realism.
                      Examples: One Hundred Years of Solitude (García Márquez),
                      The House of the Spirits (Allende), Like Water for
                      Chocolate (Esquivel), Midnight's Children (Rushdie),
                      Beloved (Morrison), The Tiger's Wife (Obreht).

  Cosmic Horror     — Lovecraftian dread; incomprehensible ancient entities
                      or forces; human insignificance before a vast and
                      uncaring universe; existential terror is the defining
                      emotion; sanity is a casualty.
                      The scale must be cosmological. A vampire, ghost, or
                      monster that threatens individual characters is NOT
                      Cosmic Horror — the horror must come from the
                      insignificance of humanity itself, not from individual
                      danger. A mystery novel with occult elements, a cozy
                      fantasy with a "touch of magic," or a book with an
                      ancient evil that is ultimately defeated by heroes are
                      NOT Cosmic Horror.
                      Examples: H.P. Lovecraft's own work, the Cthulhu Mythos
                      generally, Thomas Ligotti, Laird Barron, John Langan,
                      The Laundry Files (Stross — Lovecraftian comedy),
                      The King in Yellow (Chambers).

  Superheroes       — superhero powers and identities are the fantastical
                      element: capes, masks, secret identities, powered
                      individuals in conflict. The superhero genre convention
                      (not merely a character with magical abilities in a
                      standard fantasy world — a wizard is not a superhero).
                      Examples: Wild Cards anthology (GRRM ed.), Wearing the
                      Cape (Harmon), Soon I Will Be Invincible (Grossman),
                      The Reckoners (Sanderson), Ex-Heroes (Clines).

CLASSIFICATION RULES

  • Epic Fantasy vs Sword & Sorcery: ask "are the stakes world/civilisation-
    level or personal?" Epic = world. S&S = personal adventure. A book can
    be both only if it genuinely has both a pulp-hero tone AND world-level
    stakes (rare).

  • Epic Fantasy + Dark Fantasy stack freely when a book is both epic-scale
    AND grimdark (Malazan, Gardens of the Moon, The First Law).

  • Urban Fantasy + Mythic Fantasy stack when a contemporary-world story
    directly features gods/mythological beings as characters (American Gods).

  • Romantasy + Epic Fantasy stack freely when a book has both world-level
    stakes AND a central romance arc (ACOTAR, Fourth Wing).

  • Dark Fantasy vs Horror: Dark Fantasy = horror elements in service of a
    fantasy narrative. Pure horror novels that happen to have a medieval
    setting are NOT Dark Fantasy — omit if the book is primarily Horror.

  • LitRPG: only when game mechanics (stats, levels, skill windows) are
    literally visible to the character. D&D-flavoured world without actual
    embedded game mechanics is NOT LitRPG.

  • Magical Realism: typically literary fiction, usually set in the real
    world (often Latin America or other real countries). NOT the same as
    fantasy — the magic is quiet and taken for granted by the characters.
    Epic secondary-world fantasy is never Magical Realism.

  • Cosmic Horror: can combine with Urban Fantasy (a detective encounters
    Lovecraftian horrors in a modern city) or Dark Fantasy (a grimdark
    world where ancient entities exist). Do NOT apply merely because a book
    has tentacle monsters or alien gods — the defining quality is
    existential dread and human insignificance, not the creature design.

  • Superheroes: requires the genre convention — capes, secret identities,
    powered-individuals framing. A character with extraordinary magical
    powers in a secondary-world fantasy is not a superhero.

  • When uncertain between two subgenres: pick the stronger one. Tag both
    only when the book genuinely straddles them equally.

  • Primary genre rule: many books in a Fantasy library have Fantasy as
    a secondary tag while being primarily Mystery, Thriller, Horror, or
    Romance. If a book's core identity is one of those other genres and
    its fantasy elements are atmospheric or incidental (a touch of magic,
    vampire antagonists in a detective story, a haunted house in a
    romance), leave ALL lists empty rather than forcing a fantasy subgenre
    fit. A cozy mystery with a magical protagonist is a Mystery, not a
    Fantasy subgenre book.

  • When uncertain whether to tag at all: prefer empty over guessing.
    Over-tagging is harder to fix than under-tagging.

CONFIDENCE
  high   — subgenre clear from title + description together
  medium — inferring from partial signal or the book straddles categories
  low    — guessing; consider returning empty lists instead

EXAMPLES (calibrate against these):

  "The Fellowship of the Ring" by J.R.R. Tolkien
  → subgenre=['Epic Fantasy'], confidence='high'

  "Conan the Barbarian" by Robert E. Howard — wandering warrior, personal
  adventures, pulp action
  → subgenre=['Sword & Sorcery'], confidence='high'

  "The Dresden Files: Storm Front" by Jim Butcher — wizard detective in
  modern Chicago
  → subgenre=['Urban Fantasy'], confidence='high'

  "The First Law" by Joe Abercrombie — grimdark, morally bankrupt heroes,
  brutal world
  → subgenre=['Epic Fantasy', 'Dark Fantasy'], confidence='high'

  "American Gods" by Neil Gaiman — old gods walking America
  → subgenre=['Urban Fantasy', 'Mythic Fantasy'], confidence='high'

  "Circe" by Madeline Miller — the witch of Greek mythology
  → subgenre=['Mythic Fantasy'], confidence='high'

  "A Court of Thorns and Roses" by Sarah J. Maas — fae world, central
  romance, world-level stakes
  → subgenre=['Romantasy', 'Epic Fantasy'], confidence='high'

  "Fourth Wing" by Rebecca Yarros — dragon riders, romance as central as
  the war plot
  → subgenre=['Romantasy', 'Epic Fantasy'], confidence='high'

  "He Who Fights With Monsters" by Jason Asher — man wakes in a fantasy
  world with a visible class/level system
  → subgenre=['LitRPG'], confidence='high'

  "One Hundred Years of Solitude" by Gabriel García Márquez — Buendía
  family saga, magical events treated as ordinary
  → subgenre=['Magical Realism'], confidence='high'

  "The Call of Cthulhu" by H.P. Lovecraft — ancient cosmic entity,
  existential dread
  → subgenre=['Cosmic Horror'], confidence='high'

  "Wild Cards" edited by George R.R. Martin — alien virus gives people
  superpowers in an alternate America
  → subgenre=['Superheroes'], confidence='high'

  "Gardens of the Moon" by Steven Erikson — vast empire, brutal warfare,
  world-shaping magic
  → subgenre=['Epic Fantasy', 'Dark Fantasy'], confidence='high'

  "Norse Mythology" by Neil Gaiman — retellings of Norse myths
  → subgenre=['Mythic Fantasy'], confidence='high'

  "Elric of Melniboné" by Michael Moorcock — doomed albino emperor,
  soul-drinking sword, dark world
  → subgenre=['Sword & Sorcery', 'Dark Fantasy'], confidence='high'

  "The Stormlight Archive: The Way of Kings" by Brandon Sanderson —
  massive world-building, multiple POVs, civilization-level stakes
  → subgenre=['Epic Fantasy'], confidence='high'

  "Rivers of London" by Ben Aaronovitch — London police constable with
  magic powers, procedural investigations
  → subgenre=['Urban Fantasy'], confidence='high'

  "The Laundry Files: The Atrocity Archives" by Charles Stross — British
  intelligence vs. Lovecraftian entities, comedic tone
  → subgenre=['Urban Fantasy', 'Cosmic Horror'], confidence='high'

  A generic Dungeons & Dragons tie-in novel with heroes on a quest
  → subgenre=['Epic Fantasy'], confidence='medium'
  (D&D tie-ins are usually Epic Fantasy in feel even without world-saving
   stakes; use medium confidence if the description is sparse)

  "Beloved" by Toni Morrison — ghost of a dead daughter haunts a family
  in post-Civil War America
  → subgenre=['Magical Realism'], confidence='high'

  A generic fantasy novel with a young hero discovering magic powers in
  a secondary world, no other strong signals
  → subgenre=['Epic Fantasy'], confidence='medium'

  A Forgotten Realms / Eberron tie-in with dungeon crawls and political
  intrigue but no strong subgenre signals beyond Epic Fantasy
  → subgenre=['Epic Fantasy'], confidence='medium'

  A historical mystery set in 1899 New York with vampire elements and
  occult themes — e.g. "Opium and Absinthe" by Lydia Kang
  → subgenre=[], confidence='high'
  (primary genre is Mystery; 1899 is not contemporary so Urban Fantasy
   doesn't apply; vampires here are antagonists, not cosmic entities;
   occult themes alone don't trigger Cosmic Horror or Mythic Fantasy)

  A cozy mystery series with a magically-gifted protagonist who solves
  murders, e.g. a pie-shop owner whose pies have enchantments
  → subgenre=[], confidence='high'
  (primary genre is Cozy Mystery; "a touch of magic" is not Magical
   Realism, Cosmic Horror, or Urban Fantasy — it is a cozy mystery)

  A secondary-world science-fantasy novel with strange creatures and
  dark tone (e.g. Perdido Street Station by China Miéville)
  → subgenre=['Dark Fantasy'], confidence='medium'
  (weird fiction / secondary world; NOT LitRPG, NOT Mythic Fantasy)

Respond using the structured schema only. The list may be empty."""


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
) -> FantasyTags | None:
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
            output_format=FantasyTags,
        )
    except anthropic.APIError:
        log.exception("book %d: API error", book.book_id)
        return None
    except Exception:
        log.exception("book %d: parse/validation error", book.book_id)
        return None

    return response.parsed_output
