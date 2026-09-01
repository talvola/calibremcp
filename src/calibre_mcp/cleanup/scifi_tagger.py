"""LLM-driven subgenre tagging for Science Fiction books.

Sends per-book metadata (title + authors + description) to Claude Haiku,
gets back a structured set of subgenre / theme tags constrained to a
curated taxonomy.

Persistence is not this module's job.  The orchestrator (scifi.py) takes
the returned ``SciFiTags`` and emits one ``tags.add`` proposal per tag
into the propose-queue, source='scifi_llm'.
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
SOURCE = "scifi_llm"


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

SubgenreTag = Literal[
    "Space Opera",        # interstellar/galactic-scale adventure; empires, fleets,
                          # exotic aliens; epic scope is the defining quality —
                          # NOT merely "set in space"
    "Military SF",        # warfare as the dominant engine; soldiers, strategy,
                          # command structure, the experience of combat; the military
                          # unit and its mission drive the narrative
    "Cyberpunk",          # near-future dystopia with megacorporations, hackers,
                          # body modification, virtual/digital consciousness;
                          # gritty urban aesthetic is essential
    "Hard SF",            # scientific accuracy and technical extrapolation as the
                          # book's engine; the science is load-bearing, not hand-waved;
                          # orthogonal — stacks freely with other subgenres
    "Dystopia",           # oppressive, authoritarian, or deeply broken society as
                          # the central premise; the society EXISTS (hasn't collapsed);
                          # the book is a critique of that society
    "Post-Apocalyptic",   # civilisation has collapsed or is in terminal decline;
                          # survival, rebuilding, or coping with aftermath;
                          # NOT for pre-collapse dystopias
    "Time Travel",        # time manipulation as a PRIMARY plot mechanism, not a
                          # one-off framing device; the logic and consequences of
                          # time travel are load-bearing
    "Alternate History",  # real EARTH history diverges at a specific documented
                          # event or period; the divergence is central, not incidental;
                          # the story is grounded in our real world with one key change.
                          # NEVER apply to fictional universes (Star Wars, Dune, Gor,
                          # Middle-earth, etc.) even if their internal history diverges.
    "First Contact",      # humanity's encounter with alien intelligence as the
                          # central premise or a major structural element of the plot
    "Steampunk",          # Victorian-era or anachronistic technology aesthetic;
                          # steam-powered machines, clockwork; alternate 19th-century
                          # feel; the aesthetic IS the genre identity
]

ThemeTag = Literal[
    "AI & Robots",        # artificial intelligence, androids, or robots are CENTRAL
                          # characters or the nature/consciousness of AI is thematically
                          # central; NOT for incidental background robots or computers
]

ALL_TAGS: frozenset[str] = (
    frozenset(SubgenreTag.__args__)  # type: ignore[attr-defined]
    | frozenset(ThemeTag.__args__)   # type: ignore[attr-defined]
)

CONFIDENCE_MAP: dict[str, float] = {"high": 0.9, "medium": 0.7, "low": 0.5}


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class SciFiTags(BaseModel):
    """Structured tag output. Empty lists are valid."""
    subgenre: list[SubgenreTag]
    theme: list[ThemeTag]
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
You are tagging science fiction novels for a personal-library discovery system.
The taxonomy is closed — use ONLY the labels listed below.

SUBGENRE (0-3 tags): pick the dominant subgenre(s). Leave empty for books
that are generic SF/adventure without a strong subgenre identity, or for
media tie-ins whose subgenre is already clear from their series tag.

  Space Opera        — interstellar or galactic-scale adventure; empires,
                       fleets, alien civilisations; epic scope is the defining
                       quality, not merely being "set in space."
                       Examples: Foundation, Dune, the Culture series (Banks),
                       Honor Harrington (Weber), A Fire Upon the Deep (Vinge),
                       Hyperion (Simmons), Revelation Space (Reynolds),
                       Star Wars tie-ins (nearly always Space Opera).

  Military SF        — warfare as the book's engine; soldiers, tactics,
                       command hierarchy, and the lived experience of combat
                       are central. The military unit and its mission drive
                       the narrative, not merely the setting.
                       Examples: Starship Troopers (Heinlein), The Forever War
                       (Haldeman), Old Man's War (Scalzi), Hammer's Slammers
                       (Drake), The Mote in God's Eye combat sections, Armor
                       (Steakley). Honor Harrington = Space Opera + Military SF.

  Cyberpunk          — near-future dystopia defined by megacorporations,
                       hackers, body modification, virtual or digital
                       consciousness; gritty, urban, noir-adjacent aesthetic.
                       Examples: Neuromancer (Gibson), Snow Crash (Stephenson),
                       Altered Carbon (Morgan), Rainbows End (Vinge),
                       Pattern Recognition (Gibson).

  Hard SF            — scientific accuracy and technical extrapolation are
                       the book's engine; the science is load-bearing, not
                       hand-waved. Orthogonal: stacks freely with Space Opera,
                       First Contact, etc. when BOTH qualities are present.
                       Examples: The Martian (Weir), Red Mars (Robinson),
                       Rendezvous with Rama (Clarke), Dragon's Egg (Forward),
                       Blindsight (Watts), Seveneves (Stephenson).

  Dystopia           — an oppressive, authoritarian, or deeply broken society
                       is the central premise; the society EXISTS and functions
                       (it has not collapsed); the book is a critique of it.
                       Examples: 1984 (Orwell), Brave New World (Huxley),
                       The Handmaid's Tale (Atwood), Fahrenheit 451 (Bradbury),
                       We (Zamyatin), The Circle (Eggers).

  Post-Apocalyptic   — civilisation has collapsed or is in terminal decline;
                       survival, rebuilding, or coping with the aftermath.
                       Do NOT apply to pre-collapse dystopias.
                       Examples: The Road (McCarthy), Station Eleven (Mandel),
                       A Canticle for Leibowitz (Miller), Earth Abides (Stewart),
                       On the Beach (Shute), World War Z (Brooks).

  Time Travel        — time manipulation is a PRIMARY plot mechanism, not a
                       one-off framing device; the logic, paradoxes, or
                       consequences of time travel are load-bearing.
                       Examples: The Time Machine (Wells), Slaughterhouse-Five
                       (Vonnegut), Blackout/All Clear (Willis), The End of
                       Eternity (Asimov), Kindred (Butler), 11/22/63 (King).

  Alternate History  — real EARTH history diverges at a specific documented
                       event (a battle, assassination, invention); the story
                       is grounded in our recognisable real world with one
                       pivotal change.
                       Examples: The Man in the High Castle (PKD — Axis won
                       WWII), Fatherland (Harris), SS-GB (Deighton), Harry
                       Turtledove's WWII/Civil War novels, Eric Flint's 1632
                       series (West Virginians dropped into 17th-c. Germany),
                       The Years of Rice and Salt (Robinson).

                       NEVER apply to fictional universes set in space or on
                       other planets (Star Wars, Dune, Gor, the Culture,
                       Asimov's Foundation/Empire, Bujold's Nexus of Souls).
                       These are invented settings, NOT divergences from Earth
                       history, even if their societies resemble historical
                       Earth or their internal timelines have branching points.
                       Fantasy worlds (Middle-earth, Westeros) are secondary
                       worlds — also not Alternate History.

  First Contact      — humanity's first encounter with alien intelligence is
                       the central premise or a major structural element.
                       Examples: Contact (Sagan), Solaris (Lem), Childhood's
                       End (Clarke), Story of Your Life/Arrival (Chiang),
                       The Sparrow (Russell), Footfall (Niven & Pournelle).
                       Ender's Game = Military SF + First Contact (the alien
                       war IS a first-contact story).

  Steampunk          — Victorian-era or deliberately anachronistic technology
                       aesthetic; steam power, clockwork, goggles; alternate
                       19th-century feel; the aesthetic itself is the genre
                       identity.
                       Examples: The Difference Engine (Gibson & Sterling),
                       Mortal Engines (Reeve), The Clockwork Century (Priest),
                       Leviathan (Westerfeld).

THEME (0-1 tags):

  AI & Robots        — artificial intelligence, androids, or robots are
                       CENTRAL characters, OR the nature / consciousness /
                       rights of AI is thematically central to the book.
                       Do NOT apply for incidental background robots, ship
                       computers with minimal role, or brief AI cameos.
                       Examples: I, Robot (Asimov), Do Androids Dream of
                       Electric Sheep? (PKD), The Moon is a Harsh Mistress
                       (Heinlein — HAL-like HOLMES IV), Ancillary Justice
                       (Leckie — ship AI protagonist), Robopocalypse (Wilson),
                       Colossus: The Forbin Project, Klara and the Sun (Ishiguro).

CLASSIFICATION RULES

  • Space Opera vs Military SF: Space Opera = epic scope as the engine.
    Military SF = warfare as the engine. They combine freely when both
    apply (Honor Harrington, The Mote in God's Eye, Revelation Space).

  • Dystopia vs Post-Apocalyptic: Dystopia = society exists, is oppressive.
    Post-Apocalyptic = society has collapsed. They may stack if a collapsed
    civilisation has bred an oppressive survivors' society.

  • Hard SF: orthogonal quality — tag it whenever scientific rigour is
    load-bearing, regardless of other subgenres present.

  • Alternate History: requires divergence from real EARTH history at a
    real event. A fantasy empire, a space opera with feudal societies, or
    a planetary romance (Gor, Barsoom) is NEVER Alternate History regardless
    of how historically-flavoured its setting feels. When in doubt, omit it.

  • Time Travel: tag only when time travel is mechanically central. A
    framing story ("she remembered the past") does not qualify.

  • Media Tie-Ins: Star Wars tie-ins are almost always Space Opera. Star
    Trek tie-ins vary — apply subgenres only when the description makes
    the subgenre clear (e.g. a Star Trek novel with a time-travel plot
    → Time Travel; most others → leave subgenre empty).

  • When uncertain between two subgenres: pick the stronger one. Tag both
    only when the book genuinely straddles them equally.

  • When uncertain whether to tag at all: prefer empty over guessing.
    Over-tagging is harder to fix than under-tagging.

CONFIDENCE
  high   — subgenre clear from title + description together
  medium — inferring from partial signal or the book straddles categories
  low    — guessing; consider returning empty lists instead

EXAMPLES (calibrate against these):

  "Dune" by Frank Herbert — desert planet, galactic empire, prophecy
  → subgenre=['Space Opera'], theme=[], confidence='high'

  "Starship Troopers" by Robert Heinlein — mobile infantry vs Bugs
  → subgenre=['Military SF'], theme=[], confidence='high'

  "Neuromancer" by William Gibson — hacker hired to run a heist in cyberspace
  → subgenre=['Cyberpunk'], theme=[], confidence='high'

  "The Martian" by Andy Weir — stranded astronaut survives on Mars with science
  → subgenre=['Hard SF'], theme=[], confidence='high'

  "1984" by George Orwell — totalitarian surveillance state, Winston Smith
  → subgenre=['Dystopia'], theme=[], confidence='high'

  "The Road" by Cormac McCarthy — father and son walk through ash-grey America
  → subgenre=['Post-Apocalyptic'], theme=[], confidence='high'

  "Slaughterhouse-Five" by Kurt Vonnegut — Billy Pilgrim unstuck in time
  → subgenre=['Time Travel'], theme=[], confidence='high'

  "The Man in the High Castle" by Philip K. Dick — Axis won WWII
  → subgenre=['Alternate History'], theme=[], confidence='high'

  "Contact" by Carl Sagan — astronomer receives alien signal
  → subgenre=['First Contact'], theme=[], confidence='high'

  "The Difference Engine" by Gibson & Sterling — Babbage's analytical engine
  in Victorian London
  → subgenre=['Steampunk'], theme=[], confidence='high'

  "I, Robot" by Isaac Asimov — stories about the Three Laws of Robotics
  → subgenre=[], theme=['AI & Robots'], confidence='high'

  "Honor Harrington" series by David Weber — starship captain in galactic war
  → subgenre=['Space Opera', 'Military SF'], theme=[], confidence='high'

  "Ender's Game" by Orson Scott Card — child prodigy trained to fight aliens
  → subgenre=['Military SF', 'First Contact'], theme=[], confidence='high'

  "Ancillary Justice" by Ann Leckie — former warship AI seeks revenge
  → subgenre=['Space Opera'], theme=['AI & Robots'], confidence='high'

  "Blindsight" by Peter Watts — first contact with alien life near Pluto,
  hard-SF neuroscience focus
  → subgenre=['First Contact', 'Hard SF'], theme=[], confidence='high'

  "Do Androids Dream of Electric Sheep?" by Philip K. Dick — bounty hunter
  retires androids
  → subgenre=['Cyberpunk'], theme=['AI & Robots'], confidence='high'

  "A Canticle for Leibowitz" by Walter M. Miller — monks preserve knowledge
  through nuclear dark ages
  → subgenre=['Post-Apocalyptic'], theme=[], confidence='high'

  "11/22/63" by Stephen King — man travels back to prevent JFK assassination
  → subgenre=['Time Travel', 'Alternate History'], theme=[], confidence='high'

  "Outlaw of Gor" by John Norman — adventures on the planet Gor (fictional
  world with feudal/ancient aesthetics)
  → subgenre=[], theme=[], confidence='high'
  (Gor is an invented planetary-romance setting, NOT Alternate History)

  A Star Wars tie-in novel set in the Old Republic era with a diverging
  faction war
  → subgenre=['Space Opera'], theme=[], confidence='high'
  (Star Wars is a fictional universe; internal timeline branches are NOT
   Alternate History)

  "Dune" prequel or sequel novel set in a different era of the Dune universe
  → subgenre=['Space Opera'], theme=[], confidence='high'
  (Dune's setting is far-future fictional, not a divergence from Earth)

  "The Forever War" by Joe Haldeman — soldier fights century-long alien war
  while Earth changes
  → subgenre=['Military SF', 'First Contact'], theme=[], confidence='high'

  "Red Mars" by Kim Stanley Robinson — terraforming Mars, political factions
  → subgenre=['Hard SF'], theme=[], confidence='high'

  "Snow Crash" by Neal Stephenson — pizza delivery man / hacker in near-future
  corporate America
  → subgenre=['Cyberpunk'], theme=[], confidence='high'

  "Station Eleven" by Emily St. John Mandel — a travelling Shakespeare company
  after a pandemic collapses civilisation
  → subgenre=['Post-Apocalyptic'], theme=[], confidence='high'

  "Foundation" by Isaac Asimov — mathematician predicts fall of galactic empire
  → subgenre=['Space Opera'], theme=[], confidence='high'
  (Hard SF consideration: the "science" of psychohistory is more literary device
  than genuine hard-SF rigour — omit Hard SF unless description emphasises
  technical accuracy)

  A generic Star Wars tie-in with space battles and the Force
  → subgenre=['Space Opera'], theme=[], confidence='high'

  A generic Star Trek tie-in without a specific distinguishing subgenre
  → subgenre=[], theme=[], confidence='medium'
  (Star Trek is already identified by its series tag; leave subgenre empty
  unless the description clearly signals a strong subgenre)

  "Leviathan Wakes" by James S.A. Corey — solar-system-scale noir thriller
  → subgenre=['Space Opera'], theme=[], confidence='medium'
  (Space Opera by scope, but not galactic; Hard SF could apply if description
  emphasises orbital mechanics — medium confidence for the call)

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
) -> SciFiTags | None:
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
            output_format=SciFiTags,
        )
    except anthropic.APIError:
        log.exception("book %d: API error", book.book_id)
        return None
    except Exception:
        log.exception("book %d: parse/validation error", book.book_id)
        return None

    return response.parsed_output
