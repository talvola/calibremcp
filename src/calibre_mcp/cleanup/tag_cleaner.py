"""Tag-level cleanup: delete junk tags, merge near-duplicates.

Unlike the per-book proposals system, these operations work at the library
level — a tag rename is one SQL UPDATE on the tags table and instantly affects
every book that carries it.  Calibre-Web reads SQLite on each request so
changes are visible immediately; Calibre desktop needs a library refresh.

Three operation types
---------------------
delete  — strip tag from all books and drop from tags table
merge   — redirect all books from old_tag to existing canonical, drop old_tag
rename  — change tag name in place (canonical doesn't exist yet)

Passes
------
Pass 1 — mechanical deletes: URL tags, garbage keyword strings, LCSH catalog
         tags (geographic qualifiers, war/conflict entries, TV program refs).
         No human judgment needed; pattern-matched with high confidence.

Pass 2 — canonical merges: known near-duplicate pairs (Non-Fiction→Nonfiction,
         SciFi→Science Fiction, etc.).  Small hardcoded list; human-reviewed.

Pass 3 — compound tag triage: semicolon / ampersand compound tags.
         Handled separately once the list is clean from passes 1+2.

Pass 4 — ampersand tag triage: merge near-duplicate genre compounds into
         canonical tags; delete LCSH sub-category and gaming catch-all tags.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

OpKind = Literal["delete", "merge", "rename"]


@dataclass
class TagOp:
    op: OpKind
    source_tag: str
    target_tag: str | None  # None for delete
    reason: str
    books_affected: int
    pass_num: int = 1


@dataclass
class ScanResult:
    total_tags: int
    singleton_count: int
    ops: list[TagOp] = field(default_factory=list)

    @property
    def pass1_ops(self) -> list[TagOp]:
        return [o for o in self.ops if o.pass_num == 1]

    @property
    def pass2_ops(self) -> list[TagOp]:
        return [o for o in self.ops if o.pass_num == 2]

    @property
    def deletes(self) -> list[TagOp]:
        return [o for o in self.ops if o.op == "delete"]

    @property
    def merges(self) -> list[TagOp]:
        return [o for o in self.ops if o.op in ("merge", "rename")]


# ---------------------------------------------------------------------------
# Pattern matchers for Pass 1
# ---------------------------------------------------------------------------

_URL_RE        = re.compile(r'^https?://')
_GEO_ABBR_RE   = re.compile(r'\([A-Z][a-z]+\.\)')          # (N.Y.), (Mass.)
_TV_PROG_RE    = re.compile(r'\(Television Program\)', re.I)
_YEAR_RANGE_RE = re.compile(r'\b\d{4}-\d{4}\b')
_WAR_KEYWORDS  = re.compile(
    r'\b(War|Conflict|Revolution|Operation|Uprising|Occupation|Rebellion)\b'
)

# Multi-semicolon = definitely LCSH compound subject heading
_MULTI_SEMI_RE = re.compile(r'.+;.+;')

_MAX_TAG_LEN = 60  # tags longer than this are keyword-stuffed garbage

# ── Additional Pass 1 patterns ───────────────────────────────────────────────

# Trailing punctuation = LCSH terminal period or stray whitespace/period
# e.g. "Cookery; Chinese.", "Coooking .", "United States."
# Exclude abbreviation-style strings where periods are throughout: "U.S.A.", "U.K."
_TRAILING_PERIOD_RE = re.compile(r'[\s\.]+$')
_ABBREV_RE = re.compile(r'^([A-Z]\.){2,}[A-Z]?\.$')  # U.S.A., U.K., etc.

# LCSH single-semicolon with nationality/geographic qualifier after the semicolon
# e.g. "Cookery; American", "Cooking; Chinese", "Detective and Mystery Stories; American"
# The second part is always a short word/phrase — we match any "X; Y" where X and Y
# are both short (≤ 35 chars each) so we don't catch author-style "Niven; Larry - ..."
_LCSH_NATIONALITY_RE = re.compile(r'^[^;]{3,35};\s*[^;]{2,35}$')
# Nationality/place adjectives that confirm the second part is an LCSH qualifier
_NATIONALITY_WORDS = re.compile(
    r'\b(American|English|British|Australian|Canadian|Irish|Scottish|Welsh|'
    r'French|German|Italian|Spanish|Russian|Chinese|Japanese|Korean|Indian|'
    r'Mexican|Jewish|African|European|Asian|Latin|Nordic|Scandinavian|'
    r'International|General)\b',
    re.IGNORECASE,
)

# "United States" bare or with LCSH dash/qualifier — all are subject headings
_US_LCSH_RE = re.compile(r'^United States($|[\s\.\-])', re.IGNORECASE)

# Import artifacts: word immediately followed by 3+ digits, no space
# e.g. "cookie429", "C429", "Azizex666"
# Requires 3+ digits to avoid matching real tech tags like "css3", "html5", "mp3"
_IMPORT_ARTIFACT_RE = re.compile(r'^[A-Za-z]+\d{3,}$')

# Garbled multi-value tags where commas became "m " — e.g. "Cookbooksm Cookingm Foodm"
_GARBLED_M_RE = re.compile(r'\w+m\s+\w+m\s+')

# LCSH author-as-tag: "Niven; Larry - Prose & Criticism", "King; Stephen - Prose & Criticism"
_AUTHOR_LCSH_RE = re.compile(r';\s*\w+.*-\s*(Prose\s*&\s*)?Criticism', re.IGNORECASE)

# LCSH fictitious-character tags (truncated in Calibre): "Wiggin; Ender (Fictitious char"
_FICTITIOUS_RE = re.compile(r'\(Fictitious', re.IGNORECASE)

# Geographic with internal semicolon: "Manhattan (New York; N.Y.)", "Wall Street (New York; N.Y.)"
_GEO_SEMI_RE = re.compile(r'\([^)]*;[^)]*\)')

# ── Additional Pass 1 patterns (Phase A) ────────────────────────────────────

# Acclaimed.* internal review-list tags  e.g. "Acclaimed.Dark Thoughts"
_ACCLAIMED_RE = re.compile(r'^Acclaimed\.')

# Garbled single-lowercase-letter + genre word (import artifact)
# e.g. "n Fiction", "r Mystery", "s Science Fiction"
_GARBLED_LETTER_GENRE_RE = re.compile(
    r'^[a-z]\s+(Fiction|Mystery|Horror|Fantasy|Thriller|Science\s+Fiction)$'
)

# LCSH lifespan/date lines: "1934-"  "1948-"  "1485?-1540"
_LCSH_LIFESPAN_RE = re.compile(r'^\d{4}\??[-–]\d{0,4}$')

# BISAC underscore slugs: FICTION_FANTASY_GENERAL
_BISAC_SLUG_RE = re.compile(r'^[A-Z]{2,}(_[A-Z]{2,}){1,}$')

# Fiction.Genre / Fiction:Genre import format artifacts  e.g. "Fiction.Heroic Fantasy"
_FICTION_DOT_COLON_RE = re.compile(r'^Fiction[.:][A-Za-z]')


# ---------------------------------------------------------------------------
# Pass 2: hardcoded canonical merge pairs
# ---------------------------------------------------------------------------

# (source_tag, canonical_tag) — source will be merged/renamed into canonical.
# Only pairs where the merge is unambiguous and the canonical clearly dominates.
CANONICAL_MERGES: list[tuple[str, str]] = [
    ("Non-Fiction",            "Nonfiction"),
    ("Non Fiction",            "Nonfiction"),
    ("SciFi",                  "Science Fiction"),
    ("Horror & Ghost Stories", "Horror"),
    ("Thrillers & Suspense",   "Thriller"),
    # Leave 'history' (441) and 'Historical' (299) alone — distinct meanings.
    # Leave 'Star Trek' (64) — may carry non-fiction books, decide later.
]


# ---------------------------------------------------------------------------
# Pass 3: compound & tag cleanup
# ---------------------------------------------------------------------------

# Merge these into the canonical single-concept tag.
# Books that had the compound almost certainly have both halves individually.
COMPOUND_MERGES: list[tuple[str, str]] = [
    ("Sci-Fi & Fantasy",           "Science Fiction"),
    ("Collections & Anthologies",  "Anthologies"),
    ("Anthologies & Short Stories", "Anthologies"),
    ("Crime & mystery",            "Mystery"),
    ("Crime & Thriller",           "Thriller"),
    ("Crime & Mystery Fiction",    "Mystery"),
]

# Delete outright — redundant Amazon-category compounds whose component tags
# already exist on the same books, or too-broad to be useful.
COMPOUND_DELETES: list[str] = [
    "Science Fiction & Fantasy",
    "Literature & Fiction",
    "Fantasy & Magic",
    "General & Literary Fiction",
    "Modern & contemporary fiction (post c 1945)",
    "American Mystery & Suspense Fiction",
    "Mystery; Thriller & Suspense",       # last remaining big semicolon tag
    "Science Fiction; Fantasy & Scary Stories",
    "Body; Mind & Spirit",
    "Novels; other prose & writers",
    "Words; Language & Grammar",
    "Writing; Research & Publishing Guides",
    "Aeronautics; Astronautics & Space Science",
    "Voyages; Imaginary",
    # Small-count single-semicolon leftovers not caught by patterns
    "Aeronautics; Commercial",
    "Airplanes; Military",
    "Naval History; Modern",
    "Beauty; Personal",
]


# ---------------------------------------------------------------------------
# Pass 4: ampersand tag triage
# ---------------------------------------------------------------------------

# Near-duplicate genre compounds → canonical single-concept tags.
PASS4_MERGES: list[tuple[str, str]] = [
    # Science Fiction
    ("Sci Fi & Fantasy",                      "Science Fiction"),
    ("Teens - Science Fiction & Fantasy",     "Science Fiction"),
    # Audience
    ("Teen & Young Adult",                    "Young Adult"),
    # Thriller
    ("Thriller & Suspense",                   "Thriller"),
    ("Mysteries & Thrillers",                 "Thriller"),
    # Mystery
    ("Mystery & Suspense",                    "Mystery"),
    ("Mysteries & Detective Stories",         "Mystery"),
    ("International Mystery & Crime",         "Mystery"),
    ("International Crime & Mystery",         "Mystery"),
    ("Mystery & Crime",                       "Mystery"),
    ("English Mystery & Suspense Fiction",    "Mystery"),
    ("detective & mystery",                   "Mystery"),
    # Horror
    ("Horror & Supernatural",                 "Horror"),
    # Short stories / anthologies
    ("Short Stories & Novellas",              "Short Stories"),
    ("Short Stories & Anthologies",           "Short Stories"),
    ("Fiction anthologies & collections",     "Anthologies"),
    ("Anthologies & Literature Collections",  "Anthologies"),
    # Fiction
    ("Fiction & Literature",                  "Fiction"),
    # D&D
    ("D&D",                                   "Dungeons & Dragons"),
    # Cookbook technique
    ("Methods - Quick & Easy",                "Quick & Easy"),
    # Fan fiction
    ("Pride and Prejudice & Related Fandoms", "Pride & Prejudice"),
    # Military
    ("Military & Wars",                       "War & Military"),
    # Espionage
    ("Espionage & spy thriller",              "Espionage"),
]

# Delete: LCSH sub-category noise, gaming catch-alls, and compound tags
# where books already carry all the component genre tags individually.
PASS4_DELETES: list[str] = [
    # Cookbook LCSH and Amazon sub-category junk
    "Courses & Dishes",
    "Courses & Dishes - General",
    "Courses & Dishes - Bread",
    "Food & Drink",
    "Food & Drink: General",
    "Food & Drinks",
    "Food & Wine",
    "Cooking & Food",
    "Cooking; Food & Wine",
    "Drinks & Beverages",
    "Beverages - Wine & Spirits",
    "Agriculture & Food",
    "Individual Chefs & Restaurants",
    "National & Regional Cuisine",
    "General cookery & recipes",
    "Cookery dishes & courses",
    "Cookings/Courses & Dishes Appetizers",
    "Health & Healing - Weight Control",
    "Methods - Canning & Preserving",
    # Technology & Business Amazon sub-headings (keep the parent tags)
    "TECHNOLOGY & ENGINEERING / Robotics",
    "TECHNOLOGY & ENGINEERING / Electronics / General",
    "TECHNOLOGY & ENGINEERING / Aeronautics & Astronautics",
    "Business & Economics / Leadership",
    "Business & Economics / Time Management",
    "Business & Economics / General",
    "BUSINESS & ECONOMICS / Consumer Behavior",
    # Gaming/paranormal catch-alls (books already have constituent tags)
    "Role Playing & Fantasy",
    "Science Fiction & Fantasy Gaming",
    "Occult & Supernatural",
    "Paranormal & Urban",
    # LCSH junk / abbreviations / noise
    "People & Places",
    "British & Irish",
    "Russian & Former Soviet Union",
    "Video & Electronic",
    "nic & nigel",
    "S&S",
    "Twists & Turns",
    "Performing Arts/Film & Video/History & Criticism",
    "Collections & anthologies of various literary forms",
    "Fiction & related items",
    "Arts & Entertainment",
    # LCSH author-name fragments (the full form is caught by pattern; these are remnants)
    "C.J. - Prose & Criticism",
    "Walter M. - Prose & Criticism",
    "Philip K. - Prose & Criticism",
    "Umberto - Prose & Criticism",
    "William - Prose & Criticism",
    # Old D&D abbreviation (1 book; already has Dungeons & Dragons tag)
    "od&d",
    # LCSH geo/nationality sub-variants
    "Indian & South Asian",
    "Caribbean & Latin America",
    "Caribbean & Latin American",
    "Spanish & Portuguese",
    "Regional & Ethnic - Chinese",
    "Regional & Ethnic - American - General",
]


# ---------------------------------------------------------------------------
# Pass 5: food / drink tag consolidation
# ---------------------------------------------------------------------------

PASS5_MERGES: list[tuple[str, str]] = [
    # Food umbrella — broad BISAC/Amazon duplicates merge into Food
    ("Food and Drink",     "Food"),
    ("Gastronomy",         "Food"),
    ("Gourmet",            "Food"),
    ("Culinary",           "Food"),
    # Keto/low-carb family — merge into the most-used combined label
    ("ketogenic diet",        "Keto/Low-Carb"),
    ("Low Carbohydrate",      "Keto/Low-Carb"),
    ("Low-carbohydrate diet", "Keto/Low-Carb"),
    # Mystery sub-genres
    ("food mystery",          "culinary mystery"),
    ("Food mysteries",        "culinary mystery"),
    ("culinary cozy",         "culinary mystery"),
    # Spirits family
    ("Spirit",    "Spirits"),
    ("liquor",    "Spirits"),
    ("Liquors",   "Spirits"),
    ("Distilling","Spirits"),
    # Whiskey — merge Scotch spelling into the larger American-spelling tag,
    # then rename the lowercase winner to title case
    ("Whisky",   "whiskey"),   # merge first (whiskey exists, Whiskey doesn't yet)
    ("whiskey",  "Whiskey"),   # rename in-place to title case
    # Beer
    ("craft beer", "Beer"),
]

PASS5_DELETES: list[str] = [
    # Keto/low-carb keyword-stuffed singletons
    "keto adapted",
    "keto clarity",
    "keto cookbook",
    "keto diet for beginners",
    "keto diet recipes",
    "ketogenic cookbook",
    "ketogenic diet recipes",
    "ketogenic recipes",
    "low carb diet",
    "low carb diet for beginners",
    "low carb essential",
    "low carb high fat",
    "low carb weight loss",
    "low carbohydrate diet",
    "atkins diet",
    "atkins diet for beginners",
    "paleo approach",
    "paleo diet for beginners",
    # Japanese keyword-stuffed singletons (all 1-book title/keyword tags)
    "Japanese Cocktails Cookbook",
    "Japanese Cocktails Cooking",
    "Japanese Cocktails Dish",
    "Japanese Cocktails Food",
    "Japanese Cocktails Meal",
    "Japanese Cocktails Recipe Book",
    "Japanese Cocktails Recipes",
    "Japanese appetizers",
    "Japanese comfort food",
    "Japanese cooking book",
    "Japanese food culture",
    "Japanese Ramen Cookbook",
    "Japanese rice bowls",
    "Japanese small plates",
    "Japanese street food",
    "izakaya recipes",
    "Ramen",
    "ramen cookbook",
    "ramen recipes",
    "Sushi",
    # Food junk
    "Food and Drin",           # truncated typo
    "food picks thanksgiving",
    "Food*Cooking*Ingredient*Meat",  # garbled
    "Food Industry",
    "Food additives",
    "Food Preservation",
    "Drinking cups",
    "Drinking customs",
    "drinking wine",
    "Cookery (Pasta)",
    "Cooking (Pasta)",
    "Cookery by ingredient",
    "Cooking on a Bootstrap",
    "Cooking with specific gadgets",
    "Cooks",
    "RecipeTin Eats",
    "Cookbooks/Italian",       # slash-format artifact
    "Baking - Confectionery",
    "recipes",                 # redundant on cookbook books
    # Dietary singletons
    "vegan cookbook",
    "vegetarian cookbook",
    "Non-Vegetarion",          # typo
    "time tested vegetarian recipes",
    "vegetarian Indian cooking",
    # Beer/spirits noise
    "Beer Guide",
    "brewery",
    "Brewing Industry",
    "Distillers",
    "wine 101",
    "wine a day",
    "wine alcohol",
    "wine fiction",
    "wine lover",
    "wine novel",
    "Wine novels",
    "Wine Trail Series",
    "Winer",                   # typo
    "winery mystery",
    "Wine Guide",
    "Cocktail servers",
    "Cocktail servers - Louisiana",
]


# ---------------------------------------------------------------------------
# Pass 6: genre near-duplicate merges + hardcoded junk deletes
# ---------------------------------------------------------------------------

PASS6_MERGES: list[tuple[str, str]] = [
    # Mystery near-duplicates → Mystery
    ("murder mystery",                     "Mystery"),
    ("murder mystery series",              "Mystery"),
    ("Mystery and detective stories",      "Mystery"),
    ("Mystery And Suspense Fiction",       "Mystery"),
    ("detective fiction",                  "Mystery"),
    ("Detective and Mystery Fiction",      "Mystery"),
    ("mystery and detective",              "Mystery"),
    ("magic & mystery",                    "Mystery"),
    ("womens mystery",                     "Mystery"),
    ("women mystery",                      "Mystery"),
    # Cozy Mystery variants
    ("Mystery Cozy",                       "Cozy Mystery"),
    ("cosy mystery",                       "Cozy Mystery"),
    ("witch mystery",                      "Cozy Mystery"),
    ("witch cozy mystery",                 "Cozy Mystery"),
    ("flower shop mystery",                "Cozy Mystery"),
    ("Paranormal Cozy Mystery",            "Cozy Mystery"),
    ("christmas mystery",                  "Cozy Mystery"),
    ("dog mystery",                        "Cozy Mystery"),
    # Thriller variants → Thriller
    ("Crime Thriller",                     "Thriller"),
    ("Fiction.Thriller/Suspense",          "Thriller"),
    ("Fiction:Thriller",                   "Thriller"),
    ("Horror Thriller",                    "Thriller"),
    ("Domestic Thriller",                  "Thriller"),
    ("Supernatural Thriller",              "Thriller"),
    ("Crime Mystery Thriller",             "Thriller"),
    ("Mysteries & Thrillers",              "Thriller"),
    # Science Fiction typos/variants → Science Fiction
    ("Scicnce Fiction",                    "Science Fiction"),
    ("Sscience Fiction",                   "Science Fiction"),
    ("lcsh: Science Fiction",              "Science Fiction"),
    ("sci fi and fantasy",                 "Science Fiction"),
    ("Science fiction stories",            "Science Fiction"),
    ("sf-fantasy",                         "Science Fiction"),
    ("Star Wars fiction",                  "Science Fiction"),
    # Horror variants → Horror
    ("Horror stories",                     "Horror"),
    ("body horror",                        "Horror"),
    ("british horror",                     "Horror"),
    ("Southern horror",                    "Horror"),
    ("feminist horror",                    "Horror"),
    ("historical horror",                  "Horror"),
    ("ghost stories",                      "Horror"),
    # Fantasy variants → Fantasy
    ("heroic fantasy",                     "Fantasy"),
    ("regional fantasy",                   "Fantasy"),
    ("rural fantasy",                      "Fantasy"),
    ("contemporary fantasy",               "Fantasy"),
    ("fantasy adventure",                  "Fantasy"),
    # Young Adult variants → Young Adult
    ("Young Adult Fiction",                "Young Adult"),
    ("Fiction-Young Adult",                "Young Adult"),
    ("Children's Books - Young Adult Fiction", "Young Adult"),
    ("teen fiction",                       "Young Adult"),
    ("Fantasy (Young Adult)",              "Young Adult"),
    # Short Stories / Flash
    ("flash fiction",                      "Short Stories"),
    ("Short Fiction",                      "Short Stories"),
    ("Science fiction short stories",      "Short Stories"),
    ("Fantasy short stories",              "Short Stories"),
    # Women's Fiction
    ("womens fiction",                     "Women's Fiction"),
    # Romance
    ("romance adult fiction",              "Romance"),
    ("Romantic suspense fiction",          "Romance"),
    ("romance and mystery",                "Romance"),
]

PASS6_DELETES: list[str] = [
    # ── Typos / garbled ─────────────────────────────────────────────────────
    "Advanture", "Aultery", "animalled", "apocalpyse",
    "Amer", "Hiistorical Fiction", "Literacy Fiction",
    "nepalifiction",
    # ── Amazon / publisher catalog terms ────────────────────────────────────
    "Adult Trade", "Adult Literary", "Adult Literature",
    "Audio Adult: Other", "Bargain Books",
    "Popular American Fiction", "Spanish: Adult Fiction",
    "Biography True Crime Business Science",
    # ── LCSH nationality + genre qualifiers (base genre already in library) ─
    "American Horror Fiction", "English Horror Fiction",
    "American Light Romantic Fiction",
    "American fiction - 21st century",
    "American Short Story Collections",
    "American Novel And Short Story",
    "Fiction - Men's Adventure",
    "Fiction : Fantasy - Historical",
    "Fiction : Science Fiction - Series",
    "Fiction:Detective",
    "Fiction:Historical",
    "Settings for Fiction",
    "Classic fiction",
    # ── Overly specific genre+qualifier combos ───────────────────────────────
    "rv fiction", "travel trailer fiction",
    "Science Fiction. Military Fiction",
    "Triangles (Interpersonal relations) Fiction",
    "New York (State) Fiction",
    "Genetic engineering Fiction",
    "Swedish (Language) Contemporary Fiction",
    "Climate Change Fiction",
    "Space warfare Fiction",
    "Native American Fiction",
    "Male friendship Fiction",
    "Feminist Science Fiction",
    "Humor Science Fiction",
    "Inspirational Fiction",
    "Diary Fiction",
    "Satirical fiction",
    # ── Writing craft singletons (redundant with genre + Writing tags) ────────
    "writing fantasy", "writing fiction", "writing science fiction",
    "writing instruction", "Writing Exercises and Techniques",
    "creative writing workbook", "write a novel",
    # ── How-to sentence-length noise ─────────────────────────────────────────
    "how to be a better parent", "how to lose weight", "how to raise children",
    # ── Publisher / series / internal tags ──────────────────────────────────
    "bards and sages", "apex book company",
    "Anneli Lax New Mathematical Library",
    "Adventure of the Field Theorems",
    # ── Art / slash-format artifacts (not caught by pass 4) ──────────────────
    "Art/Film & Video", "Art/Surrealism/Autobiography",
    "Performing Arts / Film & Video - Screenwriting",
    # ── Misc noise ───────────────────────────────────────────────────────────
    "ALL", "5.0", "base", "acid", "abroad", "abstraction",
    "aging", "anger", "api", ".Crime", "ArgonAlt",
    "Mystery. Agatha Christie",
]


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan(cal: sqlite3.Connection) -> ScanResult:
    """Analyse Calibre's tag table and classify tags into operations.

    Returns a ScanResult with Pass-1 delete ops and Pass-2 merge ops.
    Does NOT write anything — call save_ops() to persist the plan.
    """
    rows = cal.execute("""
        SELECT t.name, COUNT(btl.book) AS cnt
        FROM tags t
        LEFT JOIN books_tags_link btl ON btl.tag = t.id
        GROUP BY t.id
        ORDER BY t.name COLLATE NOCASE
    """).fetchall()

    tag_counts: dict[str, int] = {r["name"]: r["cnt"] for r in rows}
    ops: list[TagOp] = []

    # ── Pass 1: mechanical deletes ──────────────────────────────────────────
    for name, cnt in tag_counts.items():

        if _URL_RE.match(name):
            ops.append(TagOp("delete", name, None, "URL tag", cnt, pass_num=1))
            continue

        if len(name) > _MAX_TAG_LEN:
            ops.append(TagOp("delete", name, None,
                             f"keyword-stuffed string ({len(name)} chars)", cnt, pass_num=1))
            continue

        if _TV_PROG_RE.search(name):
            ops.append(TagOp("delete", name, None, "LCSH TV-program tag", cnt, pass_num=1))
            continue

        if _GEO_ABBR_RE.search(name):
            ops.append(TagOp("delete", name, None, "LCSH geographic qualifier", cnt, pass_num=1))
            continue

        # LCSH war/event tags: require semicolon PLUS war keyword or year range
        if ";" in name and (_WAR_KEYWORDS.search(name) or _YEAR_RANGE_RE.search(name)):
            ops.append(TagOp("delete", name, None, "LCSH historical-event tag", cnt, pass_num=1))
            continue

        # Multi-semicolon compound subjects (3+ parts) — always LCSH noise
        if _MULTI_SEMI_RE.match(name):
            ops.append(TagOp("delete", name, None, "LCSH multi-part subject", cnt, pass_num=1))
            continue

        # "United States - *" / "United States.*" LCSH political/geographic
        if _US_LCSH_RE.match(name):
            ops.append(TagOp("delete", name, None, "LCSH US geographic/political", cnt, pass_num=1))
            continue

        # Trailing-period LCSH tags  e.g. "Cookery; Chinese.", "Coooking ."
        # Skip abbreviations like "U.S.A." where periods are structural
        stripped = _TRAILING_PERIOD_RE.sub("", name)
        if stripped != name and stripped and not _ABBREV_RE.match(name):
            ops.append(TagOp("delete", name, None, "LCSH trailing-period tag", cnt, pass_num=1))
            continue

        # Garbled comma-separated strings  e.g. "Cookbooksm Cookingm Foodm Nonfiction"
        if _GARBLED_M_RE.search(name):
            ops.append(TagOp("delete", name, None, "garbled comma-separated tag", cnt, pass_num=1))
            continue

        # Import artifact  e.g. "cookie429"
        if _IMPORT_ARTIFACT_RE.match(name):
            ops.append(TagOp("delete", name, None, "import artifact (word+digits)", cnt, pass_num=1))
            continue

        # LCSH single-semicolon with nationality qualifier  e.g. "Cookery; American"
        if (
            ";" in name
            and _LCSH_NATIONALITY_RE.match(name)
            and _NATIONALITY_WORDS.search(name.split(";", 1)[1])
        ):
            ops.append(TagOp("delete", name, None, "LCSH nationality qualifier", cnt, pass_num=1))
            continue

        # LCSH author-as-tag  e.g. "King; Stephen - Prose & Criticism"
        if _AUTHOR_LCSH_RE.search(name):
            ops.append(TagOp("delete", name, None, "LCSH author tag", cnt, pass_num=1))
            continue

        # LCSH fictitious-character tags  e.g. "Wiggin; Ender (Fictitious char"
        if _FICTITIOUS_RE.search(name):
            ops.append(TagOp("delete", name, None, "LCSH fictitious-character tag", cnt, pass_num=1))
            continue

        # Geographic with internal semicolon  e.g. "Manhattan (New York; N.Y.)"
        if ";" in name and _GEO_SEMI_RE.search(name):
            ops.append(TagOp("delete", name, None, "LCSH geographic (internal semicolon)", cnt, pass_num=1))
            continue

        # Acclaimed.* internal review-list tags
        if _ACCLAIMED_RE.match(name):
            ops.append(TagOp("delete", name, None, "Acclaimed.* review-list tag", cnt, pass_num=1))
            continue

        # Garbled single-letter prefix: "n Fiction", "r Mystery"
        if _GARBLED_LETTER_GENRE_RE.match(name):
            ops.append(TagOp("delete", name, None, "garbled letter-prefix tag", cnt, pass_num=1))
            continue

        # LCSH lifespan: "1934-" / "1485?-1540"
        if _LCSH_LIFESPAN_RE.match(name):
            ops.append(TagOp("delete", name, None, "LCSH lifespan/date tag", cnt, pass_num=1))
            continue

        # BISAC underscore slug: FICTION_FANTASY_GENERAL
        if _BISAC_SLUG_RE.match(name):
            ops.append(TagOp("delete", name, None, "BISAC underscore slug", cnt, pass_num=1))
            continue

        # Fiction.Genre / Fiction:Genre format artifacts
        if _FICTION_DOT_COLON_RE.match(name):
            ops.append(TagOp("delete", name, None, "Fiction.Genre format artifact", cnt, pass_num=1))
            continue

    # ── Pass 2: canonical merges ────────────────────────────────────────────
    queued = {o.source_tag for o in ops}

    for src, dst in CANONICAL_MERGES:
        if src not in tag_counts or src in queued:
            continue
        cnt = tag_counts[src]
        if dst in tag_counts:
            ops.append(TagOp("merge", src, dst,
                             f"canonical duplicate of {dst!r}", cnt, pass_num=2))
        else:
            ops.append(TagOp("rename", src, dst,
                             f"rename to canonical form {dst!r}", cnt, pass_num=2))
        queued.add(src)

    # ── Pass 3: compound & tag cleanup ─────────────────────────────────────
    # Redundant Amazon-style category compounds — books already carry the
    # individual genre tags (Science Fiction, Fantasy, Mystery, etc.)
    for src, dst in COMPOUND_MERGES:
        if src not in tag_counts or src in queued:
            continue
        cnt = tag_counts[src]
        if dst in tag_counts:
            ops.append(TagOp("merge", src, dst,
                             f"compound → canonical {dst!r}", cnt, pass_num=3))
        else:
            ops.append(TagOp("rename", src, dst,
                             f"rename compound to {dst!r}", cnt, pass_num=3))
        queued.add(src)

    for src in COMPOUND_DELETES:
        if src not in tag_counts or src in queued:
            continue
        ops.append(TagOp("delete", src, None,
                         "redundant Amazon-style compound tag", tag_counts[src], pass_num=3))
        queued.add(src)

    # ── Pass 4: ampersand tag triage ───────────────────────────────────────
    for src, dst in PASS4_MERGES:
        if src not in tag_counts or src in queued:
            continue
        cnt = tag_counts[src]
        if dst in tag_counts:
            ops.append(TagOp("merge", src, dst,
                             f"compound → canonical {dst!r}", cnt, pass_num=4))
        else:
            ops.append(TagOp("rename", src, dst,
                             f"rename compound to {dst!r}", cnt, pass_num=4))
        queued.add(src)

    for src in PASS4_DELETES:
        if src not in tag_counts or src in queued:
            continue
        ops.append(TagOp("delete", src, None,
                         "LCSH/compound tag noise", tag_counts[src], pass_num=4))
        queued.add(src)

    # ── Pass 5: food/drink tag consolidation ──────────────────────────────
    for src, dst in PASS5_MERGES:
        if src not in tag_counts or src in queued:
            continue
        cnt = tag_counts[src]
        if dst in tag_counts:
            ops.append(TagOp("merge", src, dst,
                             f"food consolidation → {dst!r}", cnt, pass_num=5))
        else:
            ops.append(TagOp("rename", src, dst,
                             f"food consolidation rename → {dst!r}", cnt, pass_num=5))
        queued.add(src)

    for src in PASS5_DELETES:
        if src not in tag_counts or src in queued:
            continue
        ops.append(TagOp("delete", src, None,
                         "food tag noise/keyword-stuffing", tag_counts[src], pass_num=5))
        queued.add(src)

    # ── Pass 6: genre near-duplicates + hardcoded junk ─────────────────────
    for src, dst in PASS6_MERGES:
        if src not in tag_counts or src in queued:
            continue
        cnt = tag_counts[src]
        if dst in tag_counts:
            ops.append(TagOp("merge", src, dst,
                             f"genre near-duplicate → {dst!r}", cnt, pass_num=6))
        else:
            ops.append(TagOp("rename", src, dst,
                             f"genre rename → {dst!r}", cnt, pass_num=6))
        queued.add(src)

    for src in PASS6_DELETES:
        if src not in tag_counts or src in queued:
            continue
        ops.append(TagOp("delete", src, None,
                         "junk/noise tag", tag_counts[src], pass_num=6))
        queued.add(src)

    return ScanResult(
        total_tags=len(tag_counts),
        singleton_count=sum(1 for c in tag_counts.values() if c == 1),
        ops=ops,
    )


# ---------------------------------------------------------------------------
# Persistence: save / load ops in proposals.db
# ---------------------------------------------------------------------------

def save_ops(ops_conn: sqlite3.Connection, ops: list[TagOp]) -> int:
    """Upsert tag ops into proposals.db.  Returns count of newly inserted rows.

    Uses INSERT OR IGNORE so running scan twice is idempotent — existing
    pending ops are preserved unchanged.
    """
    inserted = 0
    for op in ops:
        cur = ops_conn.execute("""
            INSERT OR IGNORE INTO tag_ops
                (op, source_tag, target_tag, reason, books_affected, pass)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (op.op, op.source_tag, op.target_tag, op.reason,
              op.books_affected, op.pass_num))
        inserted += cur.rowcount
    ops_conn.execute("COMMIT")
    ops_conn.execute("BEGIN")
    return inserted


def load_ops(
    ops_conn: sqlite3.Connection,
    *,
    status: str = "pending",
    pass_num: int | None = None,
) -> list[TagOp]:
    """Load ops from proposals.db, optionally filtered by pass."""
    query = "SELECT * FROM tag_ops WHERE status = ?"
    params: list = [status]
    if pass_num is not None:
        query += " AND pass = ?"
        params.append(pass_num)
    query += " ORDER BY pass, op, source_tag COLLATE NOCASE"
    rows = ops_conn.execute(query, params).fetchall()
    return [
        TagOp(
            op=r["op"],
            source_tag=r["source_tag"],
            target_tag=r["target_tag"],
            reason=r["reason"],
            books_affected=r["books_affected"],
            pass_num=r["pass"],
        )
        for r in rows
    ]


def skip_op(ops_conn: sqlite3.Connection, source_tag: str, notes: str = "") -> None:
    """Mark a pending op as skipped (won't be applied)."""
    ops_conn.execute("""
        UPDATE tag_ops SET status = 'skipped', notes = ?
        WHERE source_tag = ? AND status = 'pending'
    """, (notes, source_tag))
    ops_conn.execute("COMMIT")
    ops_conn.execute("BEGIN")


# ---------------------------------------------------------------------------
# Apply: execute ops against Calibre's DB
# ---------------------------------------------------------------------------

def open_calibre_rw(metadata_db: Path) -> sqlite3.Connection:
    """Open Calibre's metadata.db in read-write mode for tag operations."""
    conn = sqlite3.connect(metadata_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@dataclass
class ApplyResult:
    applied: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def apply_ops(
    cal: sqlite3.Connection,
    ops_conn: sqlite3.Connection,
    ops: list[TagOp],
    *,
    dry_run: bool = True,
) -> ApplyResult:
    """Execute a list of TagOps against Calibre's DB.

    When ``dry_run=True`` (default), validates the ops and reports what
    would happen without modifying anything.
    """
    result = ApplyResult()

    for op in ops:
        try:
            src_id = _tag_id(cal, op.source_tag)
            if src_id is None:
                # Tag no longer exists — already cleaned up or never existed
                result.skipped += 1
                if not dry_run:
                    _mark_applied(ops_conn, op.source_tag, notes="tag not found — skipped")
                continue

            if op.op == "delete":
                if not dry_run:
                    cal.execute("DELETE FROM books_tags_link WHERE tag = ?", (src_id,))
                    cal.execute("DELETE FROM tags WHERE id = ?", (src_id,))
                    _mark_applied(ops_conn, op.source_tag)

            elif op.op == "merge":
                dst_id = _tag_id(cal, op.target_tag)
                if dst_id is None:
                    result.errors.append(
                        f"merge {op.source_tag!r}→{op.target_tag!r}: destination not found"
                    )
                    continue
                if not dry_run:
                    # Move books that only have the source tag over to the destination.
                    # OR IGNORE handles books that already carry both.
                    cal.execute("""
                        UPDATE OR IGNORE books_tags_link
                        SET tag = ?
                        WHERE tag = ?
                    """, (dst_id, src_id))
                    # Remove any remaining links (books that had both — the IGNORE above
                    # left the duplicate source link in place).
                    cal.execute("DELETE FROM books_tags_link WHERE tag = ?", (src_id,))
                    cal.execute("DELETE FROM tags WHERE id = ?", (src_id,))
                    _mark_applied(ops_conn, op.source_tag)

            elif op.op == "rename":
                dst_id = _tag_id(cal, op.target_tag)
                if dst_id is not None and dst_id != src_id:
                    result.errors.append(
                        f"rename {op.source_tag!r}→{op.target_tag!r}: "
                        f"target already exists — use merge instead"
                    )
                    continue
                if not dry_run:
                    cal.execute("UPDATE tags SET name = ? WHERE id = ?",
                                (op.target_tag, src_id))
                    _mark_applied(ops_conn, op.source_tag)

            result.applied += 1

        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{op.source_tag!r}: {exc}")

    if not dry_run:
        cal.execute("COMMIT")
        cal.execute("BEGIN")
        ops_conn.execute("COMMIT")
        ops_conn.execute("BEGIN")

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tag_id(cal: sqlite3.Connection, name: str | None) -> int | None:
    if not name:
        return None
    row = cal.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
    return int(row["id"]) if row else None


def _mark_applied(ops_conn: sqlite3.Connection, source_tag: str, notes: str = "") -> None:
    ops_conn.execute("""
        UPDATE tag_ops
        SET status = 'applied', applied_at = ?, notes = ?
        WHERE source_tag = ?
    """, (datetime.now(UTC).isoformat(), notes, source_tag))
