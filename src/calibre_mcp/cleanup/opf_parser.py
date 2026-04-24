"""Extract and normalize metadata from an EPUB's OPF manifest.

Pure parsing — no database, no network, no logging. Handles the three
archetypes observed in Erik's library:

1. Commercial publisher EPUBs — rich OPFs (ISBN, publisher, description, pubdate).
2. Free/pulp EPUBs from ManyBooks, Project Gutenberg, etc. — minimal OPFs.
3. Sigil/converter-authored EPUBs — partial OPFs, may lack namespaces.

Field catalogue (also used by the propose-queue ``field`` column):

    title, authors, isbn, identifier.<scheme>, publisher, pubdate,
    description, language, subjects, series, series_index.
"""

from __future__ import annotations

import html
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

# Containers namespace — META-INF/container.xml only.
_CONTAINER_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}

# Publisher values that indicate provenance/placeholders, not a real publisher.
_SENTINEL_PUBLISHERS: frozenset[str] = frozenset(
    s.casefold()
    for s in (
        "ManyBooks.net",
        "Many Books",
        "Project Gutenberg",
        "Project Gutenberg Literary Archive Foundation",
        "Gutenberg",
        "Unknown",
        "Anonymous",
        "N/A",
        "None",
        "Publisher",
        "Default",
        "",
    )
)

# Match ISO-like date openings: 2014-11-10, 2010-02-16T12:00:00Z, 2009, 2009-10, etc.
_DATE_PREFIX = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?")


@dataclass(frozen=True, slots=True)
class OpfMetadata:
    """Normalized metadata extracted from one EPUB's OPF."""

    title: str | None = None
    authors: tuple[str, ...] = ()
    isbn: str | None = None
    identifiers: dict[str, str] = field(default_factory=dict)
    publisher: str | None = None
    pubdate: str | None = None
    description: str | None = None
    language: str | None = None
    subjects: tuple[str, ...] = ()
    series: str | None = None
    series_index: str | None = None


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def parse_epub(epub_path: Path | str) -> OpfMetadata | None:
    """Locate and parse the OPF inside ``epub_path``. Returns None on failure."""
    path = Path(epub_path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            opf_name = _find_opf_name(zf)
            if opf_name is None:
                return None
            with zf.open(opf_name) as f:
                return parse_opf_bytes(f.read())
    except (zipfile.BadZipFile, OSError):
        return None


def parse_opf_bytes(data: bytes) -> OpfMetadata | None:
    """Parse OPF XML bytes. ``recover=True`` tolerates the many subtly-broken
    OPFs in the wild (stray entities, mismatched tags, declared-but-unused prefixes)."""
    try:
        root = etree.fromstring(data, parser=etree.XMLParser(recover=True))
    except etree.XMLSyntaxError:
        return None
    if root is None:
        return None
    md = _find_metadata_element(root)
    if md is None:
        return None
    return _extract(md)


# ---------------------------------------------------------------------------
# OPF location within the zip
# ---------------------------------------------------------------------------


def _find_opf_name(zf: zipfile.ZipFile) -> str | None:
    # EPUB spec: META-INF/container.xml points to the OPF rootfile.
    try:
        with zf.open("META-INF/container.xml") as f:
            container = etree.fromstring(f.read(), parser=etree.XMLParser(recover=True))
        rootfile = container.find(".//c:rootfile", namespaces=_CONTAINER_NS)
        if rootfile is not None:
            path = rootfile.get("full-path")
            if path:
                return path
    except (KeyError, etree.XMLSyntaxError, OSError):
        pass
    # Fallback: any .opf in the archive.
    for name in zf.namelist():
        if name.lower().endswith(".opf"):
            return name
    return None


def _find_metadata_element(root: etree._Element) -> etree._Element | None:
    # OPFs in the wild vary in namespace handling — some declare the OPF ns
    # as default, some use a prefix, some are unnamespaced. local-name()
    # finds the element regardless.
    results = root.xpath(".//*[local-name()='metadata']")
    if results:
        return results[0]  # type: ignore[no-any-return]
    return None


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------


def _extract(md: etree._Element) -> OpfMetadata:
    title = _first_text(_children(md, "title"))
    authors = tuple(_nonempty(_text(e) for e in _children(md, "creator")))
    publisher = _clean_publisher(_first_text(_children(md, "publisher")))
    pubdate = _parse_date(_first_text(_children(md, "date")))
    description = _clean_description(_first_text(_children(md, "description")))
    language = _clean_language(_first_text(_children(md, "language")))
    subjects = tuple(_dedupe_ci(_nonempty(_text(e) for e in _children(md, "subject"))))
    identifiers = _extract_identifiers(md)
    isbn = _pick_isbn(identifiers)
    series, series_index = _extract_calibre_series(md)
    return OpfMetadata(
        title=_strip_or_none(title),
        authors=authors,
        isbn=isbn,
        identifiers=identifiers,
        publisher=publisher,
        pubdate=pubdate,
        description=description,
        language=language,
        subjects=subjects,
        series=series,
        series_index=series_index,
    )


def _children(md: etree._Element, local_name: str) -> list[etree._Element]:
    # Direct children only — we don't want a stray <title> from a <guide> subtree.
    return md.xpath("./*[local-name()=$n]", n=local_name)  # type: ignore[no-any-return]


def _extract_identifiers(md: etree._Element) -> dict[str, str]:
    """Extract all ``<dc:identifier>`` elements, keyed by scheme (lowercased).

    Schemes we see in the wild: ISBN, UUID, URI, MOBI-ASIN, calibre, BookID,
    bookid, goodreads. We preserve all of them — downstream code picks what
    it cares about.
    """
    out: dict[str, str] = {}
    for el in md.xpath("./*[local-name()='identifier']"):
        val = _text(el)
        if not val:
            continue
        # Scheme can appear in two places:
        #   <dc:identifier opf:scheme="ISBN">...
        #   <dc:identifier id="bookid">9780698...   (scheme inferred from format)
        scheme = el.get("{http://www.idpf.org/2007/opf}scheme") or el.get("scheme") or _infer_scheme(val)
        if not scheme:
            continue
        key = scheme.strip().lower()
        # Prefer the first occurrence; avoid clobbering a valid scheme with a later
        # garbage duplicate.
        out.setdefault(key, val.strip())
    return out


def _infer_scheme(value: str) -> str | None:
    v = value.strip()
    if v.lower().startswith("urn:isbn:"):
        return "isbn"
    if v.lower().startswith("urn:uuid:") or _looks_like_uuid(v):
        return "uuid"
    if _isbn_digits(v) is not None:
        return "isbn"
    return None


def _pick_isbn(identifiers: dict[str, str]) -> str | None:
    """Return a validated ISBN-13 (preferred) or ISBN-10 from the identifiers dict."""
    for key in ("isbn13", "isbn-13", "isbn", "isbn10", "isbn-10"):
        val = identifiers.get(key)
        if val:
            digits = _isbn_digits(val)
            if digits and _isbn_valid(digits):
                return digits
    # Last-ditch: scan all identifier values for one that looks like an ISBN.
    for val in identifiers.values():
        digits = _isbn_digits(val)
        if digits and _isbn_valid(digits):
            return digits
    return None


def _extract_calibre_series(md: etree._Element) -> tuple[str | None, str | None]:
    """Read ``<meta name="calibre:series"/>`` and ``<meta name="calibre:series_index"/>``."""
    series: str | None = None
    series_index: str | None = None
    for el in md.xpath("./*[local-name()='meta']"):
        name = el.get("name") or ""
        if name == "calibre:series":
            content = el.get("content")
            if content:
                series = content.strip() or None
        elif name == "calibre:series_index":
            content = el.get("content")
            if content:
                # Keep as string so fractional indices like '1.3' survive round-trip.
                series_index = content.strip() or None
    return series, series_index


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _text(el: etree._Element) -> str:
    return (el.text or "").strip()


def _first_text(elements: list[etree._Element]) -> str | None:
    for el in elements:
        t = _text(el)
        if t:
            return t
    return None


def _strip_or_none(s: str | None) -> str | None:
    if s is None:
        return None
    stripped = s.strip()
    return stripped or None


def _nonempty(items: Iterable[str]) -> list[str]:
    return [x for x in (s.strip() for s in items) if x]


def _dedupe_ci(items: Iterable[str]) -> list[str]:
    """Preserve order, drop case-insensitive duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        key = s.casefold()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _clean_publisher(raw: str | None) -> str | None:
    if not raw:
        return None
    v = raw.strip()
    if not v or v.casefold() in _SENTINEL_PUBLISHERS:
        return None
    return v


def _clean_description(raw: str | None) -> str | None:
    """HTML-decode entities but preserve the markup — Calibre stores HTML in comments.

    Drops descriptions that are effectively empty after decoding."""
    if not raw:
        return None
    decoded = html.unescape(raw).strip()
    # Strip if trivially short after decoding — Calibre's "short desc" threshold
    # we used during recon was 20 chars, match that here.
    if len(decoded) < 20:
        return None
    return decoded


def _clean_language(raw: str | None) -> str | None:
    if not raw:
        return None
    v = raw.strip().replace("_", "-")
    return v or None


def _parse_date(raw: str | None) -> str | None:
    """Normalize an OPF date to YYYY, YYYY-MM, or YYYY-MM-DD."""
    if not raw:
        return None
    m = _DATE_PREFIX.match(raw.strip())
    if not m:
        return None
    year, month, day = m.groups()
    # Guard against the Calibre "unset" sentinel (year 0101).
    if int(year) <= 101:
        return None
    parts = [year]
    if month:
        parts.append(month)
        if day:
            parts.append(day)
    return "-".join(parts)


# ---------------------------------------------------------------------------
# ISBN validation
# ---------------------------------------------------------------------------


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _looks_like_uuid(v: str) -> bool:
    return bool(_UUID_RE.match(v.strip()))


def _isbn_digits(raw: str) -> str | None:
    """Strip non-alphanumerics. Return a 10- or 13-digit string (plus trailing X
    for ISBN-10), or None if the shape doesn't match."""
    s = re.sub(r"[^0-9Xx]", "", raw or "")
    if len(s) == 13 and s.isdigit():
        return s
    if len(s) == 10 and (s[:9].isdigit() and s[9] in "0123456789Xx"):
        return s.upper()
    return None


def _isbn_valid(digits: str) -> bool:
    # All-zero ISBNs technically pass the checksum (0 mod 11 = 0, 0 mod 10 = 0)
    # but are always placeholders in the wild, never real ISBNs.
    if not digits or set(digits) <= {"0"}:
        return False
    if len(digits) == 13:
        total = 0
        for i, ch in enumerate(digits):
            n = int(ch)
            total += n if i % 2 == 0 else n * 3
        return total % 10 == 0
    if len(digits) == 10:
        total = 0
        for i, ch in enumerate(digits):
            n = 10 if ch == "X" else int(ch)
            total += n * (10 - i)
        return total % 11 == 0
    return False


def _isbn_to_13(digits: str) -> str | None:
    """Convert a digits-only ISBN-10 to its ISBN-13 form, or return an ISBN-13
    unchanged. Returns None if the input isn't a valid ISBN shape.

    ISBN-13 = '978' + first 9 digits of the ISBN-10 + new EAN-13 check digit.
    (The '979' prefix space exists but is reserved for different publishers,
    so ISBN-10→13 conversion always goes through '978'.)
    """
    if not digits:
        return None
    if len(digits) == 13:
        return digits
    if len(digits) == 10:
        body = "978" + digits[:9]
        total = sum((3 if i % 2 else 1) * int(c) for i, c in enumerate(body))
        check = (10 - total % 10) % 10
        return body + str(check)
    return None
