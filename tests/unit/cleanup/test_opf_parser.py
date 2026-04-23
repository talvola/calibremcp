"""Unit tests for the OPF parser.

These are pure tests driven by inline XML bytes — they don't need any EPUB
fixtures. A separate integration test in tests/integration/cleanup/ exercises
the parser against real sampled EPUBs from Erik's library.
"""

from __future__ import annotations

import pytest

from calibre_mcp.cleanup.opf_parser import (
    OpfMetadata,
    _isbn_digits,
    _isbn_valid,
    parse_opf_bytes,
)


# ---------------------------------------------------------------------------
# Minimal OPF builders — let each test state only the fields it cares about.
# ---------------------------------------------------------------------------


def _opf(metadata_xml: str, *, default_ns: bool = True) -> bytes:
    """Wrap a ``<metadata>`` fragment into a full OPF document.

    ``default_ns=False`` produces an OPF where the outer package uses a
    default namespace but ``<metadata>`` does not — mirrors the Hawkins
    sample we saw in Erik's library."""
    if default_ns:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:opf="http://www.idpf.org/2007/opf">'
            f"{metadata_xml}"
            "</metadata></package>"
        ).encode("utf-8")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"{metadata_xml}"
        "</metadata></package>"
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Basic field extraction
# ---------------------------------------------------------------------------


def test_extracts_core_fields() -> None:
    opf = parse_opf_bytes(_opf("""
        <dc:title>The Windup Girl</dc:title>
        <dc:creator opf:role="aut">Paolo Bacigalupi</dc:creator>
        <dc:publisher>Night Shade Books</dc:publisher>
        <dc:date>2009-09-01</dc:date>
        <dc:language>en-US</dc:language>
        <dc:identifier opf:scheme="ISBN">9781597801584</dc:identifier>
    """))
    assert opf is not None
    assert opf.title == "The Windup Girl"
    assert opf.authors == ("Paolo Bacigalupi",)
    assert opf.publisher == "Night Shade Books"
    assert opf.pubdate == "2009-09-01"
    assert opf.language == "en-US"
    assert opf.isbn == "9781597801584"


def test_multiple_authors_preserved_in_order() -> None:
    opf = parse_opf_bytes(_opf("""
        <dc:title>Collaborative Work</dc:title>
        <dc:creator>First Author</dc:creator>
        <dc:creator>Second Author</dc:creator>
        <dc:creator>  </dc:creator>
    """))
    assert opf is not None
    assert opf.authors == ("First Author", "Second Author")


# ---------------------------------------------------------------------------
# ISBN validation and normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("9780141036144", "9780141036144"),       # valid ISBN-13
        ("978-0-14-103614-4", "9780141036144"),    # valid, dashes stripped
        ("0141036141", "0141036141"),              # valid ISBN-10
        ("014103614X", None),                      # ISBN-10 shape but bad checksum
        ("0141036142", None),                      # ISBN-10 bad checksum
        ("9780141036145", None),                   # ISBN-13 bad checksum
        ("not-an-isbn", None),
        ("", None),
    ],
)
def test_isbn_extraction(raw: str, expected: str | None) -> None:
    opf = parse_opf_bytes(_opf(
        f'<dc:title>T</dc:title>'
        f'<dc:identifier opf:scheme="ISBN">{raw}</dc:identifier>'
    ))
    assert opf is not None
    assert opf.isbn == expected


def test_isbn_inferred_from_unmarked_identifier() -> None:
    # Some OPFs omit scheme attribute but the value clearly is an ISBN.
    opf = parse_opf_bytes(_opf(
        '<dc:title>T</dc:title>'
        '<dc:identifier id="bookid">9780698185395</dc:identifier>'
    ))
    assert opf is not None
    assert opf.isbn == "9780698185395"


def test_isbn_digits_helper() -> None:
    assert _isbn_digits("  978 0 14 1036144  ") == "9780141036144"
    assert _isbn_digits("0-141-03614-1") == "0141036141"
    assert _isbn_digits("ABC") is None
    assert _isbn_digits("") is None


def test_isbn_valid_helper() -> None:
    assert _isbn_valid("9780141036144") is True
    assert _isbn_valid("0141036141") is True
    assert _isbn_valid("9780141036145") is False
    assert _isbn_valid("0000000000") is True   # all zeros is a valid checksum; acceptable false positive


# ---------------------------------------------------------------------------
# Publisher normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Penguin Group US", "Penguin Group US"),
        ("ManyBooks.net", None),
        ("Project Gutenberg", None),
        ("Project Gutenberg Literary Archive Foundation", None),
        ("Unknown", None),
        ("N/A", None),
        ("", None),
        ("   ", None),
        ("  Orbit  ", "Orbit"),
    ],
)
def test_publisher_sentinel_rejection(raw: str, expected: str | None) -> None:
    opf = parse_opf_bytes(_opf(
        f"<dc:title>T</dc:title><dc:publisher>{raw}</dc:publisher>"
    ))
    assert opf is not None
    assert opf.publisher == expected


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2014-11-10", "2014-11-10"),
        ("2009-09-01T00:00:00Z", "2009-09-01"),
        ("2010-02-16 20:20:50+00:00", "2010-02-16"),
        ("2009", "2009"),
        ("2009-05", "2009-05"),
        ("0101-01-01", None),                  # Calibre "missing" sentinel
        ("not a date", None),
        ("", None),
    ],
)
def test_date_parsing(raw: str, expected: str | None) -> None:
    opf = parse_opf_bytes(_opf(
        f"<dc:title>T</dc:title><dc:date>{raw}</dc:date>"
    ))
    assert opf is not None
    assert opf.pubdate == expected


# ---------------------------------------------------------------------------
# Description: HTML-decoding + empty-drop
# ---------------------------------------------------------------------------


def test_description_html_entities_decoded() -> None:
    opf = parse_opf_bytes(_opf(
        "<dc:title>T</dc:title>"
        "<dc:description>&lt;b&gt;A gripping&amp;#8212;thriller&lt;/b&gt; that&amp;#8217;s "
        "hard to put down, across several lines of text.</dc:description>"
    ))
    assert opf is not None
    assert opf.description is not None
    assert "&lt;" not in opf.description
    assert "<b>" in opf.description


def test_short_description_dropped() -> None:
    opf = parse_opf_bytes(_opf(
        "<dc:title>T</dc:title><dc:description>short</dc:description>"
    ))
    assert opf is not None
    assert opf.description is None


# ---------------------------------------------------------------------------
# Subjects / tags
# ---------------------------------------------------------------------------


def test_subjects_deduped_case_insensitive() -> None:
    opf = parse_opf_bytes(_opf("""
        <dc:title>T</dc:title>
        <dc:subject>Science Fiction</dc:subject>
        <dc:subject>science fiction</dc:subject>
        <dc:subject>Space Opera</dc:subject>
        <dc:subject></dc:subject>
    """))
    assert opf is not None
    assert opf.subjects == ("Science Fiction", "Space Opera")


# ---------------------------------------------------------------------------
# calibre:series / calibre:series_index
# ---------------------------------------------------------------------------


def test_calibre_series_extracted() -> None:
    opf = parse_opf_bytes(_opf("""
        <dc:title>Foundation and Empire</dc:title>
        <meta name="calibre:series" content="Foundation"/>
        <meta name="calibre:series_index" content="2"/>
    """))
    assert opf is not None
    assert opf.series == "Foundation"
    assert opf.series_index == "2"


def test_calibre_series_index_preserves_fractional_string() -> None:
    # Critical for Erik's 1.3/1.5/0.5 conventions.
    opf = parse_opf_bytes(_opf("""
        <dc:title>Black Company: Side Stories</dc:title>
        <meta name="calibre:series" content="Black Company"/>
        <meta name="calibre:series_index" content="2.5"/>
    """))
    assert opf is not None
    assert opf.series == "Black Company"
    assert opf.series_index == "2.5"


def test_calibre_series_missing_both() -> None:
    opf = parse_opf_bytes(_opf("<dc:title>T</dc:title>"))
    assert opf is not None
    assert opf.series is None
    assert opf.series_index is None


# ---------------------------------------------------------------------------
# Namespace handling edge cases
# ---------------------------------------------------------------------------


def test_metadata_without_dc_prefix_still_parsed() -> None:
    """Hawkins-sample shape: <metadata xmlns:dc="..."> with dc-prefixed kids."""
    opf = parse_opf_bytes(_opf(
        '<dc:title>The Girl on the Train</dc:title>'
        '<dc:creator>Paula Hawkins</dc:creator>'
        '<dc:identifier id="bookid">9780698185395</dc:identifier>',
        default_ns=False,
    ))
    assert opf is not None
    assert opf.title == "The Girl on the Train"
    assert opf.authors == ("Paula Hawkins",)
    assert opf.isbn == "9780698185395"


def test_malformed_xml_recovers() -> None:
    # Unclosed tag — lxml with recover=True should still produce a tree.
    # The critical guarantee is the parser doesn't raise. What lxml manages
    # to extract from a broken tree is lxml-version-dependent and not worth
    # pinning in a test.
    data = (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:title>Recovered Title'           # <-- missing </dc:title>
        '<dc:creator>An Author</dc:creator>'
        '</metadata></package>'
    ).encode("utf-8")
    opf = parse_opf_bytes(data)
    assert opf is not None


def test_completely_invalid_input_returns_none() -> None:
    assert parse_opf_bytes(b"not xml at all") is None
    assert parse_opf_bytes(b"") is None


def test_opf_without_metadata_returns_none() -> None:
    data = (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">'
        '<manifest/></package>'
    ).encode("utf-8")
    assert parse_opf_bytes(data) is None


# ---------------------------------------------------------------------------
# Identifier catalog (non-ISBN)
# ---------------------------------------------------------------------------


def test_goodreads_and_amazon_identifiers_captured() -> None:
    opf = parse_opf_bytes(_opf("""
        <dc:title>T</dc:title>
        <dc:identifier opf:scheme="ISBN">9780141036144</dc:identifier>
        <dc:identifier opf:scheme="goodreads">12345</dc:identifier>
        <dc:identifier opf:scheme="AMAZON">B001XYZ</dc:identifier>
    """))
    assert opf is not None
    assert opf.identifiers.get("goodreads") == "12345"
    assert opf.identifiers.get("amazon") == "B001XYZ"


# ---------------------------------------------------------------------------
# Frozen dataclass hygiene
# ---------------------------------------------------------------------------


def test_default_opf_metadata_is_empty() -> None:
    m = OpfMetadata()
    assert m.title is None
    assert m.authors == ()
    assert m.identifiers == {}
    assert m.subjects == ()
