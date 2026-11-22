"""Files as people really have them: encodings, separators, number styles, Excel, Word."""

import zipfile
from xml.sax.saxutils import escape

import pytest

from engine import readers
from engine.readers import UnreadableFile, decode_text, number_style, rows_to_records, sniff_delimiter, to_number


def test_decoding_covers_what_excel_writes():
    text = "Café Müller – déjeuner €12"
    assert decode_text(text.encode("utf-8")) == text
    assert decode_text(text.encode("utf-8-sig")) == text
    assert decode_text(text.encode("cp1252")) == text  # "CSV (Comma delimited)" on Windows
    assert decode_text(text.encode("utf-16")) == text  # "Unicode Text"
    assert decode_text("a,b\n1,2\n".encode("utf-16-le")) == "a,b\n1,2\n"  # UTF-16 without a byte-order mark
    assert decode_text(b"a,b\x00\n1,2\n") == "a,b\n1,2\n"
    with pytest.raises(UnreadableFile):
        decode_text(b"PK\x03\x04" + b"\x00" * 50)
    with pytest.raises(UnreadableFile):
        decode_text(bytes(range(256)) * 20)


def test_delimiter_is_sniffed_per_file():
    assert sniff_delimiter("a,b,c\n1,2,3\n") == ","
    assert sniff_delimiter("a;b;c\n1,5;2;3\n4,25;5;6\n") == ";"  # decimal commas must not win
    assert sniff_delimiter("a\tb\tc\n1\t2\t3\n") == "\t"
    assert sniff_delimiter("a|b\n1|2\n") == "|"
    assert sniff_delimiter('note,amount\n"x; y; z",3\n"p; q",4\n') == ","
    assert sniff_delimiter("just one column\nvalue\n") == ","
    assert sniff_delimiter("a\tb\n1\t2\n", default="\t") == "\t"


@pytest.mark.parametrize(
    "text,style,expected",
    [
        ("42", "us", 42),
        ("$42.00", "us", 42.0),
        ("1,234.56", "us", 1234.56),
        ("1,234", "us", 1234),
        ("(15.00)", "us", -15.0),
        ("-7.25", "us", -7.25),
        ("€ 12,50", "eu", 12.5),
        ("12,50 €", "eu", 12.5),
        ("1.234,56", "us", 1234.56),  # both separators present: unambiguous whatever the column style
        ("1.234,56", "eu", 1234.56),
        ("1 234,56", "eu", 1234.56),
        ("42,50", "eu", 42.5),
        ("1.234", "eu", 1234),
        ("£3", "us", 3),
        ("USD 1,000", "us", 1000),
        ("0042", "us", "0042"),  # codes keep their zeros
        ("02134", "us", "02134"),
        ("0.5", "us", 0.5),
        ("15 %", "us", "15 %"),
        ("1e3", "us", "1e3"),
        ("12/03/2026", "us", "12/03/2026"),
        ("fuel", "us", "fuel"),
        ("", "us", ""),
        ("-", "us", "-"),
        ("1.2.3", "us", "1.2.3"),
    ],
)
def test_to_number(text, style, expected):
    got = to_number(text, style)
    assert got == expected and type(got) is type(expected)


def test_number_style_is_decided_per_column():
    assert number_style(["42,50", "1,5", "100"]) == "eu"
    assert number_style(["1.234,56", "12,00"]) == "eu"
    assert number_style(["1,234", "12,345"]) == "us"  # could be either: read the US way
    assert number_style(["1,234", "12.50"]) == "us"
    assert number_style(["42,50", "12.75"]) == "us"  # conflicting evidence: do not guess
    assert number_style(["fuel", "meals", None, 3]) == "us"


def test_rows_to_records_handles_titles_blanks_and_odd_headers():
    rows = [["Expenses 2026", "", ""], [], [" Description ", "Category", "Amount", ""], ["Shell", "fuel", "42,50", ""], ["", "", "", ""], ["Zoom", "software", "15,00", ""], ["stray", "", "", "", "x"]]
    records, warnings = rows_to_records(rows)
    assert warnings == []
    assert records == [{"Description": "Shell", "Category": "fuel", "Amount": 42.5}, {"Description": "Zoom", "Category": "software", "Amount": 15.0}, {"Description": "stray", "Category": None, "Amount": None}]
    dup, _ = rows_to_records([["a", "a", "b"], ["1", "2", "3"]])
    assert dup == [{"a": 1, "a (2)": 2, "b": 3}]
    headerless, warns = rows_to_records([["Shell gas", "fuel", "42"], ["Zoom plan", "software", "15"]])
    assert headerless[0] == {"column 1": "Shell gas", "column 2": "fuel", "column 3": 42} and len(headerless) == 2 and warns
    assert rows_to_records([]) == ([], [])
    assert rows_to_records([["only", "header"]]) == ([], [])


def _xlsx(path, rows, date_cols=(), sheets=1, inline=False):
    strings: list[str] = []

    def cell(ci, ri, v):
        ref = chr(65 + ci) + str(ri + 1)
        if isinstance(v, str):
            if inline:
                return f'<c r="{ref}" t="inlineStr"><is><t>{escape(v)}</t></is></c>'
            if v not in strings:
                strings.append(v)
            return f'<c r="{ref}" t="s"><v>{strings.index(v)}</v></c>'
        if isinstance(v, bool):
            return f'<c r="{ref}" t="b"><v>{int(v)}</v></c>'
        style = ' s="1"' if ci in date_cols and ri else ""
        return f'<c r="{ref}"{style}><v>{v}</v></c>'

    body = "".join(f'<row r="{ri + 1}">' + "".join(cell(ci, ri, v) for ci, v in enumerate(r) if v != "") + "</row>" for ri, r in enumerate(rows))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml", f'<workbook {ns} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join(f'<sheet name="Data{i + 1}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i in range(sheets)) + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(f'<Relationship Id="rId{i + 1}" Type="x" Target="worksheets/sheet{i + 1}.xml"/>' for i in range(sheets)) + "</Relationships>")
        z.writestr("xl/styles.xml", f'<styleSheet {ns}><numFmts count="1"><numFmt numFmtId="164" formatCode="yyyy\\-mm\\-dd"/></numFmts><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs></styleSheet>')
        z.writestr("xl/sharedStrings.xml", f"<sst {ns}>" + "".join(f"<si><t>{escape(x)}</t></si>" for x in strings) + "</sst>")
        for i in range(sheets):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", f"<worksheet {ns}><sheetData>{body}</sheetData></worksheet>")
    return path


def test_xlsx_first_sheet_strings_numbers_dates_and_gaps(tmp_path):
    rows = [("Description", "Category", "Amount", "Date", "Paid"), ("Shell & Co <gas>", "fuel", 42.5, 46023, True), ("Zoom", "", 15, 46024.5, False)]
    records, warnings = readers.read_table(_xlsx(tmp_path / "book.xlsx", rows, date_cols={3}, sheets=2))
    assert records == [
        {"Description": "Shell & Co <gas>", "Category": "fuel", "Amount": 42.5, "Date": "2026-01-01", "Paid": "TRUE"},
        {"Description": "Zoom", "Category": None, "Amount": 15, "Date": "2026-01-02 12:00", "Paid": "FALSE"},
    ]
    assert any("first sheet" in w for w in warnings)
    inline, _ = readers.read_table(_xlsx(tmp_path / "inline.xlsx", rows[:2], inline=True))
    assert inline[0]["Description"] == "Shell & Co <gas>"


def test_xlsx_and_docx_failures_are_explained(tmp_path):
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(UnreadableFile, match="password|damaged"):
        readers.read_table(bad)
    notbook = tmp_path / "zip.xlsx"
    with zipfile.ZipFile(notbook, "w") as z:
        z.writestr("hello.txt", "hi")
    with pytest.raises(UnreadableFile, match="not an Excel workbook"):
        readers.read_table(notbook)
    with pytest.raises(UnreadableFile, match="not a Word document"):
        readers.read_docx(notbook)


def test_docx_text_without_python_docx(tmp_path):
    w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    xml = f'<w:document {w}><w:body><w:p><w:r><w:t>Rent is </w:t></w:r><w:r><w:t xml:space="preserve">$1,850 per month.</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Deposit</w:t><w:tab/><w:t>$2,400</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p/><w:p><w:r><w:delText>removed</w:delText><w:t>Line one</w:t><w:br/><w:t>line two</w:t></w:r></w:p></w:body></w:document>'
    path = tmp_path / "lease.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)
    # Paragraphs are separated by a blank line; a line break inside a paragraph stays a line break.
    assert readers.read_docx(path) == "Rent is $1,850 per month.\n\nDeposit\t$2,400\n\nLine one\nline two"
