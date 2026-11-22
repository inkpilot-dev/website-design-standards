"""Reading the files people actually have.

A spreadsheet "exported as CSV" is whatever that person's copy of Excel produces: commas in the
US, semicolons and decimal commas in most of Europe, Windows-1252 unless "CSV UTF-8" was picked,
UTF-16 for "Unicode Text", sometimes a title line above the header. Documents arrive as Word
files and PDFs. Everything here turns such files into plain rows or plain text, or raises
:class:`UnreadableFile` with a sentence the person can act on.

Standard library only, except PDFs (pypdf, bundled).
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET  # noqa: S405 - local files the user chose; expat has entity-expansion limits

MAX_XML_BYTES = 200 * 1024 * 1024


class UnreadableFile(ValueError):
    """The file cannot be used; the message says why in plain words (no file name, no jargon)."""


# ------------------------------------------------------------------------------------------
# Text
# ------------------------------------------------------------------------------------------


def decode_text(raw: bytes) -> str:
    """Bytes of a text file -> str, whatever encoding a spreadsheet program chose."""
    if raw.startswith(b"PK\x03\x04"):
        raise UnreadableFile("is not a text file: it looks like an Excel or Word file saved under another name. Add the original .xlsx or .docx instead.")
    if raw.startswith(b"\xef\xbb\xbf"):
        text = raw[3:].decode("utf-8", errors="replace")
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="replace")
    else:
        head = raw[:4096]
        if len(head) >= 4 and head.count(b"\x00") > len(head) // 4:  # UTF-16 without a byte-order mark
            text = raw.decode("utf-16-le" if head[1::2].count(b"\x00") >= head[0::2].count(b"\x00") else "utf-16-be", errors="replace")
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp1252", errors="replace")  # Excel's default "CSV (Comma delimited)" on Windows
    text = text.replace("\x00", "")
    sample = text[:4000]
    if sample:
        junk = sum(1 for ch in sample if ch == "�" or (ord(ch) < 32 and ch not in "\t\r\n\f"))
        if junk > max(8, len(sample) // 20):
            raise UnreadableFile("is not a text file (it contains binary data).")
    return text


def read_text(path: Path) -> str:
    return decode_text(Path(path).read_bytes())


# ------------------------------------------------------------------------------------------
# Tables (CSV / TSV / Excel)
# ------------------------------------------------------------------------------------------

_DELIMITERS = (",", ";", "\t", "|")


def _count_outside_quotes(line: str, delim: str) -> int:
    n, quoted = 0, False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        elif ch == delim and not quoted:
            n += 1
    return n


def sniff_delimiter(text: str, default: str = ",") -> str:
    """The separator most lines agree on. European Excel writes ';' because ',' is its decimal mark."""
    lines = [ln for ln in text.splitlines() if ln.strip()][:40]
    best, best_score = default, 0.0
    for d in _DELIMITERS:
        counts = [_count_outside_quotes(ln, d) for ln in lines]
        if not counts or max(counts) == 0:
            continue
        common, freq = Counter(c for c in counts if c > 0).most_common(1)[0]
        score = freq / len(counts) + min(common, 20) / 1000 + (0.0005 if d == default else 0.0)
        if score > best_score:
            best, best_score = d, score
    return best


def parse_delimited(text: str, default_delimiter: str = ",") -> list[list[str]]:
    delim = sniff_delimiter(text, default_delimiter)
    return [[c for c in row] for row in csv.reader(io.StringIO(text, newline=""), delimiter=delim)]


_SPACES = "    "
_CURRENCY = "$€£¥₹₩₽"
_CODES = ("USD", "EUR", "GBP", "CAD", "AUD", "CHF", "JPY", "INR")
_NUMERIC = re.compile(r"^\d[\d.,']*$")


def _money_core(value: str) -> Optional[tuple[bool, str, bool]]:
    """("$ 1,234.50") -> (negative, "1,234.50", had_symbol); None if it is not written like a number."""
    t = value.strip(_SPACES)
    neg = False
    if len(t) > 2 and t[0] == "(" and t[-1] == ")":  # accounting style
        neg, t = True, t[1:-1].strip(_SPACES)
    had_symbol = False
    for _ in range(3):
        before = t
        t = t.strip(_SPACES)
        if t[:1] in ("-", "−"):
            neg, t = not neg, t[1:]
        elif t[:1] == "+":
            t = t[1:]
        stripped = t.strip(_CURRENCY)
        if stripped != t:
            had_symbol, t = True, stripped
        for code in _CODES:
            if t.startswith(code + " ") or t.startswith(code + " "):
                had_symbol, t = True, t[len(code) :]
            elif t.endswith(" " + code) or t.endswith(" " + code):
                had_symbol, t = True, t[: -len(code)]
        if t == before:
            break
    for sp in _SPACES:
        t = t.replace(sp, "")
    if not t or not _NUMERIC.match(t):
        return None
    return neg, t, had_symbol


def _style_evidence(core: str) -> str:
    """'eu' (decimal comma), 'us' (decimal point) or '' when the digits fit both readings."""
    has_dot, has_comma = "." in core, "," in core
    if has_dot and has_comma:
        return "eu" if core.rfind(",") > core.rfind(".") else "us"
    if has_comma:
        parts = core.split(",")
        grouped = len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:])
        return "" if grouped else "eu"
    if has_dot:
        parts = core.split(".")
        if len(parts) > 2:
            return "eu" if all(len(p) == 3 for p in parts[1:]) and len(parts[0]) <= 3 else ""
        return "" if len(parts[1]) == 3 and len(parts[0]) <= 3 else "us"
    return ""


def number_style(values: list[Any]) -> str:
    """How a column writes numbers: 'eu' only when something in it cannot be read the US way."""
    eu = us = 0
    for v in values:
        if not isinstance(v, str):
            continue
        core = _money_core(v)
        if core is None:
            continue
        ev = _style_evidence(core[1].replace("'", ""))
        eu += ev == "eu"
        us += ev == "us"
    return "eu" if eu and not us else "us"


def to_number(value: str, style: str = "us") -> Any:
    """'$1,234.50' -> 1234.5, '42,50' (eu) -> 42.5, '(15.00)' -> -15.0; anything else is returned as is."""
    core = _money_core(value)
    if core is None:
        return value
    neg, t, had_symbol = core
    t = t.replace("'", "")
    if "." in t and "," in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        t = t.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif style == "eu":
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", "")
    if not re.fullmatch(r"\d+(\.\d+)?", t):
        return value
    # 007, 02134: codes and ZIP codes, not quantities (unless written as money).
    if not had_symbol and len(t) > 1 and t[0] == "0" and not t.startswith("0."):
        return value
    number: Any = float(t) if "." in t else int(t)
    return -number if neg else number


def rows_to_records(rows: list[list[Any]], source: str = "") -> tuple[list[dict], list[str]]:
    """Raw sheet rows -> (records keyed by column name, warnings).

    Skips blank rows, a title line above the header and empty trailing columns; names unnamed
    or repeated columns; reads numbers in the column's own style.
    """
    warnings: list[str] = []
    clean = [[(c.strip(_SPACES + "\t\r\n") if isinstance(c, str) else ("" if c is None else c)) for c in row] for row in rows]
    clean = [row for row in clean if any(c != "" for c in row)]
    if not clean:
        return [], warnings
    width = max(len(r) for r in clean)
    need = 2 if width >= 2 and any(sum(1 for c in r if c != "") >= 2 for r in clean) else 1
    start = next(i for i, r in enumerate(clean) if sum(1 for c in r if c != "") >= need)
    header_row, body = clean[start], clean[start + 1 :]
    looks_like_data = any(not isinstance(c, str) or (c != "" and to_number(c) != c) for c in header_row)
    if looks_like_data:
        body = [header_row] + body
        header_row = []
        warnings.append("there is no header row, so the columns were named 'column 1', 'column 2', and so on. Add a first row with column names (for example: description, category, amount) for better results.")
    names: list[Optional[str]] = []
    seen: Counter = Counter()
    for i in range(width):
        raw = header_row[i] if i < len(header_row) else ""
        name = str(raw).strip()
        if not name:
            # An unnamed column is real only if most rows use it; a stray extra cell in one row
            # (or trailing commas) must not add a field to every example.
            filled = sum(1 for r in body if i < len(r) and r[i] != "")
            if not header_row or filled * 2 >= max(len(body), 1):
                if filled == 0:
                    names.append(None)
                    continue
                name = f"column {i + 1}"
            else:
                names.append(None)
                continue
        seen[name.lower()] += 1
        names.append(name if seen[name.lower()] == 1 else f"{name} ({seen[name.lower()]})")
    styles = {i: number_style([r[i] for r in body if i < len(r)]) for i, n in enumerate(names) if n}
    records = []
    for r in body:
        rec = {}
        for i, name in enumerate(names):
            if name is None:
                continue
            v = r[i] if i < len(r) else ""
            if isinstance(v, str):
                v = None if v == "" else to_number(v, styles[i])
            rec[name] = v
        if any(v is not None for v in rec.values()):
            records.append(rec)
    return records, warnings


def read_table(path: Path) -> tuple[list[dict], list[str]]:
    """Records from a .csv / .tsv / .txt table or the first sheet of an Excel workbook."""
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        rows, warnings = read_xlsx(path)
    else:
        rows, warnings = parse_delimited(read_text(path), "\t" if ext in (".tsv", ".tab") else ","), []
    records, more = rows_to_records(rows)
    return records, warnings + more


# ------------------------------------------------------------------------------------------
# Excel (.xlsx): a zip of XML sheets
# ------------------------------------------------------------------------------------------

_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_BUILTIN_DATE_FORMATS = set(range(14, 23)) | set(range(27, 37)) | {45, 46, 47} | set(range(50, 59))


def _zip_xml(z: zipfile.ZipFile, name: str) -> ET.Element:
    info = z.getinfo(name)
    if info.file_size > MAX_XML_BYTES:
        raise UnreadableFile("is too large to read.")
    return ET.fromstring(z.read(name))  # noqa: S314


def _is_date_format(code: str) -> bool:
    code = re.sub(r'"[^"]*"|\[[^\]]*\]|\\.', "", code or "")
    return bool(re.search(r"[ymdhs]", code, re.I)) and "general" not in code.lower()


def _excel_date(serial: float, date1904: bool) -> str:
    base = _dt.datetime(1904, 1, 1) if date1904 else _dt.datetime(1899, 12, 30)
    try:
        when = base + _dt.timedelta(days=serial)
    except (OverflowError, ValueError):
        return str(serial)
    when = (when + _dt.timedelta(seconds=0.5)).replace(microsecond=0)
    if serial < 1:
        return when.strftime("%H:%M:%S") if when.second else when.strftime("%H:%M")
    if when.hour == when.minute == when.second == 0:
        return when.strftime("%Y-%m-%d")
    return when.strftime("%Y-%m-%d %H:%M")


def _column_index(ref: str) -> Optional[int]:
    letters = "".join(ch for ch in ref if ch.isalpha()).upper()
    if not letters:
        return None
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(path: Path) -> tuple[list[list[Any]], list[str]]:
    """Rows of the first visible sheet, as text, numbers and ISO dates. Returns (rows, warnings)."""
    warnings: list[str] = []
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "xl/workbook.xml" not in names:
                raise UnreadableFile("is not an Excel workbook.")
            shared: list[str] = []
            if "xl/sharedStrings.xml" in names:
                for si in _zip_xml(z, "xl/sharedStrings.xml").iter(_M + "si"):
                    # Rich text is split into runs; phonetic guides (rPh) are not part of the value.
                    parts = [t.text or "" for t in si.findall(_M + "t")] + [t.text or "" for r in si.findall(_M + "r") for t in r.findall(_M + "t")]
                    shared.append("".join(parts))
            date_styles: set[int] = set()
            if "xl/styles.xml" in names:
                styles = _zip_xml(z, "xl/styles.xml")
                custom = {int(n.get("numFmtId", "0")): n.get("formatCode", "") for n in styles.iter(_M + "numFmt")}
                xfs = styles.find(_M + "cellXfs")
                for i, xf in enumerate(list(xfs) if xfs is not None else []):
                    fid = int(xf.get("numFmtId", "0"))
                    if fid in _BUILTIN_DATE_FORMATS or (fid in custom and _is_date_format(custom[fid])):
                        date_styles.add(i)
            workbook = _zip_xml(z, "xl/workbook.xml")
            props = workbook.find(_M + "workbookPr")
            date1904 = props is not None and props.get("date1904") in ("1", "true")
            sheets = [s for s in workbook.iter(_M + "sheet")]
            visible = [s for s in sheets if s.get("state") in (None, "visible")] or sheets
            if not visible:
                raise UnreadableFile("has no sheets.")
            rels = {r.get("Id"): r.get("Target", "") for r in _zip_xml(z, "xl/_rels/workbook.xml.rels")} if "xl/_rels/workbook.xml.rels" in names else {}
            target = rels.get(visible[0].get(_R + "id"), "worksheets/sheet1.xml")
            sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target
            if sheet_path not in names:
                raise UnreadableFile("has a first sheet that is not a normal worksheet (a chart sheet, perhaps). Move the data to the first sheet, or save it as CSV.")
            if len(visible) > 1:
                warnings.append(f"only the first sheet ('{visible[0].get('name', 'Sheet1')}') was used; the workbook has {len(visible)} sheets.")
            rows: list[list[Any]] = []
            for row in _zip_xml(z, sheet_path).iter(_M + "row"):
                cells: dict[int, Any] = {}
                nxt = 0
                for c in row.findall(_M + "c"):
                    idx = _column_index(c.get("r") or "")
                    idx = nxt if idx is None else idx
                    nxt = idx + 1
                    kind, v = c.get("t"), c.find(_M + "v")
                    raw = v.text if v is not None and v.text is not None else ""
                    value: Any
                    if kind == "s":
                        value = shared[int(raw)] if raw.isdigit() and int(raw) < len(shared) else ""
                    elif kind == "inlineStr":
                        value = "".join(t.text or "" for t in c.iter(_M + "t"))
                    elif kind == "b":
                        value = "TRUE" if raw == "1" else "FALSE"
                    elif kind == "e":
                        value = ""  # #N/A, #DIV/0! and friends carry no value
                    elif kind == "str":
                        value = raw
                    elif raw == "":
                        value = ""
                    else:
                        try:
                            num = float(raw)
                        except ValueError:
                            value = raw
                        else:
                            if int(c.get("s") or 0) in date_styles:
                                value = _excel_date(num, date1904)
                            else:
                                value = int(num) if num.is_integer() and abs(num) < 1e15 else round(num, 10)
                    cells[idx] = value
                rows.append([cells.get(i, "") for i in range(max(cells) + 1)] if cells else [])
            return rows, warnings
    except zipfile.BadZipFile as exc:
        raise UnreadableFile("could not be opened. If it is protected with a password, remove the password in Excel (File > Info > Protect Workbook) and add it again; otherwise the file may be damaged.") from exc
    except (ET.ParseError, KeyError, IndexError) as exc:
        raise UnreadableFile("could not be read as an Excel workbook. In Excel choose File > Save As > CSV UTF-8 and add that file instead.") from exc


# ------------------------------------------------------------------------------------------
# Documents
# ------------------------------------------------------------------------------------------

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx(path: Path) -> str:
    """The text of a Word document: paragraphs (and table cells) separated by a blank line, so
    that a heading stays apart from the paragraph after it."""
    try:
        with zipfile.ZipFile(path) as z:
            if "word/document.xml" not in z.namelist():
                raise UnreadableFile("is not a Word document.")
            root = _zip_xml(z, "word/document.xml")
    except zipfile.BadZipFile as exc:
        raise UnreadableFile("could not be opened. If it is protected with a password, remove the password in Word and add it again; otherwise the file may be damaged.") from exc
    except ET.ParseError as exc:
        raise UnreadableFile("could not be read as a Word document. Save it again from Word, or export it as PDF.") from exc
    paragraphs = []
    for p in root.iter(_W + "p"):
        parts = []
        for node in p.iter():
            if node.tag == _W + "t":
                parts.append(node.text or "")
            elif node.tag == _W + "tab":
                parts.append("\t")
            elif node.tag in (_W + "br", _W + "cr"):
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def read_pdf(path: Path) -> list[str]:
    """Text of each page. Raises UnreadableFile for locked or damaged files."""
    from pypdf import PdfReader  # noqa: WPS433
    from pypdf.errors import PyPdfError  # noqa: WPS433

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                unlocked = bool(reader.decrypt(""))
            except Exception:  # noqa: BLE001
                unlocked = False
            if not unlocked:
                raise UnreadableFile("is protected with a password. Open it, save or print it to a new PDF without the password, and add that file.")
        return [(page.extract_text() or "") for page in reader.pages]
    except UnreadableFile:
        raise
    except (PyPdfError, ValueError, KeyError, IndexError, TypeError, AttributeError, RecursionError, OSError) as exc:
        if "not been decrypted" in str(exc).lower() or "password" in str(exc).lower():
            raise UnreadableFile("is protected with a password. Open it, save or print it to a new PDF without the password, and add that file.") from exc
        raise UnreadableFile("could not be read as a PDF (the file may be damaged). Try opening it and saving or printing it to a new PDF.") from exc


LEGACY_HINTS = {
    ".xls": "is in the old Excel format. Open it in Excel and save it as .xlsx or CSV, then add that file.",
    ".doc": "is in the old Word format. Open it in Word and save it as .docx or PDF, then add that file.",
    ".ods": "is an OpenDocument spreadsheet. Save it as .xlsx or CSV, then add that file.",
    ".odt": "is an OpenDocument text file. Save it as .docx or PDF, then add that file.",
    ".numbers": "is an Apple Numbers file. In Numbers choose File > Export To > CSV (or Excel), then add that file.",
    ".pages": "is an Apple Pages file. In Pages choose File > Export To > PDF (or Word), then add that file.",
    ".rtf": "is a Rich Text file. Save it as .docx, PDF or plain text, then add that file.",
}
