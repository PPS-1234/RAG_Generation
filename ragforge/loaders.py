"""
Document loading, dispatched by file extension.

Design principle behind the whole project: NEW DOCUMENT SETS AND NEW FORMATS
NEVER REQUIRE CODE CHANGES to the pipeline itself. Adding a new format means
registering one function in LOADERS below - nothing in pipeline.py, cli.py,
or the vector store changes.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field


@dataclass
class RawDoc:
    """One loaded file, before chunking."""
    source: str            # file path, becomes a citation later
    text: str
    metadata: dict = field(default_factory=dict)


def _load_txt(path: str) -> list[RawDoc]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [RawDoc(source=path, text=f.read())]


def _load_pdf(path: str) -> list[RawDoc]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(RawDoc(source=path, text=text, metadata={"page": i}))
    return docs


def _load_docx(path: str) -> list[RawDoc]:
    import docx  # python-docx

    d = docx.Document(path)
    text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    return [RawDoc(source=path, text=text)] if text.strip() else []


def _load_csv(path: str) -> list[RawDoc]:
    docs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return docs
    header, body = rows[0], rows[1:]
    # One "document" per row, rendered as "col: value" pairs - keeps each row
    # independently retrievable instead of dumping the whole CSV as one blob.
    for i, row in enumerate(body, start=1):
        pairs = [f"{h}: {v}" for h, v in zip(header, row)]
        docs.append(RawDoc(source=path, text="; ".join(pairs), metadata={"row": i}))
    return docs


# Registry: extension -> loader function. Adding a format = adding one line here.
LOADERS = {
    ".txt": _load_txt,
    ".md": _load_txt,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".csv": _load_csv,
}


def load_directory(docs_dir: str) -> list[RawDoc]:
    """Walks docs_dir recursively, loads every supported file, skips the rest."""
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Not a directory: {docs_dir}")

    all_docs: list[RawDoc] = []
    skipped: list[str] = []

    for root, _, files in os.walk(docs_dir):
        for fname in sorted(files):
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            loader = LOADERS.get(ext)
            if loader is None:
                skipped.append(path)
                continue
            try:
                all_docs.extend(loader(path))
            except Exception as e:
                skipped.append(f"{path} (error: {e})")

    if skipped:
        print(f"[loaders] Skipped {len(skipped)} unsupported/failed file(s): {skipped}")

    return all_docs
