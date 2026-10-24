"""Held-out eval tasks: HumanEval (py), MultiPL-E HumanEval (6 langs), edit set (7 langs).

Task = {"id", "lang", "kind": "fresh"|"edit", "prompt", "split": "tune"|"test"}.
Split is a fixed hash of the id, so tuning never sees test tasks.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

BENCH = Path(__file__).parent
ROOT = BENCH.parent
LANG_NAME = {"py": "Python", "js": "JavaScript", "ts": "TypeScript", "go": "Go", "rs": "Rust",
             "java": "Java", "cpp": "C++"}
FENCE = {"py": "python", "js": "javascript", "ts": "typescript", "go": "go", "rs": "rust", "java": "java",
         "cpp": "cpp"}
EDIT_INSTRUCTIONS = [
    "Add a short comment above every function or method explaining what it does.",
    "Refactor this code for readability without changing its behavior.",
    "Add explicit type annotations wherever types are missing (skip if the language already requires them; "
    "then add input validation instead).",
    "Review this code for bugs and fix any you find; also improve error messages.",
]
EXCERPT_LINES = 140


def split_of(task_id: str) -> str:
    return "tune" if int(hashlib.sha1(task_id.encode()).hexdigest(), 16) % 3 == 0 else "test"


def excerpt(text: str, max_lines: int = EXCERPT_LINES) -> str:
    """Whole file if short, else the first max_lines lines cut back to the last blank line."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = lines[:max_lines]
    cut = max((i for i, ln in enumerate(head) if not ln.strip()), default=max_lines)
    return "\n".join(head[:cut])


def _fresh(task_id: str, problem: str, lang: str, stub: str) -> dict:
    prompt = (f"Complete the following {LANG_NAME[lang]} function. Reply with only the complete code "
              f"in one code block.\n\n```{FENCE[lang]}\n{stub.rstrip()}\n```")
    return {"id": task_id, "problem": problem, "lang": lang, "kind": "fresh", "prompt": prompt}


def _all_tasks() -> list[dict]:
    tasks = []
    with gzip.open(ROOT / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tasks.append(_fresh(row["task_id"], "he" + row["task_id"].split("/")[1], "py", row["prompt"]))
    for path in sorted((BENCH / "data" / "multiple").glob("humaneval-*.jsonl")):
        lang = path.stem.split("-", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            problem = "he" + row["name"].split("_")[1]  # HumanEval_12_longest -> he12
            tasks.append(_fresh(f"multiple/{row['name']}/{lang}", problem, lang, row["prompt"]))
    for line in (BENCH / "edit_set" / "files.jsonl").read_text().splitlines():
        m = json.loads(line)
        src = excerpt((BENCH / "data" / m["local"]).read_text(encoding="utf-8", errors="replace"))
        tid = f"edit/{m['repo']}/{m['path']}"
        instr = EDIT_INSTRUCTIONS[int(hashlib.sha1(tid.encode()).hexdigest(), 16) % len(EDIT_INSTRUCTIONS)]
        prompt = (f"{instr} Reply with the full updated file in one code block.\n\n"
                  f"```{FENCE[m['lang']]}\n{src.rstrip()}\n```")
        tasks.append({"id": tid, "problem": tid, "lang": m["lang"], "kind": "edit", "prompt": prompt})
    for t in tasks:
        t["split"] = split_of(t["problem"])  # all translations of a problem share a split
    return tasks


def load_tasks(split: str) -> list[dict]:
    return [t for t in _all_tasks() if t["split"] == split]
