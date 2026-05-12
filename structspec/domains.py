from __future__ import annotations

import re

DOMAIN_RULES = [
    ("python", re.compile(r"^\s*(def |class |import |from |```python|async def )", re.I)),
    ("json", re.compile(r"^\s*(\{|\[|```json)", re.I)),
    ("html", re.compile(r"^\s*(<!doctype|<html|<div|```html)", re.I)),
    ("sql", re.compile(r"^\s*(select|insert|update|delete|create|with)\b", re.I)),
    ("go", re.compile(r"^\s*(package |func |```go)", re.I)),
]


def detect_domain(text: str, default: str = "generic") -> str:
    sample = text.strip()
    for domain, pattern in DOMAIN_RULES:
        if pattern.search(sample):
            return domain
    return default


def rules_for_domain(domain: str) -> list[str]:
    return {
        "python": ["indent", "brackets", "keywords", "ngram"],
        "json": ["quotes", "brackets", "commas", "ngram"],
        "html": ["tags", "indent", "ngram"],
        "sql": ["keywords", "clauses", "ngram"],
        "go": ["braces", "keywords", "ngram"],
        "generic": ["ngram"],
    }.get(domain, ["ngram"])
