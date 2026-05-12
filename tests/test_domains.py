from __future__ import annotations

from structspec.domains import detect_domain, rules_for_domain


def test_detect_python():
    assert detect_domain("def fib(n):\n    pass") == "python"
    assert detect_domain("class Foo:\n    pass") == "python"
    assert detect_domain("import os") == "python"


def test_detect_json():
    assert detect_domain('{"key": "value"}') == "json"
    assert detect_domain("[1, 2, 3]") == "json"


def test_detect_html():
    assert detect_domain("<div>hello</div>") == "html"
    assert detect_domain("<!doctype html>") == "html"


def test_detect_sql():
    assert detect_domain("SELECT * FROM users") == "sql"
    assert detect_domain("INSERT INTO foo") == "sql"


def test_detect_go():
    assert detect_domain("package main") == "go"
    assert detect_domain("func main() {}") == "go"


def test_detect_generic():
    assert detect_domain("hello world") == "generic"


def test_rules_for_domain():
    assert "indent" in rules_for_domain("python")
    assert "brackets" in rules_for_domain("json")
    assert "tags" in rules_for_domain("html")
    assert "ngram" in rules_for_domain("generic")
    assert rules_for_domain("unknown") == ["ngram"]
