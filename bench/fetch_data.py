"""Download held-out eval data (user-approved 2026-10-24) into bench/data/.

MultiPL-E HumanEval prompts (6 langs), edit-set files and a mining corpus, both
from GitHub at a pinned commit recorded in the manifests. Edit repos and mining
repos are disjoint by construction (asserted in tests). Run: python -m bench.fetch_data
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

DATA = Path(__file__).parent / "data"
MULTIPLE_LANGS = ["js", "ts", "go", "rs", "java", "cpp"]
EXT_LANG = {".py": "py", ".ts": "ts", ".js": "js", ".go": "go", ".rs": "rs", ".java": "java",
            ".cpp": "cpp", ".cc": "cpp", ".h": "cpp", ".hpp": "cpp"}

# (owner/repo, path prefix, files to take). Licenses: see docs/research/speed-findings.md.
EDIT_REPOS = [("pallets/click", "src/click/", 6), ("sindresorhus/ky", "source/", 6),
              ("expressjs/express", "lib/", 5), ("spf13/cobra", "", 6),
              ("BurntSushi/ripgrep", "crates/", 6), ("google/gson", "gson/src/main/", 6),
              ("fmtlib/fmt", "", 5)]
MINING_REPOS = [("fastapi/typer", "typer/", 20), ("colinhacks/zod", "packages/zod/src/", 20),
                ("lodash/lodash", "", 20), ("urfave/cli", "", 20), ("clap-rs/clap", "clap_builder/src/", 20),
                ("square/okhttp", "okhttp/src/", 20), ("nlohmann/json", "include/", 20)]
EDIT_BYTES = (1500, 40000)   # long files are cut to a ~140-line excerpt in datasets.py
MINING_BYTES = (800, 30000)


def fetch_multiple(client: httpx.Client) -> None:
    for lang in MULTIPLE_LANGS:
        rows, off = [], 0
        while True:
            r = client.get("https://datasets-server.huggingface.co/rows",
                           params={"dataset": "nuprl/MultiPL-E", "config": f"humaneval-{lang}",
                                   "split": "test", "offset": off, "length": 100})
            r.raise_for_status()
            batch = [x["row"] for x in r.json()["rows"]]
            rows += batch
            off += len(batch)
            if len(batch) < 100:
                break
        out = DATA / "multiple" / f"humaneval-{lang}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps({"name": x["name"], "prompt": x["prompt"]}) for x in rows))
        print(lang, len(rows))


def fetch_repo_files(client: httpx.Client, repos, size_range, subdir: str) -> list[dict]:
    # token (if set) goes to api.github.com only: unauthenticated limit is 60 req/h
    gh = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.environ.get("GITHUB_TOKEN") else {}
    manifest = []
    for repo, prefix, n in repos:
        meta = client.get(f"https://api.github.com/repos/{repo}", headers=gh).json()
        sha = client.get(f"https://api.github.com/repos/{repo}/commits/{meta['default_branch']}", headers=gh).json()["sha"]
        tree = client.get(f"https://api.github.com/repos/{repo}/git/trees/{sha}", params={"recursive": 1}, headers=gh).json()
        picked = [t for t in sorted(tree["tree"], key=lambda t: t["path"])
                  if t["type"] == "blob" and t["path"].startswith(prefix)
                  and Path(t["path"]).suffix in EXT_LANG and size_range[0] <= t["size"] <= size_range[1]
                  and not Path(t["path"]).name.endswith((".d.ts", ".min.js"))][:n]
        for t in picked:
            raw = client.get(f"https://raw.githubusercontent.com/{repo}/{sha}/{t['path']}")
            raw.raise_for_status()
            dest = DATA / subdir / repo.replace("/", "__") / t["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw.content)
            manifest.append({"repo": repo, "sha": sha, "path": t["path"], "lang": EXT_LANG[Path(t["path"]).suffix],
                             "local": str(dest.relative_to(DATA)).replace("\\", "/")})
        print(repo, len(picked))
    return manifest


def main() -> None:
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        fetch_multiple(c)
        edit = fetch_repo_files(c, EDIT_REPOS, EDIT_BYTES, "edit_src")
        mining = fetch_repo_files(c, MINING_REPOS, MINING_BYTES, "mining")
    here = Path(__file__).parent
    (here / "edit_set").mkdir(exist_ok=True)
    (here / "edit_set" / "files.jsonl").write_text("\n".join(json.dumps(m) for m in edit))
    (here / "mining_manifest.jsonl").write_text("\n".join(json.dumps(m) for m in mining))


if __name__ == "__main__":
    main()
