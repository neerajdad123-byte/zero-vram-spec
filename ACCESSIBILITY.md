# Accessibility

Structspec is a command-line tool and an OpenAI-compatible HTTP server. It has no graphical interface of its own.

## Commitment

We want everyone to be able to install, run and contribute to Structspec, including people who use screen readers, keyboard-only navigation or high-contrast terminals.

## Supported environments

- **Command line:** all commands are plain text. Output uses no color or animation that is required to understand it.
- **Server:** `structspec serve` speaks the standard OpenAI API, so it works with any accessible client or editor you already use.
- **Documentation:** written in Markdown with headings, tables with header rows, and descriptive link text, so it reads well in screen readers and in plain-text form.

## Known limitations

- `structspec status` shows a live, refreshing view in the terminal. Refreshing output can be hard to follow with a screen reader. Use the server's metrics endpoint (`GET /v1/structspec/metrics`, plain JSON) instead.
- Benchmark results in `docs/research/` include wide tables that may be hard to navigate with a screen reader. The raw data is also available as JSON/JSONL files in `bench/results/`.

## Reporting barriers

If something in Structspec or its documentation is hard to use with your setup, please open an issue describing the barrier, your environment and the assistive technology you use. Accessibility issues are treated as bugs.
