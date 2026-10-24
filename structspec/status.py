from __future__ import annotations

import json
import time
import urllib.request


def fetch_status(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=2) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def print_status_loop(base_url: str, once: bool = False, interval: float = 1.0) -> None:
    url = base_url.rstrip("/") + "/health"
    while True:
        try:
            data = fetch_status(url)
            print(
                "\r"
                f"Structspec {data.get('status')} | "
                f"backend={data.get('backend')} | "
                f"active={data.get('requests_active')} | "
                f"requests={data.get('requests_total')} | "
                f"tok/pass={float(data.get('estimated_multiplier', 1.0)):.2f} | "
                f"accept={float(data.get('acceptance_rate', 0.0))*100:.1f}%"
                "      ",
                end="",
                flush=True,
            )
        except Exception as exc:
            print(f"\rStructspec status unavailable: {exc}      ", end="", flush=True)
        if once:
            print()
            return
        time.sleep(interval)


def run_textual_status(base_url: str, interval: float = 1.0) -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer, Header, Static
    except ImportError:
        print("Textual is not installed; falling back to terminal status. Install with: pip install structspec[tui]")
        print_status_loop(base_url, once=False, interval=interval)
        return

    class StatusApp(App):
        CSS = """
        Screen { background: #0b0f14; color: white; }
        #metrics { border: solid cyan; padding: 1; }
        #logs { border: solid #333333; padding: 1; margin-top: 1; }
        """

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static("Connecting...", id="metrics")
            yield Static("Log stream\nwaiting for requests...", id="logs")
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(interval, self.refresh_status)

        def refresh_status(self) -> None:
            try:
                data = fetch_status(base_url.rstrip("/") + "/health")
                text = (
                    f"Structspec 0.2.0 | Server {base_url} | Backend {data.get('backend')}\n\n"
                    f"Generated tokens:   {data.get('generated_tokens')}\n"
                    f"Target passes:      {data.get('target_passes')}\n"
                    f"Tokens per pass:    {float(data.get('estimated_multiplier', 1.0)):.2f}\n"
                    f"Draft acceptance:   {float(data.get('acceptance_rate', 0.0))*100:.1f}%\n"
                    f"Active requests:    {data.get('requests_active')}"
                )
            except Exception as exc:
                text = f"Structspec status unavailable\n{exc}"
            self.query_one("#metrics", Static).update(text)

    StatusApp().run()
