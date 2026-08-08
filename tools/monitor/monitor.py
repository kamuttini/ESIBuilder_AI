"""Entrypoint of the project monitor: raccolta metriche + dashboard.

    python3 tools/monitor/monitor.py all            # raccoglie e rigenera la dashboard
    python3 tools/monitor/monitor.py collect        # solo raccolta
    python3 tools/monitor/monitor.py build          # solo dashboard
    python3 tools/monitor/monitor.py serve          # http://127.0.0.1:8765/

`all` e il comando da lanciare dopo ogni training o eval: rilegge `artifacts/`,
aggiorna lo store e riscrive `artifacts/71_monitor/index.html`.
"""

from __future__ import annotations

import argparse
import errno
import functools
import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dashboard  # noqa: E402
import collect_metrics  # noqa: E402

DEFAULT_ARTIFACTS = "artifacts"
DEFAULT_STORE_NAME = "71_monitor"


def _store_for(artifacts_root: str, store: Optional[str]) -> Path:
    return Path(store).resolve() if store else Path(artifacts_root).resolve() / DEFAULT_STORE_NAME


def cmd_collect(args: argparse.Namespace) -> int:
    store = _store_for(args.artifacts_root, args.store)
    argv = ["--artifacts-root", args.artifacts_root, "--out-dir", str(store)]
    if args.overrides:
        argv += ["--overrides", args.overrides]
    return collect_metrics.main(argv)


def cmd_build(args: argparse.Namespace) -> int:
    store = _store_for(args.artifacts_root, args.store)
    argv = ["--store", str(store)]
    if args.out:
        argv += ["--out", args.out]
    return build_dashboard.main(argv)


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_collect(args)
    return rc or cmd_build(args)


def cmd_serve(args: argparse.Namespace) -> int:
    store = _store_for(args.artifacts_root, args.store)
    index = store / "index.html"
    if not index.exists():
        print(f"index.html mancante in {store}: lancia prima `monitor.py all`", file=sys.stderr)
        return 1
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(store))
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer(("127.0.0.1", args.port), handler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        print(
            f"porta {args.port} gia' occupata (spesso da una galleria di review lasciata aperta).\n"
            f"Rilancia con --port <altra porta>, oppure apri direttamente {index}",
            file=sys.stderr,
        )
        return 1
    with httpd:
        url = f"http://127.0.0.1:{args.port}/index.html"
        print(f"monitor su {url}  (Ctrl+C per uscire)")
        if args.open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor dello stato del progetto ESIBuilder AI")
    parser.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--store", default=None, help="default: <artifacts-root>/71_monitor")
    sub = parser.add_subparsers(dest="command")

    p_collect = sub.add_parser("collect", help="scandisce artifacts/ e aggiorna lo store")
    p_collect.add_argument("--overrides", default=None)
    p_collect.set_defaults(func=cmd_collect, out=None)

    p_build = sub.add_parser("build", help="rigenera la dashboard HTML dallo store")
    p_build.add_argument("--out", default=None)
    p_build.set_defaults(func=cmd_build, overrides=None)

    p_all = sub.add_parser("all", help="collect + build")
    p_all.add_argument("--overrides", default=None)
    p_all.add_argument("--out", default=None)
    p_all.set_defaults(func=cmd_all)

    p_serve = sub.add_parser("serve", help="serve la dashboard in locale")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--open", action="store_true", help="apre il browser")
    p_serve.set_defaults(func=cmd_serve, overrides=None, out=None)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
