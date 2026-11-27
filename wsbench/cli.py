"""Command-line interface to the local workbench."""
import argparse
import json
import sys
from pathlib import Path


def _line(verdict):
    row = "{label} (line {line}): {status}".format_map(verdict)
    metrics = verdict["metrics"]
    if verdict["error"]:
        return row + " {code}: {message}".format_map(verdict["error"])
    if verdict["measured"]:
        row += " measured=" + verdict["measured"]
    if "dloss" in metrics:
        row += f" dloss={metrics['dloss']:+.4f}"
    if "passed" in metrics:
        row += " passed={passed}/{total}".format_map(metrics)
    return row


def _progress(file, received, total):
    """One progress line per downloaded file of at least 1 MiB, rewritten in place (only on a terminal)."""
    if total is not None and total < 2**20:
        return
    text = f"{file}: {received / 2**20:.1f} MiB" + (f" of {total / 2**20:.1f} MiB" if total else "")
    print("\r" + text, end="\n" if total and received >= total else "", file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wsbench", description="Execute weight-space experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="start the local web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    run = commands.add_parser("run", help="execute a WSL script")
    run.add_argument("file", type=Path)
    run.add_argument("--json", type=Path, metavar="OUT.json")
    check = commands.add_parser("check", help="check WSL syntax, types and hints")
    check.add_argument("file", type=Path)
    fetch = commands.add_parser("fetch", help="download a pinned model or the ARC corpus "
                                "(all = pythia-14m and arc; the 1.1 GB transfo-xl-wt103 only by name)")
    fetch.add_argument("what", choices=["pythia-14m", "transfo-xl-wt103", "arc", "all"],
                       help="all = pythia-14m and arc; transfo-xl-wt103 (1.1 GB) is fetched only by name")
    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn
        uvicorn.run("wsbench.server.app:app", host=args.host, port=args.port)
        return 0
    if args.command == "fetch":
        try:
            from .models.fetch import fetch as fetch_model
            progress = _progress if sys.stderr.isatty() else None
            if args.what in ("all", "pythia-14m"):
                fetch_model("pythia-14m", progress)
            if args.what == "transfo-xl-wt103":
                fetch_model(args.what, progress)
            if args.what in ("all", "arc"):
                from .arc.data import fetch_arc
                fetch_arc()
            return 0
        except Exception as exc:
            print(f"E_DOWNLOAD {exc}", file=sys.stderr)
            return 1
    try:
        script = args.file.read_text(encoding="utf-8")
        if args.command == "check":
            from .lang.hints import hints
            from .lang.parser import parse
            from .lang.validate import validate
            statements, issues = parse(script)
            errors = issues + validate(statements)
            problems = [(e, "error") for e in errors] + [(h, "warning") for h in hints(statements)]
            for entry, kind in sorted(problems, key=lambda item: (item[0]["line"], item[0]["col"])):
                print(f"{args.file}:{entry['line']}:{entry['col']}: {kind}: {entry['code']} {entry['message']}")
            return int(bool(errors))
        from .exec.executor import run_script
        results = run_script(script)
        for verdict in results:
            print(_line(verdict))
        if args.json:
            args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return int(any(v["status"] == "ERROR" for v in results))
    except OSError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
