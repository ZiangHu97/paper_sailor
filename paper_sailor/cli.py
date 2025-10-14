from __future__ import annotations

import argparse
import json

from .agent import run_session
from .storage import list_sessions
from .workflow import run_planner_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paper-sailor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a minimal session")
    p_run.add_argument("--topic", required=True)
    p_run.add_argument("--session", required=True, help="session id")
    p_run.add_argument("--max-papers", type=int, default=10)

    p_plan = sub.add_parser("plan", help="Run planner-driven exploration")
    p_plan.add_argument("--topic", required=True)
    p_plan.add_argument("--session", required=True)
    p_plan.add_argument("--max-rounds", type=int, default=6)
    p_plan.add_argument("--search-limit", type=int, default=8)

    sub.add_parser("sessions", help="List sessions")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        note = run_session(args.topic, args.session, max_papers=args.max_papers)
        print(json.dumps(note, ensure_ascii=False, indent=2))
        return 0
    elif args.cmd == "plan":
        note = run_planner_session(
            args.topic,
            args.session,
            max_rounds=args.max_rounds,
            search_limit=args.search_limit,
        )
        print(json.dumps(note, ensure_ascii=False, indent=2))
        return 0
    elif args.cmd == "sessions":
        for s in list_sessions():
            print(s)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
