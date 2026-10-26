"""Dev CLI for driving the agent without the API or UI.

    python -m src.cli chat u1 "Explain transformers in more detail"
    python -m src.cli set u1 stated length concise
    python -m src.cli profile u1
    python -m src.cli log u1
    python -m src.cli accept u1 length
"""
import argparse
import json
import sys

from src import config
from src.agent.core import Agent
from src.agent.llm_client import LLMClient, LLMError


def main() -> None:
    try:
        _run()
    except (LLMError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


def _run() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="send a message through the full loop")
    chat.add_argument("user_id")
    chat.add_argument("message")

    set_cmd = sub.add_parser("set", help="set a profile value")
    set_cmd.add_argument("user_id")
    set_cmd.add_argument("kind", choices=["stated", "observed"])
    set_cmd.add_argument("dimension")
    set_cmd.add_argument("value")

    for name in ("profile", "log"):
        sub.add_parser(name).add_argument("user_id")

    accept = sub.add_parser("accept", help="adopt observed value as stated")
    accept.add_argument("user_id")
    accept.add_argument("dimension")

    args = parser.parse_args()
    llm = LLMClient.from_config() if args.command == "chat" else None
    agent = Agent(config.DB_PATH, llm)

    if args.command == "chat":
        result = agent.respond(args.user_id, args.message)
        print(f"[{result['model_chosen'].upper()} · {result['model_id']} · {result['latency_ms']} ms]")
        print(f"why: {result['routing_reason']}")
        if result["observed_updates"]:
            print(f"learned: {result['observed_updates']}")
        if result["suggestion"]:
            print(f"ask: {result['suggestion']}")
        print("\n" + result["response"])
    elif args.command == "set":
        update = agent.profiles.update_stated if args.kind == "stated" else agent.profiles.update_observed
        update(args.user_id, args.dimension, args.value)
        print("ok")
    elif args.command == "profile":
        stated, observed = agent.profiles.get_all_profiles(args.user_id)
        print(json.dumps({"stated": stated, "observed": observed}, indent=2))
    elif args.command == "log":
        for d in agent.logger.get_recent_decisions(args.user_id):
            print(f"{d['timestamp']}  {d['model_used']:<5}  {d['conflict_score']:.2f}  {d['user_message'][:50]}")
    elif args.command == "accept":
        agent.accept_observed(args.user_id, args.dimension)
        print("ok")


if __name__ == "__main__":
    main()
