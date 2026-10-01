#!/usr/bin/env python3
"""
RAGForge CLI - a runtime RAG generator.

    python main.py ingest --docs ./sample_docs --collection demo
    python main.py ask --collection demo --question "What is this document about?"
    python main.py chat --collection demo
    python main.py list-collections

Switching to a completely different document set is:
    python main.py ingest --docs ./some_other_folder --collection other
    python main.py ask --collection other --question "..."
No code was touched to do that - this is the core requirement of the assessment.
"""
import argparse
import json
import sys

from ragforge import pipeline, vectorstore


def cmd_ingest(args):
    result = pipeline.ingest(args.docs, args.collection)
    print(json.dumps(result, indent=2))


def cmd_ask(args):
    result = pipeline.ask(args.collection, args.question, top_k=args.top_k)
    print(f"\nProvider: {result['provider']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (distance={s['distance']})")


def cmd_chat(args):
    print(f"RAGForge interactive chat - collection '{args.collection}'. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = pipeline.ask(args.collection, question, top_k=args.top_k)
        print(f"\nAssistant [{result['provider']}]: {result['answer']}\n")
        for s in result["sources"]:
            print(f"  source: {s['source']} (distance={s['distance']})")
        print()


def cmd_list_collections(args):
    for name in vectorstore.list_collections():
        print(name)


def main():
    parser = argparse.ArgumentParser(description="RAGForge - runtime RAG generator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a folder of documents into a named collection")
    p_ingest.add_argument("--docs", required=True, help="Path to a folder of documents")
    p_ingest.add_argument("--collection", required=True, help="Name for this document set")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a single question against a collection")
    p_ask.add_argument("--collection", required=True)
    p_ask.add_argument("--question", "-q", required=True)
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive Q&A loop against a collection")
    p_chat.add_argument("--collection", required=True)
    p_chat.add_argument("--top-k", type=int, default=5)
    p_chat.set_defaults(func=cmd_chat)

    p_list = sub.add_parser("list-collections", help="List all indexed document sets")
    p_list.set_defaults(func=cmd_list_collections)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
