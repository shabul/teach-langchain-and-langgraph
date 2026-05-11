"""
Local Private Doc/Code Chat — CLI entry point.

Usage:
  # First run — index the repo and start chatting
  python main.py --dir ../.. --reindex

  # Subsequent runs — load existing index, resume session
  python main.py --dir ../..

  # Different directory, named session
  python main.py --dir ~/my-notes --session notes --reindex

  # Point at a single project folder
  python main.py --dir /path/to/my-project --session myproject

Commands during chat:
  /quit or /exit  — exit
  /sources        — show indexed source count
  /session        — show current session/thread ID
"""
from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage

from graph import make_graph
from indexer import index_directory, load_vectorstore
from lm_studio_chat import ChatLMStudio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chat with your local docs/code via LM Studio + LangGraph RAG"
    )
    p.add_argument("--dir", required=True, help="Directory to index")
    p.add_argument("--session", default="default", help="Session/thread ID (for checkpointing)")
    p.add_argument("--reindex", action="store_true", help="Force re-embedding of all files")
    p.add_argument("--model", default="openai/gpt-oss-20b", help="LM Studio model name")
    p.add_argument("--base-url", default="http://localhost:1234", help="LM Studio base URL")
    p.add_argument("-k", type=int, default=3, help="Number of chunks to retrieve per query")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Build index ---
    print(f"\n{'='*55}")
    print("  Local Doc Chat  |  LM Studio + LangGraph + Chroma")
    print(f"{'='*55}")

    try:
        vectorstore = index_directory(args.dir, force_reindex=args.reindex)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)

    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})

    # --- Build LLM + graph ---
    llm = ChatLMStudio(model=args.model, base_url=args.base_url)
    app = make_graph(retriever, llm)

    config = {"configurable": {"thread_id": args.session}}

    # --- Chat loop ---
    print(f"\nModel : {args.model} @ {args.base_url}")
    print(f"Index : {args.dir}  (k={args.k})")
    print(f"Session: '{args.session}'  (history saved in-process via MemorySaver)")
    print("\nType your question. Commands: /quit  /sources  /session")
    print("-" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye!")
            break

        if user_input == "/sources":
            count = vectorstore._collection.count()
            print(f"[info] {count} chunks in index.")
            continue

        if user_input == "/session":
            print(f"[info] session = '{args.session}'")
            continue

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            answer = result["messages"][-1].content
            print(f"\nAssistant: {answer}")
        except Exception as e:
            print(f"\n[error] {e}")
            print("Is LM Studio running? Check that the server is active on", args.base_url)


if __name__ == "__main__":
    main()
