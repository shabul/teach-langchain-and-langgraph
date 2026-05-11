"""
Local Private Doc/Code Chat — CLI entry point.

Usage:
  # First run — index the repo and start chatting
  ./start.sh --dir ../.. --reindex

  # Subsequent runs — load existing index, resume session
  ./start.sh --dir ../..

  # Custom session and directory
  ./start.sh --dir ~/my-notes --session notes --reindex

Commands during chat:
  /quit or /exit  — exit
  /sources        — show chunk count in index
  /session        — show current session/thread ID
  /model          — show which LLM is active
"""
from __future__ import annotations

import argparse
import os
import sys

import requests
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from graph import make_graph
from indexer import index_directory, load_vectorstore
from lm_studio_chat import ChatLMStudio


# ── LLM detection ─────────────────────────────────────────────────────────────

def _lm_studio_up(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/api/v1/models", timeout=2)
        return r.ok
    except Exception:
        return False


def get_llm(args: argparse.Namespace) -> tuple[BaseChatModel, str]:
    """
    Return (llm, description) using the best available backend:
      1. LM Studio (local, no key needed)
      2. OpenAI  (OPENAI_API_KEY in .env)
      3. Google  (GOOGLE_API_KEY in .env  →  gemini-2.0-flash)
    Exits with a clear message if none are available.
    """
    # ── 1. Try local model ──
    if _lm_studio_up(args.base_url):
        llm = ChatLMStudio(model=args.model, base_url=args.base_url)
        return llm, f"{args.model} (local via LM Studio @ {args.base_url})"

    # ── 2. Load .env for cloud keys ──
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    google_key  = os.getenv("GOOGLE_API_KEY", "").strip()

    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", api_key=openai_key), "gpt-4o-mini (OpenAI cloud)"

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return (
            ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_key),
            "gemini-2.0-flash (Google cloud)",
        )

    # ── 3. Nothing available ──
    print("\n[error] No LLM available.")
    print("  LM Studio is not running AND no API key found in .env")
    print()
    print("  Options:")
    print("   A) Start LM Studio, load a model, enable the REST API server — then rerun.")
    print("   B) Add a key to .env:")
    print("        GOOGLE_API_KEY=AIza...   (get it from aistudio.google.com)")
    print("        OPENAI_API_KEY=sk-...    (optional alternative)")
    print("      Then rerun:  ./start.sh --dir <your-dir>")
    sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chat with your local docs/code via LangGraph RAG"
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

    print(f"\n{'='*55}")
    print("  Local Doc Chat  |  LangGraph + Chroma RAG")
    print(f"{'='*55}")

    # --- Build index ---
    try:
        vectorstore = index_directory(args.dir, force_reindex=args.reindex)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}")
        sys.exit(1)

    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})

    # --- Pick LLM (local or cloud fallback) ---
    llm, llm_label = get_llm(args)
    app = make_graph(retriever, llm)

    config = {"configurable": {"thread_id": args.session}}

    # --- Banner ---
    print(f"\nModel  : {llm_label}")
    print(f"Index  : {args.dir}  (k={args.k})")
    print(f"Session: '{args.session}'")
    print("\nType your question. Commands: /quit  /sources  /session  /model")
    print("-" * 55)

    # --- Chat loop ---
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

        if user_input == "/model":
            print(f"[info] model = {llm_label}")
            continue

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            print(f"\nAssistant: {result['messages'][-1].content}")
        except Exception as e:
            print(f"\n[error] {e}")


if __name__ == "__main__":
    main()
