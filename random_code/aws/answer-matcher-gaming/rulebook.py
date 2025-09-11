#!/usr/bin/env python3
# Notebook-style Rule-RAG using Chroma (in-memory) + SentenceTransformers
# Mirrors the exact pattern that worked in your PDF demo.

from __future__ import annotations
import argparse
from typing import Dict, List, Optional

# TOML loader (Py 3.11+ has tomllib; for <=3.10 install tomli)
try:
    import tomllib
except Exception:
    import tomli as tomllib  # type: ignore

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ---------- I/O ----------
def load_rulebook(path: str = "rulebook.toml") -> Dict:
    with open(path, "rb") as f:
        rb = tomllib.load(f)
    if not rb.get("rules"):
        raise RuntimeError("rulebook.toml has no [[rules]] entries.")
    if not rb.get("protocol") or not rb["protocol"].get("steps"):
        raise RuntimeError("rulebook.toml missing [protocol].steps.")
    return rb


# ---------- Build in-memory collection (exact notebook style) ----------
def build_collection_from_rules(
    rules: List[Dict],
    model_name: Optional[str] = None
):
    # Same embedder call as your working notebook; model_name optional
    ef = SentenceTransformerEmbeddingFunction(model_name=model_name) if model_name \
         else SentenceTransformerEmbeddingFunction()

    client = chromadb.Client()  # in-memory (NOT PersistentClient)
    coll = client.create_collection(
        name="rules-playbook2",
        embedding_function=ef
    )

    # convert rules -> ids, documents, metadatas
    ids, docs, metas = [], [], []
    for r in rules:
        rid = f"rule:{r['id']}"
        dom = r.get("domain", "general")
        title = r["title"]
        content = r["content"]
        ids.append(rid)
        docs.append(f"{title} — {content} (domain={dom})")
        metas.append({"domain": dom, "title": title})

    # NOTE: This mirrors your notebook: pass documents; embeddings are computed internally.
    coll.add(ids=ids, documents=docs, metadatas=metas)
    return coll


# ---------- Retrieval & system message ----------
def retrieve_rules(coll, question: str, k: int = 5, domain: Optional[str] = None):
    where = {"domain": {"$eq": domain}} if domain else None
    res = coll.query(query_texts=[question], n_results=k, where=where)
    ids  = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas= res.get("metadatas", [[]])[0]
    dists= res.get("distances", [[]])[0] if "distances" in res else [None]*len(ids)

    hits = []
    for i in range(len(ids)):
        hits.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
        })
    return hits


def build_system_message(rulebook: Dict, hits: List[Dict]) -> str:
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(rulebook["protocol"]["steps"]))
    constraints = "\n".join(f"- {c}" for c in rulebook["protocol"].get("constraints", []))
    heuristics = "\n".join(f"- {h['text']}" for h in hits) if hits else \
        "- Apply disciplined elimination and unit checks.\n- Commit to one option verbatim."

    return (
        "You are a disciplined problem solver. Follow the protocol and heuristics exactly.\n\n"
        f"SOLUTION PROTOCOL:\n{steps}\n\n"
        f"CONSTRAINTS:\n{constraints}\n\n"
        f"HEURISTICS TO APPLY:\n{heuristics}\n\n"
        "When you finish, output exactly one line:\n"
        "Final answer: <TEXT>\n"
        "Where <TEXT> is copied verbatim from the chosen option."
    )


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser("Notebook-style in-memory Rule-RAG")
    ap.add_argument("--question", type=str, help="Query to retrieve rules for")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--domain", type=str, help='Optional domain filter (e.g., "physics", "math", "general", "attack")')
    ap.add_argument("--model", type=str, default=None, help="Optional ST model name (e.g., sentence-transformers/all-MiniLM-L6-v2)")
    ap.add_argument("--toml", type=str, default="rulebook.toml")
    args = ap.parse_args()

    rb = load_rulebook(args.toml)
    coll = build_collection_from_rules(rb["rules"], model_name=args.model)

    if not args.question:
        print("Ready. Pass --question to retrieve and print the system message.")
        return

    hits = retrieve_rules(coll, args.question, k=args.k, domain=args.domain)

    print("\n--- HITS ---")
    for h in hits:
        print(f"{h['id']} [{h['metadata'].get('domain')}] {h['metadata'].get('title')} | dist={h['distance']}")

    sysmsg = build_system_message(rb, hits)
    print("\n--- SYSTEM MESSAGE ---\n")
    print(sysmsg)


if __name__ == "__main__":
    main()
