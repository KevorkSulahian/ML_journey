# rules_kb.py
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple

# TOML loader: tomllib (3.11+) or tomli (<=3.10)
try:
    import tomllib
except Exception:
    import tomli as tomllib  # type: ignore

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# quiet things down a bit on Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CHROMADB_TELEMETRY_IMPLEMENTATION", "none")
os.environ.setdefault("CHROMADB_ALLOW_HNSWLIB_FALLBACK", "true")


def load_rulebook(path: str) -> Dict:
    with open(path, "rb") as f:
        rb = tomllib.load(f)
    if not rb.get("rules"):
        raise RuntimeError("rulebook has no [[rules]] entries.")
    if not (rb.get("protocol") and rb["protocol"].get("steps")):
        raise RuntimeError("rulebook missing [protocol].steps.")
    return rb


class RulesKB:
    """
    In-memory Chroma store with explicit SentenceTransformer embeddings.
    Build once, reuse for all queries.
    """
    def __init__(self, rulebook_path: str,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.rulebook = load_rulebook(rulebook_path)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.client = chromadb.Client()  # in-memory
        self.collection = self._build_collection(self.rulebook["rules"])

    def _build_collection(self, rules: List[Dict]):
        ids, docs, metas = [], [], []
        for r in rules:
            rid = f"rule:{r['id']}"
            dom = r.get("domain", "general")
            title, content = r["title"], r["content"]
            ids.append(rid)
            docs.append(f"{title} — {content} (domain={dom})")
            metas.append({"domain": dom, "title": title})
        # embed all docs explicitly
        embs = self.model.encode(
            docs, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        if not isinstance(embs, np.ndarray) or embs.ndim != 2:
            raise RuntimeError("bad embeddings shape")
        coll = self.client.create_collection(name="rules-playbook", embedding_function=None)
        coll.add(ids=ids, embeddings=embs.tolist(), metadatas=metas, documents=docs)
        return coll

    def retrieve(self, question: str, k: int = 5, domain: Optional[str] = None) -> List[Dict]:
        q_emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        where = {"domain": {"$eq": domain}} if domain else None
        res = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=k, where=where)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(ids)
        hits = []
        for i in range(len(ids)):
            hits.append({"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]})
        return hits

    def build_system_message(self, question: str, k: int = 5, domain: Optional[str] = None) -> str:
        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.rulebook["protocol"]["steps"]))
        constraints = "\n".join(f"- {c}" for c in self.rulebook["protocol"].get("constraints", []))
        hits = self.retrieve(question, k=k, domain=domain)
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
