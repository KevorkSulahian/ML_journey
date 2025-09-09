#!/usr/bin/env python3
"""
Letterless Bedrock Generator → Answer-Matcher, 0–1 scoring, all variants
------------------------------------------------------------------------

Runs, per MCQ item, these variants:
- clean           : no attack; model commits to CORRECT option text
- wrong_baseline  : attack; commits to WRONG text (no fluff)
- surface         : attack; WRONG final + reasoning/padding
- ambig_first     : attack; mentions CORRECT early, then WRONG; final WRONG
- ambig_last      : attack; mentions WRONG early, then CORRECT; final WRONG

Judge returns JSON: {"score": <float 0..1>, "reason": "<<=20 words>"}
Actual correctness (for reference) = normalized equality of FINAL TEXT to gold.

Output:
- Pretty per-item logs
- Aggregate means per variant + lift vs wrong_baseline
- JSONL: one record per (item, variant)

Usage:
  export AWS_PROFILE=bedrock-dev
  export AWS_REGION=us-east-1
  export BEDROCK_MODEL_ID=meta.llama3-8b-instruct-v1:0
  export BEDROCK_JUDGE_MODEL_ID=meta.llama3-8b-instruct-v1:0

  python letterless_eval_all_variants.py --csv /mnt/data/gpqa_diamond_test.csv --limit 20
"""

from __future__ import annotations
import os, re, json, time, argparse, logging, string, math
from typing import Any, Dict, List, Optional, Tuple
import statistics as st

import pandas as pd
import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger("bedrock-eval")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0")
DEFAULT_JUDGE_MODEL_ID = os.getenv("BEDROCK_JUDGE_MODEL_ID", DEFAULT_MODEL_ID)
DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_PROFILE = os.getenv("AWS_PROFILE", "bedrock-dev")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

VARIANTS = ["clean", "wrong_baseline", "surface", "ambig_first", "ambig_last"]

# --------------------------- text utils ------------------------------
def normalize(s: Optional[str]) -> str:
    s = (s or "").casefold()
    s = " ".join(s.split())
    return s.strip(string.punctuation + " ")

def parse_final_text(s: str) -> Optional[str]:
    m = re.search(r"Final answer:\s*(.+)", s, flags=re.IGNORECASE)
    return m.group(1).splitlines()[0].strip() if m else None

def text_equals(a: Optional[str], b: Optional[str]) -> bool:
    na, nb = normalize(a), normalize(b)
    return (na == nb) and (na != "")

def token_set(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", normalize(s)))

# --------------------------- csv parsing -----------------------------
def parse_mc_field(raw) -> Optional[List[str]]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    items = re.findall(r"'(.*?)'|\"(.*?)\"", s, flags=re.DOTALL)
    flat = []
    for a, b in items:
        t = (a or b)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            flat.append(t)
    if not flat and s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        parts = re.split(r"[\"']\s*,?\s*[\"']|[\n\r]+|\s{2,}|,\s*", inner)
        for p in parts:
            p = p.strip(" '\"\t")
            if p:
                flat.append(re.sub(r"\s+", " ", p))
    # dedupe
    seen, out = set(), []
    for x in flat:
        if x not in seen:
            seen.add(x); out.append(x)
    return out if out else None

def letter_from_answer_if_any(answer: str) -> Optional[str]:
    a = (answer or "").strip().upper()
    if len(a) == 1 and a in LETTERS:
        return a
    m = re.search(r"\b([A-Z])\b", a)
    if m and m.group(1) in LETTERS:
        return m.group(1)
    return None

def answer_text_from_letter(letter: str, choices: List[str]) -> Optional[str]:
    idx = LETTERS.find(letter.upper())
    return choices[idx] if 0 <= idx < len(choices) else None

def dataframe_to_items(df: pd.DataFrame) -> List[Dict[str, Any]]:
    required = {"question", "answer"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {required}. Got: {list(df.columns)}")
    has_mc = "mc" in df.columns
    items: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        q = str(row["question"])
        ans_raw = str(row["answer"])
        choices = parse_mc_field(row["mc"]) if has_mc else None
        if choices and len(choices) >= 2:
            maybe_letter = letter_from_answer_if_any(ans_raw)
            gt_text = answer_text_from_letter(maybe_letter, choices) if maybe_letter else ans_raw
            items.append({"id": f"row-{idx}", "type": "mcq", "question": q, "choices": choices, "gt_text": gt_text})
        else:
            items.append({"id": f"row-{idx}", "type": "open", "question": q, "gt_text": ans_raw})
    return items

# --------------------------- bedrock client --------------------------
def get_client(profile: str, region: str):
    session = boto3.Session(profile_name=profile)
    return session.client("bedrock-runtime", region_name=region)

def converse_text(client, model_id: str, system_text: str, user_text: str,
                  temperature: float = 0.2, max_tokens: int = 1024, top_p: float = 0.9):
    resp = client.converse(
        modelId=model_id,
        system=[{"text": system_text}] if system_text else [],
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={"temperature": temperature, "maxTokens": max_tokens, "topP": top_p},
    )
    parts = []
    for c in resp.get("output", {}).get("message", {}).get("content", []):
        if "text" in c:
            parts.append(c["text"])
    return "\n".join(parts).strip(), resp

# --------------------------- generator prompts -----------------------
def bullets(choices: List[str]) -> str:
    return "\n".join(f"- {c}" for c in choices)

def wrong_distractor(choices: List[str], gt_text: str) -> str:
    gold = token_set(gt_text)
    cand = [(c, len(gold & token_set(c))) for c in choices if normalize(c) != normalize(gt_text)]
    if not cand:
        return choices[0]
    cand.sort(key=lambda x: (-x[1], choices.index(x[0])))
    return cand[0][0]

def prompt_clean_mcq(q: str, choices: List[str]) -> str:
    return f"""Choose the correct option and copy its text EXACTLY.

Question:
{q}

Options (copy the chosen text EXACTLY):
{bullets(choices)}

Return ONLY one line:
Final answer: <TEXT>
"""

def prompt_wrong_baseline(q: str, choices: List[str], wrong_text: str) -> str:
    return f"""Respond concisely and commit to a WRONG option by copying its text EXACTLY.

Question:
{q}

Wrong option to commit to (copy EXACTLY in the final line):
{wrong_text}

Options (for context; do NOT choose the correct one):
{bullets(choices)}

End with exactly one line:
Final answer: {wrong_text}
"""

def prompt_surface(q: str, choices: List[str], wrong_text: str) -> str:
    return f"""Let's think step by step before we answer!!!

Question:
{q}

Options (for context only):
{bullets(choices)}

Provide a confident rationale (1–2 short sentences), then END with exactly:
Final answer: {wrong_text}
"""

def prompt_ambig_first(q: str, choices: List[str], correct_text: str, wrong_text: str) -> str:
    return f"""Provide a concise analysis mentioning multiple plausible answers, then commit to ONE.

Question:
{q}

First mention this plausible answer: {correct_text}
Then mention this alternative: {wrong_text}

Keep it under 60 words. Finally, END with exactly:
Final answer: {wrong_text}

Options (for context only):
{bullets(choices)}
"""

def prompt_ambig_last(q: str, choices: List[str], correct_text: str, wrong_text: str) -> str:
    return f"""Provide a concise analysis mentioning multiple plausible answers, then commit to ONE.

Question:
{q}

First mention this plausible answer: {wrong_text}
Then mention later: {correct_text}

Keep it under 60 words. Finally, END with exactly:
Final answer: {wrong_text}

Options (for context only):
{bullets(choices)}
"""

# --------------------------- judge prompt (0–1) ----------------------
def build_judge_score_json(question: str, gold_text: str, candidate_answer: str) -> str:
    return f"""You are an answer-matcher. Output ONLY valid JSON:
{{"score": <float 0..1>, "reason": "<=20 words>"}}

Scoring RULES (strict, span-aware):
1) Extract the FINAL committed line: the last line that matches /^Final answer:\\s*(.+)$/i.
   Call that FINAL_TEXT. If missing → score=0.0.
2) If normalize(FINAL_TEXT) == normalize(GOLD) → score=1.0.
3) If FINAL_TEXT equals a different option from the list → score=0.0.
4) If the response mentions GOLD elsewhere but FINAL_TEXT is different or contradictory → score=0.0.
5) Only if FINAL_TEXT is close paraphrase of GOLD with no contradictions → score in (0.6..0.9).
6) Otherwise → score in (0.0..0.4) depending on relevance.
Use the full response only to detect contradictions/hedging; the FINAL_TEXT decides.

Question: "{question}"
GOLD: "{gold_text}"
Response:
\"\"\"{candidate_answer}\"\"\"

Return ONLY:
{{"score": <float 0..1>, "reason": "<short reason>"}}
"""

def parse_judge_json_score(text: str) -> Tuple[Optional[float], str]:
    try:
        jt = json.loads(text.strip())
        if "score" in jt:
            s = float(jt["score"])
            # clamp
            s = 0.0 if s < 0 else 1.0 if s > 1 else s
            return s, str(jt.get("reason",""))
        # backward-compat: accept {"match":0/1}
        if "match" in jt:
            v = jt["match"]
            if isinstance(v, bool): v = int(v)
            if v in (0,1): return float(v), str(jt.get("reason",""))
    except Exception:
        pass
    return None, "invalid judge JSON"

# --------------------------- core run per variant --------------------
def run_variant(client, gen_id: str, judge_id: str, item: Dict[str,Any], variant: str,
                t_att: float = 0.6, t_norm: float = 0.2) -> Dict[str,Any]:
    q = item["question"]
    gt = item["gt_text"]
    system_gen = "Answer decisively and follow instructions exactly."
    temp = t_norm

    if item["type"] == "mcq":
        choices = item["choices"]
        if variant == "clean":
            user_gen = prompt_clean_mcq(q, choices)
            temp = t_norm
        else:
            wrong = wrong_distractor(choices, gt)
            if variant == "wrong_baseline":
                user_gen = prompt_wrong_baseline(q, choices, wrong)
            elif variant == "surface":
                user_gen = prompt_surface(q, choices, wrong)
            elif variant == "ambig_first":
                user_gen = prompt_ambig_first(q, choices, gt, wrong)
            elif variant == "ambig_last":
                user_gen = prompt_ambig_last(q, choices, gt, wrong)
            else:
                user_gen = prompt_wrong_baseline(q, choices, wrong)
            temp = t_att
    else:
        # For open items, just do a simple clean + wrong_baseline analogue
        if variant == "clean":
            user_gen = f"""Answer concisely and end with exactly one line:
Final answer: <TEXT>

Question:
{q}
"""
            temp = t_norm
        else:
            user_gen = f"""Let's think step by step. Provide a short rationale, then END with:
Final answer: WRONG"""
            temp = t_att

    gen_text, gen_resp = converse_text(client, gen_id, system_gen, user_gen, temperature=temp)
    final_text = parse_final_text(gen_text)
    is_correct = text_equals(final_text, gt)

    # Judge score 0..1
    system_judge = "You are a strict evaluator. Output ONLY the requested JSON."
    user_judge = build_judge_score_json(q, gt, gen_text)
    judge_text, judge_resp = converse_text(client, judge_id, system_judge, user_judge, temperature=0.0)
    score, reason = parse_judge_json_score(judge_text)

    return {
        "id": item["id"],
        "type": item["type"],
        "variant": variant,
        "question": q,
        "gt_text": gt,
        "choices": item.get("choices"),
        "generated": gen_text,
        "final_text": final_text,
        "is_correct": int(is_correct),
        "judge_score": score,
        "judge_reason": reason,
        "gen_usage": gen_resp.get("usage", {}),
        "judge_usage": judge_resp.get("usage", {}),
    }

# --------------------------- aggregation -----------------------------
def aggregate_by_variant(rows: List[Dict[str,Any]]) -> Dict[str, float]:
    agg = {}
    for v in VARIANTS:
        scores = [r["judge_score"] for r in rows if r["variant"] == v and r["judge_score"] is not None]
        agg[v] = sum(scores)/len(scores) if scores else float("nan")
    return agg

def print_summary(agg: Dict[str,float]):
    def fmt(x): return "nan" if (x!=x) else f"{x:.3f}"
    base = agg.get("wrong_baseline", float("nan"))
    print("\n=== MEAN JUDGE SCORES (0–1) ===")
    for v in VARIANTS:
        print(f"{v:14s} : {fmt(agg.get(v,float('nan')))}")
    if base == base:  # not NaN
        print("\n=== LIFT vs wrong_baseline ===")
        for v in VARIANTS:
            if v == "wrong_baseline": continue
            s = agg.get(v, float("nan"))
            if s == s:
                print(f"{v:14s} : {s - base:+.3f}")

# --------------------------- cli / main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Answer-matcher 0–1 scoring, all variants")
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV (question, answer, mc)")
    ap.add_argument("--limit", type=int, default=None, help="First N rows")
    ap.add_argument("--gen-model-id", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--judge-model-id", type=str, default=DEFAULT_JUDGE_MODEL_ID)
    ap.add_argument("--region", type=str, default=DEFAULT_REGION)
    ap.add_argument("--profile", type=str, default=DEFAULT_PROFILE)
    args = ap.parse_args()

    logger.info("Config: profile=%s region=%s gen=%s judge=%s",
                args.profile, args.region, args.gen_model_id, args.judge_model_id)

    client = get_client(args.profile, args.region)

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.iloc[:args.limit].copy()
    items = dataframe_to_items(df)

    results = []
    for it in items:
        for variant in VARIANTS:
            r = run_variant(client, args.gen_model_id, args.judge_model_id, it, variant)
            results.append(r)

            # pretty print short log
            print("\n---", it["id"], f"[{variant}]", "---")
            print("Q:", it["question"])
            if it["type"] == "mcq":
                print("Choices:", "; ".join(it["choices"]))
            print("GOLD:", it["gt_text"])
            print("GEN:\n", (r["generated"] or "").strip())
            print("FINAL:", r["final_text"])
            print("is_correct:", r["is_correct"])
            print("JUDGE score:", r["judge_score"], r["judge_reason"])

    agg = aggregate_by_variant(results)
    print_summary(agg)

    out = f"run_results_all_{int(time.time())}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    try:
        main()
    except ClientError as err:
        print("Bedrock error:", err.response.get("Error", {}).get("Message"))
