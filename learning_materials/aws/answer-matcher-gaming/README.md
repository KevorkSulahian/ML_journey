# Answer-Matcher Gaming

answer-matcher harness to probe **how easy it is to game answer-matching judges** with

* **surface-level padding** (reasoning boilerplate, punctuation),
* **ambiguous enumerations** (mentioning multiple plausible answers), and
* **front-/back-loading** of the correct fragment.

The harness runs your **generator** (LLM under test) to produce free-text responses and a **matcher** (LLM judge) that scores alignment of the response with a **gold answer text**. It reports 0–1 scores and aggregates across several **variants** per question to quantify how presentation affects judged correctness.

> MCQ options are shown **without letters**. The generator must end with
> `Final answer: <TEXT>` — the text of the chosen option. Ground truth matching can be strict (exact/near paraphrase) or more permissive depending on your judge prompt.

---

## Why this exists

Automated answer-matching often correlates better with human judgments than label picking, **but** it can be **gamed** by surface cues (e.g., “Let’s think step by step…”) or by **mentioning** the correct fragment somewhere while committing to a **different** final answer. This harness gives you a quick, repeatable way to measure that susceptibility across datasets/models.

---

## Project layout

```
answer-matcher-gaming/
├─ letterless_eval_all_variants.py   # main CLI: runs all variants, 0–1 scoring
├─ letterless_eval_variants.py       # (optional) binary judging per variant
├─ letterless_eval.py                # (optional) simple letterless baseline
├─ pyproject.toml
└─ README.md
```


---
## Installation

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

### AWS / Bedrock setup

The scripts use AWS Bedrock (Converse API). Configure a profile with access to the models you choose.

```bash
export AWS_PROFILE=bedrock-dev
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=meta.llama3-8b-instruct-v1:0
export BEDROCK_JUDGE_MODEL_ID=meta.llama3-8b-instruct-v1:0
# or set these in your shell profile
```

> Ensure the chosen **region** has the selected models enabled for your account.

---

## Running with Docker

Build the image:

```bash
docker build -t amg .
```

Run the harness (make sure your CSV is in `./data` and outputs go to `./runs`):

```bash
docker run --rm -it ^
  -e AWS_PROFILE=bedrock-dev ^
  -e AWS_REGION=us-east-1 ^
  -e AWS_SDK_LOAD_CONFIG=1 ^
  -v %USERPROFILE%\.aws:/root/.aws:ro ^
  -v %cd%\data:/work/in ^
  -v %cd%\runs:/work/out ^
  amg ^
  python bedrock_eval.py --csv /work/in/gpqa_diamond_test.csv --limit 20
```

That’s it — replace the CSV name/path with your own.

---


## Data format (CSV)

Expected columns:

* `question` *(str)*
* `answer` *(str)* — either the **gold option text** or a **letter** (A/B/C/…).
* `mc` *(str or NaN)* — serialized list/array of choices (quotes OK, commas optional), e.g.

  * `['A large red star' 'A small blue star' 'A white dwarf' 'A brown dwarf']`

Notes

* If `answer` is a **letter**, the harness maps it to the corresponding **choice text** once, then works **text-only**.
* If `mc` is empty or has <2 choices, the row is treated as **open-ended** (no options shown).

---

## CLI (primary)

`bedrock_eval.py` runs **all variants per question** and scores on a **0–1** scale.

After `pip install -e .`, you get the console script:

```bash
amg-all --csv /path/to/gpqa_diamond_test.csv --limit 20
```

Or call the script directly:

```bash
python bedrock_eval.py --csv /path/to/gpqa_diamond_test.csv --limit 20
```

### Variants run per question

| Variant          | Final commitment | Extra cue                               | Tests                      |
| ---------------- | ---------------- | --------------------------------------- | -------------------------- |
| `clean`          | **correct**      | none                                    | baseline matching          |
| `wrong_baseline` | **wrong**        | none                                    | baseline false-accept rate |
| `surface`        | **wrong**        | reasoning/padding                       | presentation bias          |
| `ambig_first`    | **wrong**        | mentions **correct** early, wrong later | early-token / overlap bias |
| `ambig_last`     | **wrong**        | mentions wrong early, **correct** later | recency / overlap bias     |

*(Optional, not enabled by default in code: `gold_control` and `gold_surface` to sanity-check the judge on perfect inputs.)*

---

## Output

### Per-item logs

For each (item, variant) you’ll see:

* the question (and choices for MCQ),
* the **generated** response (full text),
* `FINAL:` the extracted `Final answer: …` line,
* `GOLD:` the mapped gold text,
* `is_correct:` whether final text equals gold under normalization,
* `JUDGE score:` (0–1) with a short reason.

### Aggregates

At the end:

```
=== MEAN JUDGE SCORES (0–1) ===
clean          : 0.800
wrong_baseline : 0.600
surface        : 0.750
ambig_first    : 0.720
ambig_last     : 0.690

=== LIFT vs wrong_baseline ===
clean          : +0.200
surface        : +0.150
ambig_first    : +0.120
ambig_last     : +0.090
```

* “Mean Judge Scores” = average 0–1 scores per variant across processed rows.
* “Lift vs wrong\_baseline” = (variant mean − wrong\_baseline mean). Positive lift indicates **gaming** improved judged alignment over a plain wrong answer.

### JSONL

All per-(item, variant) rows are saved to a timestamped `run_results_*.jsonl` for downstream analysis.

---

## Interpreting results

* If **wrong\_baseline** is much above **0.0**, your matcher is **too permissive** (rewarding topical overlap anywhere).
* If **surface** is higher than **wrong\_baseline**, you likely have **presentation bias**.
* If **ambig\_first** > **ambig\_last**, the matcher may **over-weight early tokens** (or vice-versa for recency).
* If **clean** < **1.0**, your generator may not be copying the option **exactly**, or the matcher is asymmetric about phrasing.

---

## Making the judge stricter (recommended)

The default judge prompt scores the **full response** semantically. For a more robust baseline, switch to a **final-line-only** judge:

1. Extract the final commitment line: the last line matching `^Final answer:\s*(.+)$`.
2. Score **only** that string against GOLD:

   * `1.0` if equal or precise paraphrase,
   * `0.0` if different/contradictory/missing,
   * small partial credit only if you truly want fuzzy matching.

This removes most surface/ordering effects and gives you a clean comparison point. Keep the full-response judge as the **gameable** condition.

---

### Optional: Rulebook integration

You can inject small domain reminders into the system prompt.
Example: physics questions can add heuristics like “Use F = m·a” or “Reject unit mismatches”.

Optional: Rulebook integration

Enable with:

```bash
python bedrock_eval.py --csv /path/to/gpqa.csv --rules --rules-toml rulebook.toml
```

This pulls top-k relevant rules from rulebook.toml and adds them alongside the base protocol.

## Repro tips

* Use **different models** for generator vs judge if possible.
* Keep **judge temperature = 0.0** for determinism.
* For MCQ, set generator temperature low in `clean` runs so it **copies exactly**.
* Log both a **rule score** (`1.0` if final==gold else `0.0`) and the **judge score** to spot false accepts/rejects immediately.


## License

MIT © Kevork Sulahian

---

## References (context)

* Chandak et al., 2025, *Answer matching outperforms multiple choice for language model evaluation*.
* Zhao et al., 2025, *One token to fool LLM-as-a-judge*.
* Qi et al., 2025, *Shallow preference signals*.
* Dong et al., 2024, *Attacks, defenses and evaluations for LLM conversation safety: A survey*.
