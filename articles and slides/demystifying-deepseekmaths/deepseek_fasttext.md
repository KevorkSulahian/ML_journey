# Demystifying DeepSeekMath’s Data Pipeline: A FastText-Based Reproduction and Analysis

## 1. Why Data Collection Matters (and Often Gets Overlooked)

When you read most LLM papers, the emphasis falls on model architecture, parameter counts, or fine‐tuning tricks. Yet DeepSeekMath shows that where and how you gather your pretraining data in a domain like mathematics (would work in most other fields) can make or break performance. In fact, DeepSeekMath-Base 7B, after training on its carefully curated math corpus, outperformed much larger models on competition‐level benchmarks, simply by focusing on high‐quality data rather than scaling parameters alone.

Intuitively, mathematical content on the open web is rare relative to “generic” web pages. Moreover, math pages often embed formulas, LaTeX snippets, or domain‐specific jargon that simple keyword‐based filtering (“does the page contain the word ‘integral’?,” etc.) will miss or misclassify. Consequently, constructing a clean, rich, math‐focused web corpus requires:

1. **An initial seed of bona fide math pages**, so that a classifier can learn what math looks like.
2. **Iterative expansion**: find new math‐y pages that aren’t in your seed, then retrain the classifier to improve recall.
3. **Domain‐level reasoning**: identify entire websites or subpaths (e.g., `mathoverflow.net/questions`) that are math‐intensive.
4. **Deduplication and contamination‐filtering**: avoid repeatedly scraping the same content or accidentally including test‐set questions (e.g., GSM8K problems).
5. **Token‐budget management**: since “math” pages can be verbose (lots of symbols, proofs, code snippets), decide concretely how many tokens you want from each page to fill your overall 100 B+ token goal.

This multi‐stage, multi‐iteration approach is what gave DeepSeekMath its edge: by the fourth round of data collection, they amassed **120 B math‐related tokens** from **35.5 million web pages**—all rigorously filtered for true mathematical content and decontaminated of benchmark questions.


## 2. DeepSeekMath’s Data‐Collection Pipeline (2.1 of the Paper)

Let’s unpack the paper’s main “Data Collection and Decontamination” steps, focusing on the highlights you’ll want to mirror or adapt in your own code:

### Initial Seed (OpenWebMath)

DeepSeekMath begins with OpenWebMath (a curated collection of high‐quality math web text, ≈ 13.6 B tokens).

- They randomly sample 500 K math pages from OpenWebMath as positive examples.  
- For negative examples, they sample 500 K random pages from the (unfiltered) Common Crawl dump.  

These combined 1 million examples (500 K math / 500 K non‐math) train a fastText classifier with hyperparameters:  
`dim=256, lr=0.1, wordNgrams=3, minCount=3, epoch=3`.

---

### Iterative Recall of Math Pages (4 Rounds)

After training the classifier on seed data, they run it over a URL‐deduplicated Common Crawl corpus of 40 B HTML pages.  
Any page that scores above a certain threshold is tentatively labeled “math.”  

- They sort by classifier score and preserve only the top chunk (e.g., the highest‐scoring pages that together amount to 40 B tokens on the first pass).  
- **Domain‐based expansion**: They group Common Crawl pages by domain (e.g., all pages at `mathoverflow.net`) and check what fraction were already “collected” in the previous iteration.  
    - If an entire domain has > 10% of its pages labeled as math, that domain is flagged as math‐related.  
    - They then hand‐annotate the specific URL paths that truly contain math (e.g., `/questions`).  
    - Any Common Crawl URLs under those paths that weren’t previously selected are added to the seed.  

This yields a richer set of positives for retraining the classifier.  

They repeat this four times. By the end of Round 4, nearly 98% of new math pages have already been found, so they stop.  
The result is **35.5 million math pages totaling 120 B tokens**.

---

### Deduplication (URL‐level & MD5 Sketching)

- Before classification, they run URL‐based deduplication to collapse trivial redirects or mirrored pages.  
- After fetching the HTML, they do a near‐duplicate check via MD5 on the first 3,000 characters of each page’s text.  
    - If two pages share the same 3,000‐char MD5 hash, one is dropped.

---

### HTML→Plain Text

Once a page is fetched (WARC segment):  
- They strip all tags (regex `<[^>]+>` → spaces) and collapse whitespace.  
- This yields a coarse “plain text” version to feed to the classifier.

---

### Benchmark Contamination Filtering

- They remove any page that contains a 10‐gram substring appearing in GSM8K, MATH, CMATH, etc.  
    - For shorter n‐grams (≥ 3), they do exact matching.  

This ensures their pretraining data doesn’t leak test problems.

---

### Token‐Budget Selection

- Each candidate page yields an estimated token count (via their tokenizer).  
- They rank pages by classifier confidence and keep adding pages until they hit 120 B tokens (stopping when they exceed budget).  

This “greedy by confidence” method ensures the highest‐quality pages fill the budget before including lower‐confidence ones.

---

## How This Creates a “High-Quality Math Corpus”

- **Seed → Classifier → Recall** iteratively refines what “math” means to the model.  
- **Domain flags** (e.g., `mathoverflow.net`) catch entire sites that a pure page‐classifier might miss.  
- **Deduplication** ensures you don’t waste tokens on near‐identical content.  
- **N-gram filtering** removes test‐set contamination.  
- **Confidence-driven token budget** prioritizes truly math-heavy pages first.  

All together, this pipeline produces a math corpus that is:  

- **Large**: 120 B tokens.  
- **Multilingual**: Although English dominates, they also keep Chinese math pages (e.g., Gaokao problems).  
- **Clean**: No GSM8K/MATH question leaks.


## 3. Your Python Code: A Single-Pass, FastText-Based Math Scraper

Below is the Python code that implements a simplified version of the DeepSeekMath data collection pipeline using FastText for classification. This code will help you train a FastText classifier on a math dataset, scrape web pages, and filter them based on the classifier's predictions.

### 3.1. Training a FastText Classifier on a Math Dataset

```python
# 1- Install required packages
%pip install datasets -q
%pip install cdx-toolkit -q
%pip install warcio fasttext tqdm tiktoken -q
```

```python
import fasttext
import json
import itertools
import pandas as pd
import requests
import io
import re
import hashlib
import tqdm
from datasets import load_dataset
from warcio.archiveiterator import ArchiveIterator
import cdx_toolkit
import tiktoken

```

```python
# 2- Load the dataset
# https://huggingface.co/datasets/kenhktsui/math-classifiers-data
ds = load_dataset("kenhktsui/math-classifiers-data") # you should probably spend some time to understand the dataset structure
```

```python
# 3- Write out training & validation files in fastText format
with open("math.train", "w") as f:
    for example in ds['train']:

        label = example['label']
        label_str = f"__label__{label}" 
        f.write(f"{label_str} {example['text']}\n")

with open("math.valid", "w") as f:
    for example in ds['test']:

        label = example['label']
        label_str = f"__label__{label}"
        f.write(f"{label_str} {example['text']}\n")
```

```python
# 4- Train with hyperparameters matching DeepSeekMath (dim=256, lr=0.1, wordNgrams=3, minCount=3, epoch=3)
model = fasttext.train_supervised(
    input="math.train",
    lr=0.1,
    dim=256,
    wordNgrams=3,
    epoch=3,
    minCount=3,
    verbose=2
)

# 5- Save the model to disk
model.save_model("model/math-classifier.bin")
```

```python
# 6- Evaluate quickly on train/valid
math = fasttext.load_model("model/math-classifier.bin")
print("Train metrics:", math.test("math.train"))
print("Valid metrics:", math.test("math.valid"))
```

```python
print("Label for 'What is the integral of x^2 ?':", math.predict("What is the integral of x^2 ?"))
print("Label for 'in the politics of the United States, what is the role of the president?':", math.predict("in the politics of the United States, what is the role of the president?"))
```

```python
# --- 3.2: Build a small CDX index (1 000 HTML pages) in pure Python ---

import cdx_toolkit
import pandas as pd

cdx = cdx_toolkit.CDXFetcher(source="cc")
query = "commoncrawl.org/*"
print("Size estimate for query:", cdx.get_size_estimate(query))

rows = []
for obj in cdx.iter(query, limit=1000, filter=["status:200", "mime:text/html"]):
    rows.append((obj["url"], obj["filename"], obj["offset"], obj["length"]))

df = pd.DataFrame(rows, columns=["url", "warc", "offset", "length"])
df.to_csv("cc-index.csv", sep=",", index=False, header=False)

```

```python
# --- 3.3: Fetch each WARC segment, dedupe, strip HTML, classify with fastText (pure Python) ---

# 1) Load the tiny 1,000-row index
columns = ["url", "warc", "offset", "length"]
rows = pd.read_csv("cc-index.csv", sep=",", names=columns, dtype=str)



seen_url = set()
seen_sim = set()
scored = []

for _, row in tqdm.tqdm(rows.iterrows(), total=len(rows)):
    url = row["url"]
    warc_filename = row["warc"]
    offset = int(row["offset"])
    length = int(row["length"])

    warc_url = f"https://data.commoncrawl.org/{warc_filename}"
    byte_range = f"bytes={offset}-{offset + length - 1}"

    try:
        resp = requests.get(warc_url, headers={"Range": byte_range}, timeout=15)
        resp.raise_for_status()
    except Exception:
        continue

    for rec in ArchiveIterator(io.BytesIO(resp.content)):
        if rec.rec_type != "response":
            continue
        try:
            html = rec.content_stream().read().decode("utf-8", "ignore")
        except Exception:
            continue

        # 1) URL-level dedupe
        if url in seen_url:
            continue
        seen_url.add(url)

        # 2) Near-dup via MD5 on first 3,000 chars
        md5_prefix = hashlib.md5(html[:3000].encode("utf-8", "ignore")).hexdigest()
        if md5_prefix in seen_sim:
            continue
        seen_sim.add(md5_prefix)

        # 3) Strip tags → plain-ish text
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        # 4) Classify with fastText
        label, prob = math.predict(text)
        scored.append({
            "url": url,
            "text": text,
            "label": label[0],
            "score": float(prob[0])
        })

print(f"Fetched & scored {len(scored)} pages")

```

```python
# --- 3.4: Count tokens and select pages until 1 M token budget is reached ---

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
TOKENS_BUDGET = 1_000_000

for doc in scored:
    doc["ntok"] = len(enc.encode(doc["text"]))

scored.sort(key=lambda d: d["score"], reverse=True)

kept = []
total_tokens = 0
for doc in scored:
    if total_tokens + doc["ntok"] > TOKENS_BUDGET:
        break
    kept.append(doc)
    total_tokens += doc["ntok"]

print(f"Kept {len(kept)} docs → {total_tokens:,} tokens (~{total_tokens/1e6:.2f} M)")

```

```python
# --- 3.5: Write selected pages to JSONL (pure Python) ---

import json

with open("pass1-math.jsonl", "w", encoding="utf-8") as out:
    for doc in kept:
        out.write(json.dumps({
            "url": doc["url"],
            "text": doc["text"],
            "label": doc["label"],
            "score": doc["score"]
        }, ensure_ascii=False) + "\n")

print("Wrote pass1-math.jsonl 👍")

```

## 4. Recommendations for a “Full-Scale” Version

If you want to scale from 1 M tokens all the way up to 120 B tokens (and truly match DeepSeekMath), here are the concrete steps I would add:

### Pre-Fetch URL Deduplication
- Maintain a `seen_urls` set and skip any URL before requesting its WARC.

### HTML Parsing Instead of Regex
- Use `BeautifulSoup` to drop `<script>`, `<style>`, `<nav>`, `<footer>` tags, then extract `<article>` or the largest `<div>` by text length.

### Benchmark N-Gram Filtering
- Build an Aho–Corasick automaton of all 10-grams from GSM8K/MATH/CMATH. Drop pages that match any of these.

### Domain Feature & Manual Annotation
- Track `(domain_name → (#pages_seen, #pages_labelled_math))`.
- After each pass, flag domains where >10% of pages were classified as math.
- Hand-annotate URL patterns for those domains and add them into the next round’s seed.

### Iterative Classifier Retraining
- After each pass, split your “kept” pages into high-confidence positives, low-confidence candidates, and negatives.
- Label a random subset of low-confidence pages, retrain `fastText` on the expanded seed, and re-run.
- Repeat until gains diminish.

### Token Budgeted Collection (Two-Tiered)
- First estimate `approx_tokens = len(text.split())` to skip large pages when near budget.
- Then compute exact `len(tiktoken.encode(text))` only for borderline candidates.

### Sharding & Output
- When you write `pass1-math.jsonl`, split into 128 or 256 shards by `hash(url) % N`.
- Also build a CSV `index_of_shards.csv` mapping each URL to its shard & byte offset.

## 5. Final Thoughts & Conclusion

Building a 120 B-token, high-quality math corpus is nontrivial. You have to:

- **Start with a good seed**: Use OpenWebMath or a similarly curated dataset.
- **Train a strong classifier**: FastText is a good choice if speed is a priority, ensuring diversity in positives and negatives.
- **Iterate**: Recall, expand domains, retrain, and recall again—scrubbing contaminated pages at each pass.
- **Deduplicate ruthlessly**: Perform deduplication at both the URL level and via near-duplicate hashing.
- **Budget tokens**: Aim for 120 B tokens, not 10 trillion. Prioritize the most trustworthy pages.
- **Shard your final output**: Ensure your pretraining pipeline can scale effectively.

In my demo script, I implemented a mini DeepSeekMath pipeline:

- FastText → tiny CDX slice (1,000 pages) → MD5 deduplication → strip tags → classify → greedy 1 M token selection → JSONL.

This provides a taste of the mechanics.

However, to match the paper’s performance, you’d need to implement the full iterative, domain expansion, and contamination filtering steps.

DeepSeekMath demonstrated that all a 7 B model needed to solve competition-level MATH (51.7% top-1 accuracy) was good data—not 540 B parameters like Minerva. As the open-source community, we can replicate this success by carefully building domain-specific corpora for other fields.

I hope this guide helps you appreciate “data collection” as a first-class research direction—perhaps the unsung hero behind every high-performing LLM.


## References

1. Zhihong Shao et al. “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.” *arXiv:2402.03300* (27 Apr 2024).  
    [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

2. Hugging Face Dataset: Math Classifiers Data  
    [https://huggingface.co/datasets/kenhktsui/math-classifiers-data](https://huggingface.co/datasets/kenhktsui/math-classifiers-data)

```python
# convert ipynb to markdown
import nbformat
def convert_ipynb_to_md(ipynb_path, md_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                f.write(cell.source + '\n\n')
            elif cell.cell_type == 'code':
                f.write('```python\n' + cell.source + '\n```\n\n')
convert_ipynb_to_md("math-classifier.ipynb", "math-classifier.md")
```

