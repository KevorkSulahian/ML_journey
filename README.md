# Kevork Sulahian's ML Journey

This repository is my technical CV plus working lab notebook for ML/LLM engineering.
It includes finished mini-projects, experiments, course work, and a roadmap of what I want to build next.

## Quick Start (Portfolio View)

- Start here: [`docs/README.md`](./docs/README.md)
- Project map: [`docs/project-map.md`](./docs/project-map.md)
- Current focus ("now"): [`docs/now.md`](./docs/now.md)
- Roadmap / next goals: [`docs/roadmap.md`](./docs/roadmap.md)
- Writing index: [`articles/README.md`](./articles/README.md)
- Resume: [`resume/README.md`](./resume/README.md)
- 2D portfolio: [kevorksulahian.github.io/2d-portfolio](https://kevorksulahian.github.io/2d-portfolio/)

## What This Repo Shows

- Breadth: C fundamentals, ML/LLM notebooks, evals, agents, RAG, FastAPI, Docker, and small tooling projects.
- Iteration style: course-driven learning + experiments + write-ups.
- Direction: moving from exploratory notebooks into stronger evaluation, systems, and production-oriented work.

## Featured Work (Selected)

- [Gaming The Answer Matcher](https://huggingface.co/papers/2601.08849) is my AAAI Accepted Paper
- [FinMAs (external repo)](https://github.com/KevorkSulahian/agentic-llm-for-better-results): agentic financial analysis project (ongoing).
- [`random_code/aws/answer-matcher-gaming`](./random_code/aws/answer-matcher-gaming/): harness for testing how answer-matching LLM judges can be gamed by formatting/order/surface cues.
- [`random_code/neofetch`](./random_code/neofetch/README.md): a Python + `rich` neofetch-style CLI clone.
- [`random_code/news_hub`](./random_code/news_hub/): small web app experiment (templates/static/app structure).
- [`random_code/docker/fastapi`](./random_code/docker/fastapi/): Dockerized FastAPI experiments.
- [`Karapathy/GPT_from_scratch/BLT.ipynb`](./Karapathy/GPT_from_scratch/BLT.ipynb): Byte Latent Transformer-inspired notebook implementation.
- [`Karapathy/GPT_from_scratch/token.ipynb`](./Karapathy/GPT_from_scratch/token.ipynb): minBPE/tokenization learning implementation.
- [`C/`](./C/README.md): CC4E/K&R-style C practice and small projects.

## Writing

The best entry point for writing is [`articles/README.md`](./articles/README.md). Highlights:

- [Demystifying DeepSeekMath’s Data Pipeline: A FastText-Based Reproduction and Analysis](https://huggingface.co/blog/herooooooooo/demystifying-deepseekmaths-data-pipeline-a-fasttex)
- [Financial Analysis With Langchain and Crewai](https://huggingface.co/blog/herooooooooo/financial-analysis-with-langchain-and-crewai)
- [Automate Job Applications (Python + Ollama)](https://huggingface.co/blog/herooooooooo/automation-job-applications-with-python-and-ollama)
- [FineTune CLIP](https://medium.com/@kevork.ysulahian/finetune-clip-with-huggingface-2f0abc23c57c)

## Current Focus

See [`docs/now.md`](./docs/now.md) for the maintained list. Current themes:

- Agentic workflows (improving FinMAs and agent orchestration patterns)
- LLM evals and robustness (judge reliability, reward hacking, eval design)
- Systems/foundations (C, CS puzzles, nandgame, algorithms)
- Interview-ready ML engineering depth (implementation + explanation)

## Repository Map

```text
ML_journey/
├── README.md                  # portfolio landing page (this file)
├── docs/                      # navigation, roadmap, "now", archive
├── articles/                  # article index + local article drafts/notebooks
├── C/                         # C learning track (CC4E / small programs)
├── Karapathy/                 # Karpathy course notebooks + GPT experiments
├── random_code/               # experiment lab (apps, evals, agents, RAG, etc.)
├── resume/                    # PDF resume + quick links
└── imgs/                      # small image assets used in docs
```

## Folder Guides

- [`random_code/README.md`](./random_code/README.md): categorized guide to experiments (agents, evals, RAG, FastAPI, Docker, notebooks).
- [`Karapathy/README.md`](./Karapathy/README.md): learning notes index for Karpathy-based work.
- [`C/README.md`](./C/README.md): C progression by chapter/project.
- [`articles/README.md`](./articles/README.md): writing/publications index.
- [`resume/README.md`](./resume/README.md): resume links.

## Why `random_code` Still Exists

The folder name is legacy. I kept it to avoid breaking old paths and links while adding structure on top.
The new indexes in `docs/` and `random_code/README.md` are the navigation layer, and I can rename/migrate folders incrementally later.

## Current Reading / Motivation Board

This repo also tracks what I'm actively studying and what I want to build next.
For the full list, use [`docs/roadmap.md`](./docs/roadmap.md) instead of keeping everything in the root README.

![Bishop Deep Learning cover](./imgs/bishopDL.jpg)
