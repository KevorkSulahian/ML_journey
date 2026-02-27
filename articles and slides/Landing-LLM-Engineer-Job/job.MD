# How I Landed Multiple Senior LLM Engineer Offers (And The Brutal Reality of AI Interviews Right Now)

The goal of this article is simple: to share the exact strategies and interview realities I’ve experienced since November while interviewing for LLM Engineer, Reseracher and similar roles. No fluff, just what actually works right now.

## Most Roles Never Show Up on Job Boards

The job search feels especially brutal right now.

I keep hearing the same story: someone got laid off, someone is quietly interviewing, someone is tired of applying everywhere. I end up helping friends through this a lot, and after many conversations, one thing stands out:

**A lot of people are putting in serious effort on the least effective path.**

They spend hours polishing a resume, refreshing job sites, and submitting application after application. And the most common outcome is not even a rejection.

It is silence. No reply. No update. Nothing.

### The key idea

A large share of hiring happens **without** a public posting. Roles get filled through referrals, introductions, and trust that already exists. Sometimes the job never appears online at all.

So if you rely only on listings, you are:

* targeting a smaller portion of the real market
* competing with a huge crowd where your only signal is a PDF

### Four moves that work better

**1) Start with people who already know your work**
Reach out to close friends, former teammates, and past managers. Do not ask them to hire you. Let them know you are exploring and share what you are aiming for. The goal is to be top of mind when they hear about an opening.

**2) Reconnect with recruiters you have spoken to before**
If a recruiter reached out in the past, they already have context on you. Strong recruiters often learn about openings before they go public. A quick check-in can move faster than cold applying.

**3) Use your wider network**
Do not ignore lighter connections like LinkedIn contacts, online community friends, or people you met at an event. These relationships often lead to new teams and fresh opportunities. Send a friendly note and ask for a short conversation, not a favor.

**4) Get ready before you reach out**
Make it easy for someone to help you quickly:

* resume updated
* a short description of what you want next
* active presence where your industry spends time
* read about the companies values and their goals
* read their blogs and use their own words while talking

### Bottom line

Job boards are only part of the market. Your next role is more likely to come from a conversation than a form. Put more energy into relationships and make it easy for people to point you in the right direction.

---

### The Technical Screen: What They Actually Want to Hear

Once you bypass the application void and secure the interview, the technical evaluation begins. The standard for LLM Engineers has evolved rapidly. You are no longer just an engineer, with tools like Claude Code and Codex you are now an Architect, you are evaluated on *how you think*, your *problem solving skills* matters most.

A fantastic framework for this mindset is outlined in Sakana AI's [Unofficial Guide to Prepare for a Research Position](https://pub.sakana.ai/Unofficial_Guide/). Here are the core principles rephrased for applied LLM engineering:

* **Distill the Problem Space:** Don't just answer the prompt. Ask insightful questions that expose the core uncertainties of a vague problem. Figure out if you are solving the *right* problem before writing a single line of code.
* **Prototype the Riskiest Assumption:** Your goal isn't to build a bloated system right out of the gate, it's to build the absolute minimum viable experiment that tests your core hypothesis.
* **Defend Your Reasoning:** Every architectural choice you make needs a reason. "I tried X expecting Y, but observed Z" is the exact loop interviewers want to see.
* **Communicate with Low Ambiguity:** State your conclusions *first*, then explain the "why." If you don't know something, explicitly call it out. The best interviews evolve from Q&A into deep, peer-to-peer technical discussions.
* **Embrace "Good" Rabbit Holes:** If you find a shared interest with your interviewer (like a specific mathematical nuance of attention mechanisms), dive into it. Demonstrating deep curiosity sets you apart from candidates who only know high-level concepts.
* My addition to this is to turn the interview into a conversation two colleagues would have, let them feel like you are already their team member.

---

### The Deep Technical Q&A (10 Questions)

You’re right: the first version reads like “facts you should know.” Senior interviews usually probe “can you *design* this under constraints, explain the trade-offs, and prove it works.”

Here’s a deeper rewrite you can drop in. Same themes, but framed as design prompts with implementation details, failure modes, and how to validate.

---

## The Technical Screen

This round is often 60–120 minutes and it’s less about definitions and more about decisions. A strong interviewer will keep tightening constraints to see if you understand *why* certain approaches work, *where they break*, and *how you’d implement them in the real world*. If you feel like the topic is not one you are comfortable in ask for a change it's ok.

Below are 10 questions that come up a lot.

---

## 1) LoRA:

**Question:** You need to fine-tune a 70B model on a single GPU with limited VRAM. How would you implement LoRA, and where exactly does the memory saving come from?

**Answer**

* **Math + placement:** For a frozen weight (W \in \mathbb{R}^{d \times k}), add (\Delta W = BA) with (A \in \mathbb{R}^{r \times k}), (B \in \mathbb{R}^{d \times r}) (or equivalent conventions), (r \ll \min(d,k)). Forward path becomes (Wx + (BA)x). Of course this is easy to explain visually, try to explain it better in words or if you struggle like me, share your screen open up a board and write it down.
* **Why VRAM drops:** Not because the forward weights shrink, but because you avoid storing **gradients + optimizer state** (e.g., Adam’s (m,v)) for the full (W). You only pay that cost for the small adapter params.
* **Implementation decisions:** which modules get adapters (often attention proj and maybe MLP), rank selection, adapter scaling, dropout, and whether to use **QLoRA** (4-bit base weights + LoRA in higher precision).
* **Validation:** compare quality vs full finetune on a held-out set, plus check regression on general tasks to catch overfitting/forgetting.

---

## 2) DPO

**Prompt:** Why choose DPO over classic RLHF, and how would you implement it safely?

**Answer**

* **What you’re optimizing:** DPO turns preferences into a direct policy update using a loss shaped like a **logistic classification** on the difference of log-likelihoods:
  [
  \mathcal{L} = -\log \sigma\left(\beta \left[(\log \pi_\theta(y^+|x)-\log \pi_\theta(y^-|x))-(\log \pi_{\text{ref}}(y^+|x)-\log \pi_{\text{ref}}(y^-|x))\right]\right)
  ]
  with (\pi_{\text{ref}}) as a fixed reference policy and (\beta) controlling how “aggressive” updates are.
* **Why it’s attractive:** fewer moving parts than reward model + PPO, typically easier to stabilize and reproduce.
* **Practical implementation:** data cleaning for preference pairs, length bias handling, batching with shared prefix (x), careful choice of (\beta), and monitoring KL drift from the reference to prevent “personality collapse.”
* **When not to use it:** weak or noisy preferences, distribution shift, or tasks needing exploration-like behavior.
* **Validation:** offline win-rate on preference evals, plus online guardrails (toxicity, refusal accuracy, policy compliance).

---

## 3) Domain fine-tuning without breaking the base model

**Prompt:** You must adapt a general assistant to a narrow domain. How do you reduce catastrophic forgetting while still improving domain performance?

**Answer**

* **Primary lever:** **rehearsal/data mixing** (keep a slice of high-quality general instruction data) so the model continues seeing “general” behaviors during updates.
* **Regularization options:** KL penalty toward base model outputs on a general set, or “don’t move too far” constraints (conceptually similar to anchoring the behavior).
* **Parameter strategy:** adapters (LoRA) to localize changes; optionally unfreeze only top layers if needed.
* **Training plan:** staged approach: (1) adapter warm-up on domain, (2) mixed training, (3) short alignment pass for tone/safety.
* **How you know it’s working:** domain task metrics go up *and* general evals stay flat; also measure refusal regressions, instruction-following, and hallucination rates.

---

## 4) RAG

**Prompt:** Your RAG system retrieves good info, yet accuracy drops when context gets long. How do you redesign prompt assembly and retrieval to address “lost in the middle”?

**Answer**

* **Root cause framing:** attention and salience biases lead to better recall near the start/end of the context window.
* **Implementation moves:**

  * **Re-ranking** (cross-encoder or LLM re-ranker) to ensure the best chunks are truly best.
  * **Context packing strategy:** place highest-value evidence early, then a structured “evidence block” late (for recency), rather than dumping chunks in score order.
  * **Compression:** query-focused summarization of lower-ranked chunks so they do not consume tokens.
  * **Chunking fixes:** overlap, section-aware chunking, and deduplication to avoid repeating near-identical passages.
* **Validation:** per-question ablations: score-order vs packed-order vs compressed; track citation coverage and faithfulness.

---

## 5) Quantization

**Prompt:** You need INT4 serving with minimal quality loss. How do you pick between GPTQ and AWQ, and what does a correct calibration/eval look like?

**Answer**

* **Decision criteria:** target hardware, tolerance for quality drop, model architecture quirks (outliers), and latency goals.
* **Core difference:** GPTQ typically uses second-order approximations to minimize reconstruction error; AWQ uses activation statistics to protect “important” weights.
* **Implementation details that matter:** calibration dataset representativeness, group size, per-channel scaling, and outlier handling.
* **Evaluation beyond perplexity:** task-level checks (reasoning, format-following, domain QA), plus latency and memory profiling.
* **Risk callout:** quantization can silently break tool-calling or structured outputs even when perplexity barely moves.

---

## 6) KV cache and PagedAttention

**Prompt:** Your throughput collapses at high concurrency due to KV cache pressure. Explain KV caching and how you would implement a paged/block-based cache manager.

**Answer**

* **What KV cache stores:** keys/values per layer per token, making decode faster but memory-hungry.
* **Why naive caching hurts:** preallocating contiguous buffers for max sequence lengths causes fragmentation and waste.
* **Paged design:** allocate fixed-size blocks, map sequences to blocks, and reuse blocks across batches with an allocator. This enables continuous batching and reduces wasted VRAM.
* **Engineering concerns:** eviction policy for long-lived sessions, block size trade-offs, and how you prevent race conditions in multi-request scheduling.
* **Metrics:** cache hit rate, effective utilization, tokens/sec at p95 latency.

---

## 7) RoPE and long context

**Prompt:** You inherited a 4k-context RoPE model but product requires 32k. What’s your approach, and what breaks if you do it wrong?

**Answer**

* **Mechanism:** RoPE rotates representations with position-dependent angles; pushing to unseen positions can degrade attention patterns.
* **Extension approaches:** position index scaling (position interpolation), or other rescaling strategies, followed by **targeted long-context fine-tuning** on synthetic + real long sequences.
* **Training recipe:** mix short and long contexts, emphasize retrieval-style tasks, and monitor for “attention drift” (model ignoring early evidence).
* **Validation:** long-context needle tests, multi-hop QA, and regression checks on original 4k tasks.

---

## 8) RAG evaluation

**Prompt:** How would you evaluate a RAG system so you can ship improvements confidently and catch regressions?

**Answer**

* **Separate the pipeline:** retrieval quality, grounding/faithfulness, and final answer usefulness are distinct failure modes.
* **Offline set-up:** curated eval set with query, expected evidence, and failure tags; log retrieved chunks and generation.
* **Measurements that map to reality:**

  * retrieval: Recall@k, MRR, NDCG (plus “evidence coverage” if you can label it)
  * grounding: citation alignment checks, quote-based verification, or judge models with strict prompting
  * usefulness: task success scoring and human review for edge cases
* **Online:** A/B tests, escalation rate, user satisfaction, and hallucination reports with replayable traces.
* **Anti-pattern:** “We just use ROUGE/BLEU” for open-ended QA.
* **Offline:** We could use answer-matcher technique on an Golden Set.

---

## 9) Inference performance

**Prompt:** Your service must handle 1,000+ concurrent users with tight p95 latency. Walk through the bottlenecks and what you would do first.

**Answer**

* **Phase split:** prefill tends to be compute-heavy; decode is usually bandwidth/latency sensitive because it is token-by-token.
* **High-impact levers:** FlashAttention-style kernels for prefill, continuous batching, KV cache optimizations, and speculative decoding when appropriate.
* **System design:** request scheduling, max tokens policy, streaming strategy, and admission control to prevent tail latency blow-ups.
* **Proof it worked:** before/after profiling, tokens/sec, GPU utilization, p95/p99 latency, and quality checks (since quantization and speculation can degrade output).

---

## 10) Multi-agent robustness

**Prompt:** You built an autonomous multi-agent workflow (planner, executor, verifier). How do you test reliability and prevent runaway behavior in production?

**Answer**

* **Failure taxonomy:** loops, compounding errors, tool misuse, and “confident nonsense” propagation between agents.
* **Hard constraints:** step budgets, token budgets, timeouts, tool permissions, and circuit breakers (stop conditions).
* **Detection:** cycle detection on state, semantic similarity on repeated messages, and anomaly triggers for repeated tool calls.
* **Observability:** per-step traces, tool payload logs, and outcome labels to enable postmortems and targeted fixes.
* **Fallbacks:** degrade to a simpler single-agent mode, ask clarifying questions, or hand off to a human when risk is high.

---

### The Coding Round

Here was the most surprising part of my interview loops: **Companies actively encourage you to use AI coding assistants.** Whether it's Codex, Cursor, or Claude Code, interviewers no longer care if you can memorize boilerplate. They care about *how* you use these tools. You are heavily judged on your prompting architecture, how fast you navigate edge cases, and your ability to verify AI-generated code against reality.

Usually, they will ask you to implement one of the deep technical concepts discussed earlier. For example, you won't be asked to invert a binary tree (Thankfully), you will be asked to implement write a mock implementation of PagedAttention block allocation, or set up a custom loss function for an RLHF pipeline, or implement an AI assistant, I've been asked to make a RAG system 3 times already. Failing to catch the errors of your LLM will likely cost you the offer, don't use the cheaper ones and good luck!