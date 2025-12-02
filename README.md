<p align="center">
<!--   <img src="./imgs/bishopDL.jpg" width="120"/> -->
  <h1>Kevork Sulahian’s ML Journey</h1>
  <p>Exploring C, LLMs, evals, and everything in between.</p>
  <a href="https://github.com/KevorkSulahian/kevorksulahian-ml_journey/stargazers">
  </a>
</p>

## 📑 Table of Contents
- [About](#about)  
- [Structure](#%EF%B8%8F-structure)  
- [✅ Completed projects/code and learning](#-completed-projectscode)  
- [📝 my articles](#-my-articles)  
- [🚧 Things I'm working on right now](#-things-im-working-on-right-now)  
- [💡 Self-improvement guides](#self-improvement-guides)  
- [📚 Completed stuff that I recommend](#-completed-stuff-that-i-recommend)  
- [🏆 tiny achievements](#tiny-achievements)  
- [🔮 To do in the future](#to-do-in-the-future)  
- [✨ Maybe things](#maybe-things)  

## About
This will be my own journey of learning to be better at ML. This is meant to encourage me to learn and build more often.  
My [2d resume is here](https://kevorksulahian.github.io/2d-portfolio/)

* The [`C`](./C) folder in this directory is for the [cc4e.com](https://cc4e.com) class. More coming soon.  
* The [`Karapathy`](./Karapathy) folder is for the Karapathy learning videos.  
* And I am doing my experiments in the [`random_code`](./random_code/) folder. I will try to add more explanations once I have them—venv will probably be missing for most of them 😊.

## 🏗️ Structure
```text
kevorksulahian-ml_journey/
├── README.md
├── articles/
├── C/
├── imgs/
├── Karapathy/
├── random_code/
└── resume/
```
<details>
<summary>🔽 Expand full tree</summary>

```text
kevorksulahian-ml_journey/
├── README.md
├── articles/
│   ├── README.md
│   └── demystifying-deepseekmaths/
│       ├── deepseek_fasttext.ipynb
│       └── deepseek_fasttext.md
├── C/
│   ├── chapter_1/
│   │   ├── build/
│   │   ├── Celsius-Fahrenheit
│   │   ├── Celsius-Fahrenheit.c
│   │   ├── char_counter.c
│   │   ├── Fahrenheit-Celsius
│   │   ├── Fahrenheit-Celsius.c
│   │   ├── file_copy
│   │   ├── file_copy.c
│   │   ├── hello
│   │   ├── hello.c
│   │   ├── num_occ
│   │   └── num_occ.c
│   ├── chapter_2/
│   │   ├── 2.2.c
│   │   ├── sum_avg.c
│   │   └── to_lower.c
│   ├── chapter_3/
│   │   ├── calc.c
│   │   ├── expand.c
│   │   └── itob.c
│   ├── chapter_4/
│   │   └── excercise.c
│   ├── chapter_5/
│   │   ├── find_patern.c
│   │   ├── find_pattern.exe
│   │   ├── get_int.c
│   │   ├── get_int.exe
│   │   ├── lbs290
│   │   └── lbs290.c
│   ├── chapter_6/
│   │   ├── date.c
│   │   ├── dll.c
│   │   └── make_and_find_linkedlist.c
│   ├── chapter_7/
│   │   ├── ex1.c
│   │   └── ex2.c
│   ├── first_project/
│   │   ├── main.c
│   │   ├── main.exe
│   │   └── test.txt
│   └── function_testing/
│       ├── main.c
│       ├── math_functions.c
│       ├── math_functions.h
│       └── program
├── imgs/
│   └── bishopDL.jpg
├── Karapathy/
│   ├── GPT_from_scratch/
│   │   ├── bigram.ipynb
│   │   ├── BLT.ipynb
│   │   ├── gpt-dev.ipynb
│   │   ├── input.txt
│   │   └── token.ipynb
│   ├── gpt-2/
│   │   ├── hellaswag.py
│   │   ├── play.ipynb
│   │   └── train_gpt2.py
│   └── makemore/
│       └── 1.ipynb
├── random_code/
│   ├── a2a-samples/
│   ├── algoverse/
│   │   ├── micro_grad_code.ipynb
│   │   ├── section-1-evaluations.ipynb
│   │   └── section-2-interpretability.ipynb
│   ├── aws/
│   │   ├── amazon-bedrock-workshop/
│   │   ├── answer-matcher-gaming/
│   │   │   ├── bedrock_eval.py
│   │   │   ├── chroma_rules/
│   │   │   ├── data/
│   │   │   ├── dockerfile
│   │   │   ├── pyproject.toml
│   │   │   ├── README.md
│   │   │   ├── rulebook.py
│   │   │   └── rulebook.toml
│   │   └── aws_testing.ipynb
│   ├── chonkie/
│   │   └── first_attempt.ipynb
│   ├── crewai/
│   │   ├── agentic_sales_pipeline/
│   │   │   ├── code.ipynb
│   │   │   ├── configs/
│   │   │   └── crewai_flow.html
│   │   ├── ai_news/
│   │   │   ├── news/
│   │   │   ├── pyproject.toml
│   │   │   ├── README.md
│   │   │   ├── src/
│   │   │   └── uv.lock
│   │   ├── Dockerfile
│   │   ├── fastapi_app/
│   │   │   ├── app/
│   │   │   └── requirements.txt
│   │   ├── fastapi_crew/
│   │   │   ├── knowledge/
│   │   │   ├── pyproject.toml
│   │   │   ├── README.md
│   │   │   ├── src/
│   │   │   └── uv.lock
│   │   ├── meeting_minutes/
│   │   │   ├── pyproject.toml
│   │   │   ├── README.md
│   │   │   └── src/
│   │   ├── video_analysis/
│   │   │   ├── knowledge/
│   │   │   ├── pyproject.toml
│   │   │   ├── README.md
│   │   │   ├── src/
│   │   │   ├── uv.lock
│   │   │   ├── video.mp3
│   │   │   └── video.mp4
│   │   └── video_with_gpt/
│   │       ├── pyproject.toml
│   │       ├── README.md
│   │       ├── result.txt
│   │       └── src/
│   ├── demaned_forecast_kaggle/
│   │   └── code.ipynb
│   ├── docker/
│   │   ├── fastapi/
│   │   │   ├── compose.yaml
│   │   │   ├── Dockerfile
│   │   │   ├── main.py
│   │   │   ├── model_starter.py
│   │   │   ├── README.Docker.md
│   │   │   ├── requirements.txt
│   │   │   └── test_call.py
│   │   └── simple/
│   │       ├── Dockerfile
│   │       └── main.py
│   ├── dpsy/
│   │   └── first_try.ipynb
│   ├── fastapi/
│   │   ├── basic/
│   │   │   └── main.py
│   │   └── vercel/
│   ├── finbert sentiment analysis/
│   │   └── code.ipynb
│   ├── hf/
│   │   └── small_course/
│   │       └── chapter_1/
│   ├── lancedb/
│   │   ├── cohere_reranker.ipynb
│   │   ├── data/
│   │   └── requirements.txt
│   ├── langgraph/
│   │   ├── arxiv_paper_agent_nemo/
│   │   └── react.ipynb
│   ├── llm_eval/
│   │   ├── logprob_learnings.ipynb
│   │   └── ragas.ipynb
│   ├── LLM_from_scratch/
│   │   ├── chapter_2/
│   │   ├── chapter_3/
│   │   ├── chapter_4/
│   │   └── chapter_5/
│   ├── mcp/
│   │   ├── proj_1/
│   │   └── proj_2/
│   ├── mlz/
│   │   ├── 1/
│   │   ├── 2/
│   │   ├── 3/
│   │   └── 4/
│   ├── neofetch/
│   │   ├── main.py
│   │   ├── mine.png
│   │   └── README.md
│   ├── news_hub/
│   │   ├── app.py
│   │   ├── static/
│   │   └── templates/
│   ├── NM/
│   │   ├── code_algo_eval.ipynb
│   │   ├── code.ipynb
│   │   ├── nlm_quoraQuestions_WeightedWords_Tokenization.ipynb
│   │   └── train.csv.zip
│   ├── openai_video_process/
│   │   ├── process_video.ipynb
│   │   ├── video.mp3
│   │   └── video.mp4
│   ├── quick_codes/
│   │   ├── assistant.py
│   │   ├── quick_access.ipynb
│   │   └── testing_in_notebooks.ipynb
│   ├── rag/
│   │   └── semantic cache.ipynb
│   ├── smol_course/
│   │   ├── dpo_finetuning_example.ipynb
│   │   ├── orpo_finetuning_example.ipynb
│   │   ├── sft_peft.ipynb
│   │   └── smol_agent.ipynb
│   ├── time_series/
│   │   └── timesfm/
│   ├── unsloth/
│   │   └── llama_synthdata_finetune.ipynb
└── resume/
    └── Kevork Sulahian.pdf
```
</details>

## ✅ Completed Projects/Code and Learning
- [FinMAs](https://github.com/KevorkSulahian/agentic-llm-for-better-results)  
- [Byte Latent mini implemntation](./Karapathy/GPT_from_scratch/BLT.ipynb), mine has a static patches 4 bites each.  
- [NeoFetch](./random_code/neofetch/README.md) yes i use windows :/  
- [Karapathy minbpe](https://github.com/karpathy/minbpe) done [here](./Karapathy/GPT_from_scratch/token.ipynb)
- [Learning C cc4e](https://www.cc4e.com/) kinda finished most of the book 
- I have older stuff but I'm still too lazy to add them  

## 📝 my articles
#### Note: some of the articles are basically implementation of tutorials so not fully mine
- [Demystifying DeepSeekMath’s Data Pipeline: A FastText-Based Reproduction and Analysis](https://huggingface.co/blog/herooooooooo/demystifying-deepseekmaths-data-pipeline-a-fasttex). Simply summarized, how they are classifying the data that they captured online.  
- [Financial Analysis With Langchain and Crewai](https://huggingface.co/blog/herooooooooo/financial-analysis-with-langchain-and-crewai) some testing from langchain and crewai docs  
- [Automate Job Applications](https://huggingface.co/blog/herooooooooo/automation-job-applications-with-python-and-ollama) auto-rejection deserves auto-apply  
- [FineTune Clip](https://medium.com/@kevork.ysulahian/finetune-clip-with-huggingface-2f0abc23c57c)  
- [Zero-shot Classification](https://medium.com/@kevork.ysulahian/zero-shot-classification-and-detection-made-simple-with-huggingface-000d63d53bfe)  
- [Data Scrapping project with R](https://medium.com/@kevork.ysulahian/real-life-data-scrapping-project-scrapping-job-postings-with-r-47a6091f4866)  
- [My First C project](https://huggingface.co/blog/herooooooooo/c-first-project)  

## 🚧 Things I'm working on right now
- [FinMAs](https://github.com/KevorkSulahian/agentic-llm-for-better-results) working (slowly) on adding vision and updating the crew to work with flow/  
- [I will need to pour my life into this](https://x.com/rasbt/status/1790007260615217156)  
- [neetcode a day keeps the unemployment away](https://neetcode.io/courses)  
- Also reading ![Bishop Deep Learning](./imgs/bishopDL.jpg) it's a new theoretical reintro to the math and theory  
- [deep ml exercises](https://www.deep-ml.com/)  
- [nand game](https://nandgame.com/)  
- [puzzles](https://www.cs.cmu.edu/puzzle/)  
- [A long list of open problems and concrete projects in evals](https://docs.google.com/document/d/1gi32-HZozxVimNg5Mhvk4CvW4zq8J12rGmK_j2zxNEg/edit?pli=1&tab=t.0#heading=h.q14pjbvzx1x)  

### Self-improvement guides
- [write better code](https://marvelousmlops.substack.com/p/bridging-the-gap-converting-data)  
- [Document your grind](https://lelouch.dev/blog/august-2024/documenting-your-grind/)  
- [Reclaiming your attention and focus - Unc advice for the world](https://x.com/LukasHozda/status/1860442337534468333)  
- [Machine Learning Interviews](https://huyenchip.com/machine-learning-systems-design/toc.html)  
- [LLM (ML) Job Interviews (Fall 2024) - Process](https://mimansajaiswal.github.io/posts/llm-ml-job-interviews-fall-2024-process/)
- [How To work in with Mentors or managers](https://medium.com/@jurgens_24580/reflections-on-strategies-for-successful-meetings-with-undergraduate-researchers-ae22306ecd8d)

## 📚 Completed stuff that I recommend
- [LLama from scratch](https://blog.briankitano.com/llama-from-scratch/#:~:text=Before%20you%20even%20look%20at,and%20evaluating%20as%20you%20go) I really enjoyed how this person was implementing the paper, use it to implement future papers.  
- [Reward Hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) what could go wrong in RLHF  
- [LLM tuning and alignment series by HF](https://argilla.io/blog/mantisnlp-rlhf-part-8/) Explains different techniques  
- [What We’ve Learned From A Year of Building with LLMs](https://applied-llms.org/) Guide to building llms from exp  
- [Byte Latent Transformer: Patches Scale Better  
Than Tokens](https://arxiv.org/pdf/2412.09871) instead of a single byte, they use patches, which could scale better with images in the future  
- [High-Quality Human Data](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/) How human data is made, based on what I do daily, this is a helpful read. It's not always about the model...  
- [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)  
- [A vision researcher’s guide to some RL stuff: PPO & GRPO](https://yugeten.github.io/posts/2025/01/ppogrpo/)  
- [understanding-multimodal-llms](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)  
- I highly recommend the [A vision researcher’s guide to some RL stuff](https://yugeten.github.io/posts/2025/01/ppogrpo/) for its simplicity  
- [AI progress visualized](https://theaidigest.org/progress-and-dangers), it's a really fun read.  
- [The State of Reinforcement Learning for LLM Reasoning - 2025](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training?utm_campaign=post) would be worth to go over the papers at the end one more time.  
- [Sabotage Evaluations for Frontier Models](https://arxiv.org/pdf/2410.21514) Great paper on evals  
- [KL Divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)  
- [Momentum](https://distill.pub/2017/momentum/)  
- [BroadCasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) how python works  
- [Singular Vector Decomposition](https://gregorygundersen.com/blog/2018/12/10/svd/)  
- [GRPO Video explenation](https://youtu.be/XeUB4h1OO1g?si=UwQiG3az286G4b1c)  
- [Visual info](https://colah.github.io/posts/2015-09-Visual-Information/) pretty decent article would read again.  
- [diffusion-theory-from-scratch](https://iclr-blogposts.github.io/2024/blog/diffusion-theory-from-scratch/)  

## tiny achievements
- [HF smol course](https://github.com/huggingface/smol-course). Small but super nice intro to using HF finetuning and more.  
- [learnpytorch.io mostly following and not much else](https://www.learnpytorch.io/)  
- [hf agents](https://huggingface.co/learn/agents-course)  

## To do in the future

#### Blogs
##### Blog writers
- [ML Blogs](https://cneuralnets.netlify.app/mlblogs)  
- [hackerllama](https://osanseviero.github.io/hackerllama/blog/)  
- [eugeneyan](https://eugeneyan.com/start-here/)  

##### Just a blog
- [GNN intro](https://distill.pub/2021/gnn-intro/)  
- [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)  
- [Physics of Language Models](https://antaripasaha.notion.site/Physics-of-Language-Models-understanding-hidden-reasoning-process-1045314a563980c68566e4ecc1e32ef6)  
- [Policy Gradient](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)  
- [Flash Attention](https://benjaminwarner.dev/2023/08/16/flash-attention-compile)  
- [LLM visualized](https://bbycroft.net/llm)  
- [LLM training](https://rentry.org/llm-training)  
- [LLM Research Insights:](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction?)  
- [FineWeb: decanting the web for the finest text data at scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)  
- [torch.compile, the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.66t0x3z84jio)  
- [Build a GENAI platform](https://huyenchip.com/2024/07/25/genai-platform.html)  
- [New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training)  

## Maybe things

### Courses
- [parallel computing](https://gfxcourses.stanford.edu/cs149/fall24/courseinfo)  
- [harvard research class](https://docs.google.com/document/d/1uvAbEhbgS_M-uDMTzmOWRlYxqCkogKRXdbKYYT98ooc/edit?pli=1&tab=t.0#heading=h.o3hogvl0ayc1)  
- [ML Engineering](https://github.com/stas00/ml-engineering)  
- [Financial RL](https://github.com/AI4Finance-Foundation/FinRL/tree/c34190153d84c376dcacaf18b57097a6272b0286)  
- [Open AI RL](https://spinningup.openai.com/en/latest/)  

### Code
- [video tracking](https://github.com/roboflow/supervision)  
- [dspy rag](https://www.kaggle.com/code/iamleonie/rag-with-gemma-on-hf-and-weaviate-in-dspy)  
- [Write your Own Virtual Machine](https://www.jmeiners.com/lc3-vm/)  

- [Competitive Programmer’s Handbook](https://cses.fi/book/book.pdf)  
- [HPC](https://en.algorithmica.org/hpc/complexity/)  
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)
