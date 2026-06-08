# Measuring Pragmatic Alignment in LLM-Based Agents
**Research Case Study · University of Trier · NLP Master's Program · WS 2025/26**
> Can LLMs generate replies that don't just *look* like human discourse — but *function* like it?
---
## Overview
**This project investigates **pragmatic alignment** in LLM-generated text: the degree to which a language model's output performs the same *communicative function* as an authentic human reply, not just mimicking its surface form.
Standard text similarity metrics (BLEU, BERTScore, sentiment) measure *style*. They tell you whether the model sounds human. This project operationalizes a different question: **does the model perform the same social act?**
The pipeline covers the full research workflow — corpus curation, exploratory analysis, a novel multi-dimensional annotation scheme, and a pragmatics-aware generation loop using LLM + XGBoost critics.**
---
## The Core Problem: The Mimicry Gap
We analyzed 1,000 English political discourse samples from X (Twitter), comparing **authentic human replies** against **Qwen3 8B fine-tuned replies** for the same target tweet.
Initial EDA revealed a critical split:
| Dimension | Finding |
|---|---|
| Style similarity | **High** — sentiment distribution and POS frequency matched closely |
| Classification accuracy | **53%** — barely above chance; a standard classifier couldn't tell them apart |
| Lexical overlap (Jaccard) | **~2%** — the model uses completely different words |
| Semantic similarity (SBERT) | **~45%** — moderate, but not functional equivalence |
**Conclusion:** The model mimics the *form* of discourse but substitutes the *substance*. This is the Mimicry Gap — and it cannot be captured by traditional metrics alone.
---
## Annotation Scheme
To measure functional equivalence, we developed a **5-dimension pragmatic annotation scheme**:
| Dimension | Labels | What it captures |
|---|---|---|
| **Stance** | Support / Contest | The political position |
| **Action** | Statement / Question / Command | The structural speech act |
| **Personalness** | Personal Experience / General Fact | How the claim is grounded |
| **Sarcasm** | Sarcastic / Serious | Pragmatic tone |
| **Politeness** | Rude / Polite | The safety/register register |
See [`docs/Annotation_Guidelines.pdf`](docs/Annotation_Guidelines.pdf) for full inter-annotator agreement protocol and label definitions.
---
## Pragmatics-Aware Generation Pipeline
The core technical contribution is an **RLHF-inspired agentic generation loop** that steers LLM output toward functional alignment:
```
Tweet + Gold Labels
       ↓
LLaMA (via Groq API) → candidate reply
       ↓
(tweet + reply) → SBERT embeddings (384-d)
       ↓
4 × XGBoost Critics [STANCE | ACTION | PERSONALNESS | POLITENESS]
       ↓
Probability-weighted alignment score
       ↓
Score ≥ 0.75 → ACCEPT
Score < 0.75 → tighten prompt → regenerate
```
### Scoring Logic
Each XGBoost critic outputs a probability for the gold label class. The overall alignment score is the weighted mean:
```python
score = mean([p_stance, p_action, p_personalness, p_politeness])
# Threshold: score >= 0.75 → accept | else → rewrite with tighter constraints
```
---
## Repository Structure
```
├── data/
│   ├── raw/                  # Original Twitter corpus (1,000 samples)
│   └── processed/            # Annotated dataset with pragmatic labels
│
├── notebooks/
│   ├── 1_eda.ipynb           # Corpus analysis + Mimicry Gap findings
│   ├── 2_classification.ipynb # Dataset prep + label inspection
│   ├── 3_label_encoders.ipynb # LabelEncoder fitting + serialization
│   ├── 4_rag_baseline.ipynb  # RAG-based reply generation baseline (Phi-2)
│   └── 5_agentic_pipeline.ipynb  # Full generation + critic feedback loop
│
├── docs/
│   └── Annotation_Guidelines.pdf
│
└── README.md
```
---
## Setup
```bash
git clone https://github.com/faizan10152/Measuring-Pragmatic-Alignment-in-LLM-Based-Agents.git
cd Measuring-Pragmatic-Alignment-in-LLM-Based-Agents
pip install -r requirements.txt
```
**Required API keys** (set in `.env`):
```
GROQ_API_KEY=your_groq_key
```
Run notebooks in order (1 → 5). Each notebook's output feeds the next.
---
## Key Results
- A standard binary classifier achieved only **53% accuracy** distinguishing authentic vs. LLM-generated political replies — confirming the style-level mimicry
- XGBoost critics trained on SBERT embeddings achieved **[insert F1 scores per dimension]** across the four pragmatic dimensions
- The generation loop improved alignment scores on **[X%]** of initially-rejected replies through prompt re-steering
---
## Tech Stack
`Python` · `Qwen3 8B` · `LLaMA (Groq)` · `XGBoost` · `SBERT / sentence-transformers` · `scikit-learn` · `pandas` · `Jupyter`
---
## Context
This is a research case study completed as part of the **Master's in Natural Language Processing** at the University of Trier (WS 2025/26). The project was completed collaboratively; my contributions focused on the agentic pipeline design, XGBoost critic training, and the scoring/threshold framework.
---
*Pragmatic alignment is about more than sounding human — it's about acting human.*
