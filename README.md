Measuring Pragmatic Alignment in LLM-Based Agents

Research Case Study | University of Trier | Winter Semester 2025/26

This repository contains the code, data, and documentation for our research on measuring the paragmatic alignment in LLM-based agents. This project investigates whether Large Language Models (LLMs) can simulate human social behavior not just in style, but in pragmatic function.

Project Overview

The Problem: LLMs are trained to be helpful and harmless (RLHF). While they can generate text that looks like a human tweet (correct grammar, similar sentiment), we hypothesize that they fail to perform the same "communicative work" in adversarial or political contexts.

The Goal: To operationalize and measure "Functional Equivalence" does the synthetic reply perform the same social move (e.g., sarcasm, personal attack, agreement) as the authentic human reply?

The Dataset:

Source: 1,000 samples of English political discourse from X (Twitter).

Comparison: For each target tweet, we compare an Authentic Human Reply vs. a Fine-Tuned Qwen3 8B Model Reply.

Key Findings (Exploratory Data Analysis)

Our initial analysis (see /notebooks/deliverable_1_eda.ipynb) revealed a critical "Mimicry Gap":

Style Match (High): The LLM matches human Sentiment distribution and Part-of-Speech frequency almost perfectly. A standard classifier could only distinguish them with 53% accuracy (barely better than random chance).

Content Mismatch (High): The Lexical Overlap (Jaccard) is extremely low (~2%), and Semantic Similarity is only moderate (~45%).

Conclusion: The model mimics the form of the discourse but changes the substance. This necessitates a multi-dimensional annotation scheme to capture why they differ.

The Annotation Scheme

To measure this gap, we developed a novel 5-dimension annotation scheme (see /docs/Annotation_Guidelines.pdf):

Stance: Support vs. Contest (The Politics)

Action: Statement vs. Question vs. Command (The Structure)

Personalness: Personal Experience vs. General Fact (The Grounding)

Sarcasm: Sarcastic vs. Serious (The Vibe)

Politeness: Rude vs. Polite (The Safety Filter)


Pragmatics-Aware Reply Generation with LLaMA and XGBoost Critics

This repository implements a pragmatics-aware text generation pipeline inspired by RLHF-style ideas.
A large language model (LLaMA via Groq) generates replies to tweets, which are then evaluated by four trained XGBoost critic models (STANCE, ACTION, PERSONALNESS, POLITENESS).
Based on the critic feedback, the reply is either accepted or rewritten using a prompt-steering loop.
It uses reward-guided generation and prompt adaptation, inspired by RLHF principles.



Dataset Tweet + Gold Labels
        ↓
LLaMA (Groq) generates reply
        ↓
(tweet + reply) → SBERT embeddings
        ↓
XGBoost Critics (4 heads)
        ↓
Probability-based score / loss
        ↓
If score ≥ threshold → ACCEPT
Else → tighten prompt → regenerate

📓 Notebook Descriptions (Execution Order Matters)
1️⃣ Classification.ipynb

Purpose:
Prepares and analyzes the annotated dataset used for training and evaluation.

What it does:

Loads the annotated dataset (CSV/Excel)

Inspects pragmatic labels:

STANCE

ACTION

PERSONALNESS

POLITENESS

Ensures data consistency and cleanliness

Output:
A clean dataset used consistently across all later notebooks.

2️⃣ Label_Encodersipynb.ipynb

Purpose:
Creates and saves LabelEncoders for each pragmatic dimension.

Why this is critical:
The XGBoost models are trained on numeric class IDs, but the dataset contains string labels (e.g., SUPPORT, ASK).

What it does:

Fits a LabelEncoder for each label column

Prints class-to-ID mappings

Saves encoders as:
label_encoders.pkl

Output used by:

Agentic_Pipelineipynb.ipynb

3️⃣ XGB_Model_Downloadipynb.ipynb

Purpose:
Loads or prepares the four trained XGBoost critic models.

Models:

Output file	Task
xgb_output_0.json	STANCE
xgb_output_1.json	ACTION
xgb_output_2.json	PERSONALNESS
xgb_output_3.json	POLITENESS

What it does:

Loads trained XGB models

Saves them in .json format

Ensures compatibility with SBERT embeddings

Important:
Models are trained on SBERT embeddings (384-d).

4️⃣ Agentic_Pipelineipynb.ipynb

Purpose:
Implements the full agentic generation + critic feedback loop.

This is the main notebook of the project.

What it does:

Loads:

SBERT model

XGB critics

Label encoders

Dataset (optional)

Sends a tweet to LLaMA (Groq API)

Generates multiple candidate replies

Embeds (tweet + reply) using SBERT

Gets critic predictions (probabilities if available)

Computes an alignment score

Applies a threshold-based decision:

✅ Accept reply if score ≥ threshold

❌ Otherwise rewrite with stricter prompt constraints

Returns the best reply found

Key features:

Best-of-N sampling

Probability-based scoring

Label-specific prompt tightening

Dataset-driven or manual testing

Scoring and Threshold Logic

Each critic outputs a probability for the gold class

Overall score is the weighted average of the four probabilities

STANCE = 0.45
ACTION = 0.50
PERSONALNESS = 0.90
POLITENESS = 0.85

Score = average ≈ 0.675

Score ≥ 0.75 → ACCEPT
Score < 0.75 → REWRITE

