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


