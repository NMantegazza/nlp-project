# Natural Language Processing Project – Word Embeddings and Fine Tuning LLM

![Language](https://img.shields.io/badge/language-Python-blue.svg)
[![University](https://img.shields.io/badge/Politecnico%20di%20Milano-NLP%20Project-red)](https://www.polimi.it)
![Academic Year](https://img.shields.io/badge/AY-2024%2F2025-lightgrey)

## 📝 Overview

This repository contains my individual contribution to the **Natural Language Processing** project at **Politecnico di Milano**.  
The focus of this work is on **training custom word embeddings** and **fine-tuning a large language model (LLaMA2)** on a biomedical question-answering dataset ([*PubMedQA*](https://pubmedqa.github.io)).  

## 🎯 Project Objectives

The project explores advanced NLP techniques aimed at improving the understanding and reasoning capabilities of large models within the biomedical domain.  
Specifically, this work focuses on two main goals:

1. **Word2Vec Embedding Training**  
   Develop, train, and evaluate **Word2Vec** embeddings on biomedical text corpora to capture domain-specific semantic relationships between terms.

2. **Fine-Tuning LLaMA2 on PubMedQA**  
   Adapt [**Meta LLaMA2**](https://huggingface.co/meta-llama/Llama-2-7b-hf) model for biomedical question answering through parameter-efficient fine-tuning techniques, assessing improvements in performance and generalization.

## ⚙️ Implementation Details

The implementation is divided into two Jupyter notebooks:

* `Word2Vec_embeddings.ipynb` – trains Word2Vec embeddings using biomedical datasets, explores vector similarities, and visualizes semantic structures through dimensionality reduction (PCA/t-SNE).  
* `LLaMa2_fine_tuning_on_PubMedQA_A.ipynb` – preprocesses the PubMedQA dataset, fine-tunes the LLaMA2 model using PyTorch and Hugging Face Transformers, and evaluates the model on domain-specific QA tasks.

### 🧩 Frameworks and Libraries

The main technologies used include:

* **Python** – Core programming language.  
* **PyTorch** – Model training and fine-tuning backend.  
* **Transformers** (Hugging Face) – Model management, tokenization, and LoRA fine-tuning.  
* **Gensim** – Word2Vec training and evaluation.  
* **Scikit-learn** – Dimensionality reduction and clustering analysis.  
* **Pandas / NumPy / Matplotlib** – Data manipulation and visualization.

## 📊 Results and Insights

Key takeaways from the experiments:

* **Custom Word2Vec embeddings** captured meaningful semantic relationships between biomedical concepts (e.g., drug–disease and gene–function associations).  
* **Fine-tuned LLaMA2** demonstrated improved performance on PubMedQA, achieving higher accuracy in question-answering compared to the base model.  
* These findings highlight the **impact of domain adaptation** when applying general-purpose LLMs to specialized contexts like biomedicine.

## 📚 Citation

> **PubMedQA: A Dataset for Biomedical Research Question Answering**  
> Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu.  
> *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 2019.
