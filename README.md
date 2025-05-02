# Adaptive Multi-Hop Retrieval via Encoder Training and Query Refinement

## Project Overview

This project presents an adaptive multi-hop retrieval framework that combines **iterative dense retriever fine-tuning** with **attention-based query embedding refinement**. The goal is to improve retrieval accuracy in multi-hop question answering tasks, where a query requires retrieving a chain of interdependent documents.

Our approach extends traditional dense retrieval (DPR) by enabling both the retriever and query embeddings to evolve at each hop based on retrieval feedback. This allows the system to dynamically adapt to context, reduce error propagation, and improve answer coverage.

---

## Features

- Dense retriever fine-tuning with iterative contrastive learning
- Attention-based dynamic query embedding refinement across hops
- Confidence-based fallback mechanisms to maintain retrieval robustness
- Evaluated on HotpotQA and 2WikiMultiHopQA datasets

---

## File Descriptions

- `IST558_Final_Hotpot.ipynb`: Code implementation for fine-tuning DPR, multi-hop retrieval, and query refinement.
- `IST 558 Final Project Report_Group 10.pdf`: Full report detailing the methodology, experiments, and findings.

---

## Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install transformers faiss-cpu sentence-transformers scikit-learn

2. **Download and preprocess HotpotQA or 2WikiMultiHopQA:**

Ensure data is in the expected format (question, context, supporting facts).

3. **Run the notebook:**

Modify parameters like number of hops or alpha blending factor if needed.


## Results Summary
On HotpotQA, our model achieved:
60.1% Average Recall
61.4% MRR
51.0% All-Support Accuracy

On 2WikiMultihopQA, we achieved:
Average Recall: 52.1%
Mean Reciprocal Rank (MRR): 75.7%
All-Support Accuracy: 20.0%

Attention-based refinement and iterative training contributed to significant improvements over the raw DPR baseline.

