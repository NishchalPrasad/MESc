# MESc
Multi-stage Encoder-based Supervised with-clustering (MESc) classification framework

This repository contains the code for our ECIR 2024 paper "[Exploring Large Language Models and Hierarchical Frameworks for Classification of Large Unstructured Legal Documents](https://doi.org/10.1007/978-3-031-56060-6_15)", Advances in Information Retrieval, 46th European Conference on Information Retrieval, ECIR 2024.
* **Authors:** Nishchal Prasad, Mohand Boughanem, Taoufiq Dkaki 

Our proposed framework is gisted below:

<img src="/Images/MESc_architecture.png" alt="MESc architecture" width="300"/>
<!-- ![Architecture](/Images/MESc_architecture.png) -->

>Legal judgment prediction suffers from the problem of long case documents exceeding tens of thousands of words, in general, and having a non-uniform structure. Predicting judgments from such documents becomes a challenging task, more so on documents with no structural annotation. We explore the classification of these large legal documents and their lack of structural information with a deep-learning-based hierarchical framework which we call MESc; “Multi-stage Encoder-based Supervised with-clustering”; for judgment prediction. Specifically, we divide a document into parts to extract their embeddings from the last four layers of a custom fine-tuned Large Language Model, and try to approximate their structure through unsupervised clustering. Which we use in another set of transformer encoder layers to learn the inter-chunk representations. We analyze the adaptability of Large Language Models (LLMs) with multi-billion parameters (GPT-Neo, and GPT-J) with the hierarchical framework of MESc and compare them with their standalone performance on legal texts. We also study their intra-domain(legal) transfer learning capability and the impact of combining embeddings from their last layers in MESc. We test these methods and their effectiveness with extensive experiments and ablation studies on legal documents from India, the European Union, and the United States with the ILDC dataset and a subset of the LexGLUE dataset. Our approach achieves a minimum total performance gain of approximately 2 points over previous state-of-the-art methods.



# Short description of the work:
* We explore the problem of judgment prediction from large unstructured legal documents and propose a hierarchical multi-stage neural classification framework named “Multi-stage Encoder-based Supervised with-clustering” (MESc). This works by extracting embeddings from the last four layers of a fine-tuned encoder of a large language model (LLM) and using an un-supervised clustering mechanism to approximate the structure. Alongside the embeddings, these approximated structure labels are processed through another set of transformer encoder layers for final classification.
* We show the effect of combining features from the last layers of transformer-based LLMs ([BERT](https://doi.org/10.18653/v1/n19-1423), [GPT-Neo](https://api.semanticscholar.org/CorpusID:245758737), [GPT-J](https://huggingface.co/docs/transformers/en/model_doc/gptj)), along with the impact on classification upon using the approximated structure.
* We study the adaptability of domain-specific pre-trained multi-billion parameter LLMs to such documents and study their intra-domain(legal) transfer learning capability (both with fine-tuning and in MESc).
* We performed extensive experiments and analysis on four different datasets ([ILDC](https://aclanthology.org/2021.acl-long.313) and [LexGLUE’s](https://aclanthology.org/2022.acl-long.297) ECtHR(A), ECtHR(B), and SCOTUS) and achieved a total gain of ≈ 2 points in classification on these datasets.


If you consider this work to be useful, please cite it as:

```bash
@InProceedings{10.1007/978-3-031-56060-6_15,
author="Prasad, Nishchal
and Boughanem, Mohand
and Dkaki, Taoufiq",
title="Exploring Large Language Models and Hierarchical Frameworks for Classification of Large Unstructured Legal Documents",
booktitle="Advances in Information Retrieval",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="221--237",
isbn="978-3-031-56060-6"
}
```
This codebase is being updated and will be completed in soon. 

# Contact

For any queries, feel free to contact Nishchal Prasad (prasadnishchal.np@gmail.com)
