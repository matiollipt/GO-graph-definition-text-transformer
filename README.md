# Predicting Gene Ontology (GO) Aspects with Fine-Tuned Text Transformer

High-throughput [DNA sequencing](https://en.wikipedia.org/wiki/DNA_sequencing#High-throughput_sequencing_(HTS)_methods) has give us the power to sequence the entire genome of a species within a day. With the sequence on hands, we can deploy **computational models to identify and annotate genes** based on their characteristics, attributing correct functions and subcellular locations of the gene product given the gene sequence. For example, the [Critical Assessment of Protein Function Annotation (CAFA)](https://www.cell.com/trends/genetics/fulltext/S0168-9525(13)00166-2) competition engages the data science community in enhancing protein prediction by utilizing features derived from [Gene Ontology (GO)](https://geneontology.org/docs/ontology-documentation/).

The Gene Ontology is represented as a **directed acyclic graph (DAG**) where **each node represents a specific GO term**. Each GO term defines a particular aspect of genes and their products. 

The GO terms are organized into three main categories: **Molecular Function (MF)**, **Cellular Component (CC)** and **Biological Process (BP)**:

- <font color="grey">***Molecular Function (MF)***</font>: These terms define the **activities** performed by gene products, such as *catalysis* or *transport*. These functions can be further refined by more specific GO terms, for example, "protein kinase activity" within the broader category of "catalysis".

- <font color="grey">***Cellular Component (CC)***</font>: These terms specify the subcellular **locations** of gene products, including compartments like *chloroplast* or *nucleus*, as well as macromolecular complexes like *proteasome* or *ribosome*.

- <font color="grey">***Biological Process (BP)***</font>: These terms delineate the biological **pathways** in which gene products are involved, ranging from 'DNA repair' and 'carbohydrate metabolic process' to overarching processes like *biosynthetic processes*.


The relationships between these terms are **hierarchical**, with **parent-child relationships** indicating broader and more specific terms, respectively. This hierarchical structure allows researchers to **annotate genes and gene products**, providing valuable information about their functions and roles in biological processes. For more information about how the GO graph is structured, please refer to my previous post [GO-graph-EDA](https://github.com/matiollipt/GO-graph-EDA) and the [Gene Ontology reference](https://geneontology.org/docs/ontology-documentation). For now, it is essential to know that **each node representing a GO term has specific attributes**.

A feature often overlooked when deploying the GO graph to assist in the prediction and classification of gene functions is the **textual definition of each GO term**. For example, the GO term ID [GO:0015986](https://www.ebi.ac.uk/QuickGO/term/GO:0015986) is defined as *"The transport of protons across the plasma membrane to generate an electrochemical gradient (proton-motive force) that powers ATP synthesis."*, along with other attributes shown below:

---
 **GO id**: GO:0015986

- **name**: 'proton motive force-driven plasma membrane ATP synthesis'
- **namespace**: 'biological_process'
- **def**: '"The transport of protons across the plasma membrane to generate an electrochemical gradient (proton-motive force) that powers ATP synthesis." [GOC:mtg_sensu, ISBN:0716731363]'
- **synonym**: ['"ATP synthesis coupled proton transport" BROAD []', '"plasma membrane ATP synthesis coupled proton transport" EXACT []']
 - **is_a**: ['GO:0015986']}

 ---

In this first simple exercise, we will use the text definitions of GO terms to **fine-tune** a pre-trained Large Language Model (LLMs) based on the [transformer architecture](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)). The **goal** is to leverage this LLM to **categorize GO term definitions** based on their alignment with the major GO ontologies (BP, CC, and MF).

The transformer architecture, proposed by the [Google Brain Team](https://research.google/) in the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), revolutionized [Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing) by introducing the idea of using solely **attention mechanisms** to identify dependencies between words in a text sequence. This approach contrasts with previous ones by dropping **recurrent layers** in [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) networks to handle (i.e., keep track of previous words/tokens) text sequences. In the Transformer architecture, the order of the sequence is retained through **positional encoding**, and the **dependencies** are computed using multi-head attention to keep track of more than one dependency in the input sequence.

<figure><p align="center"><img src="transformer_architecture_Vaswani_etal_2017.png" alt="Transformer Basic Architecture" width=400px></p></figure>

**Transformers** have gained significant attention in the machine learning landscape, contributing to the development of generative models like [chatGPT](https://chat.openai.com). Text transformer-based architectures are specifically engineered to analyze the relationships between words in sentences, considering their positions to accuratelly capture the meaning of the text. This capability is instrumental in tasks such as text classification and ranking, as well as enabling generative models to generate new sentences in response to queries.


>**Fine-tuning** is a powerful technique in machine learning that involves using models that have undergone **semi-supervised training** on extensive data from various sources. This process imparts the model with underlying semantic features of the language. After pre-training, the model is further trained in a **supervised manner** for a specific task, such as text generation, classification, or label prediction (for instance, identifying whether emails are spam or not). During this phase, the deeper encoded features of the pre-trained model are retained, but the trainable parameters of the output layer(s) are adjusted. For example, a feed-forward neural network can be added for binary classification of "spam" vs "not spam." These steps embody the concept of [Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/), a powerful machine learning technique that involves applying knowledge learned from one task effectively to another task.

In the **next post**, we will fine-tune the BERT model to **generate better embeddings of GO term definitions for protein function prediction**.
