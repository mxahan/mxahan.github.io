# Trending Topics 2

---
layout: post
title: "TrendingTopics"
categories: Hotopics
---

The part1 consists of 10 topics, this is continuation covering some other interesting Topics

- [Collaborative Filtering](#collaborative-filtering)
- [Metric Learning](#metric-learning)
- [Federated Learning](#federated-learning)
- [BERT](#bert)
- [Self-supervision](#self-supervision)
- [Divergence algorithm](#divergence-algorithm)
- [Meta Learning](#meta-learning)


# Collaborative Filtering

# Metric Learning


# Federated Learning
[Nice Introduction](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

Decentralized training! Google uses in mobile phone.

- Start from a generalized model
- Personalize it [Many other personalize it too!]
- Track the ensemble change (from different users) and update the local model.

# Atrous Convolution and friends

[Motivation](https://towardsdatascience.com/review-deeplabv1-deeplabv2-atrous-convolution-semantic-segmentation-b51c5fbde92d)

Origin French "a trous" or hole, a.k.a dilated convolution.

<img src="https://latex.codecogs.com/gif.latex?y[i]=\sum__{k=1}^K x[i + r.k]w[k]">

where rate, r is an positive integer and r = 1 means regular convolution. Allow to enlarge the field of view.

#### Atrous Spatial Pyramid Pooling (ASPP)

#### Fully connected Conditional Random Field (CRF)


# BERT

one of the great post and [motivation](http://jalammar.github.io/illustrated-bert/).

Two most important point
  - Semi supervised training (Wiki, large language) [MLM, NSP]
  - Supervised data specific training (QA, ....)

BERT build on
  - [Semi supervised sequence learning](https://arxiv.org/abs/1511.01432) - Fine tuning concept
  - [ELMo](https://arxiv.org/abs/1802.05365) - Contextual embedding
  - [ULM-Fit](https://arxiv.org/abs/1801.06146) - Fine tuning and **tx learning**
  - OpenAI transformer
  - Transformer (Vaswani et al)

Model Architecture:
  - BERT base
  - BERT large

Need to have ideas regarding the word embedding and contextual word embedding.

Transformer: Better long term dependencies than LSTMs. Encoder-decoder for MT. But how to use it for sentence?

OpenAI transformer: Decoder of transformer only!! Predict next words 7000 books for training. But unidirectional?

BERT: Bidirectional, used encoders!!

Pretraining:
  - MLM: Mask to rescue from word seeing itself in bidirectional setting. 15% words masked in their approach.
  - Two sentence task:

Down-streaming Task:
  - sentence classification
    - Single
    - Pair
  - QA tasks
  - Sentence tagging

BERT for feature Extraction:
  - Contextual word embedding
  - Named entity recognition

[another summary](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)

[original release](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

# Self Supervision

[Best starting point](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#contrastive-predictive-coding)

[great paper collection](https://github.com/jason718/awesome-self-supervised-learning) Definitely check it out.

Self-supervised task also named as pretext task. Before the final/downstream tasks.

# Divergence Algorithm

[contrastive divergence medium](https://medium.com/datatype/restricted-boltzmann-machine-a-complete-analysis-part-3-contrastive-divergence-algorithm-3d06bbebb10c)

# Meta-learning
[initial ideas](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)

learning to learn. Generalized approach in learning ! (supervised, RL)

Where we can apply! Problem definition:

Solutions: Model-based, metric-based, optimization Based

Metric-based: Siamese Neural Network, Matching Network, Relation Network, Prototypical Network

Model-based: Memory-augmented NN, Meta-Network

Optimization-based: Meta-learner, MAML, first order MAML, Reptile 


# Attention

[link](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#neural-turing-machines)
