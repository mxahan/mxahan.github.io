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
