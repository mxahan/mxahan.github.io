---
layout: post
title: "Paper_Summary1"
categories: papers
---

# Clevrer: Collision events for video representation and reasoning [1]

#### IMRAD
Missing in datasets; underlying logic, temporal and causal structure behind the reasoning process!! they study these gaps from **complementary perspective**.

Video reasoning models! - provide benchmark - Video dataset proposal!

Motivation: CLEVR, psychology

<img src = "https://d3i71xaburhd42.cloudfront.net/7f3ecbe546efed8ba42812f977354c16590bad77/2-Figure1-1.png">
Figure: Sample from the dataset [1]

Analysis of various visual reasoning models on the CLEVRER.

### Prior Arts
Video Understanding, VQA, Physical and Causal reasoning.
#### Argument and Assumption/ Context
Dataset to fill the missing domain should: Video, diagnostic annotation, temporal relation, Explanation, prediction, counterfactual. Need to match this.

<img src  = "https://d3i71xaburhd42.cloudfront.net/7f3ecbe546efed8ba42812f977354c16590bad77/3-Table1-1.png">
Figure: Dataset Comparison [1]

#### Problem statement
Missing in datasets; underlying logic, temporal and causal structure behind the reasoning process!! Can we prepare a dataset covering this missing links!
#### contribution
Dataset

NS-DR model ??

#### Approach and Experiments:
Controlled environment. Offered CLEVRER dataset.
CLEVRER:
 - 10000, 5000, 5000 (T/V/T) - 5 seconds videos
 - Object and events:
 - Causal structure
 - Video Generation - Simulator
 - 4 types of Quesions

#### Evaluations
used base line models:
- Language-only models: Q-types, LSTM, weak baseline
- VQA: CNN + LSTM, TVQA,
- Compositional visual reasoning: IEP,
combined with Various implementation tricks.

<img src ="https://d3i71xaburhd42.cloudfront.net/7f3ecbe546efed8ba42812f977354c16590bad77/6-Table2-1.png">
Figure: Evaluated models [1]


#### Results:
Show different model strengths.

#### Neuro-Symbolic Dynamic Reasoning - Model
oracle model
- Video Frame parser - Mask RCNN
- Neural Dynamic Predictor - PropNet
- Question Parser - Attention based seq2seq model
- Program Executor -

#### My thoughts
- Oracle Model
- How it concludes other model strengths
#### Key ideas & Piece of Pie
Better dataset preparation

#### Reference
[1] Yi, Kexin, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B. Tenenbaum. "Clevrer: Collision events for video representation and reasoning." arXiv preprint arXiv:1910.01442 (2019).
