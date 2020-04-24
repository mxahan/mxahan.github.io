---
layout: post
title: "Paper_Summary1"
categories: papers
---

I understand the hard-work a researcher has to go through to get a publication or continuing research. As a researcher I take ideas from multiple sources for my own research to contribute in a effective way towards my fields. It is absolutely devilish/unacceptable to knowingly plagiarize/steal ideas from people without properly crediting them.

This blog contains review of different interesting papers for my personal interest. I have tried to cite them in a correct way all the way. Please let me know If I have mistaken to cite something in the proper way. Moreover I would be glad if somebody wants to discuss/ argue with my understandings. I am always flexible to change my understanding based on facts. {(zhasan3@umbc.edu) subject: Missed citation in proper way/ Paper discussion from blog}

Paper List

- [Clevrer: Kexin et al 2019](#clevrer:-collision-events-for-video-representation-and-reasoning-[1])
- [Style Gan- Tero Karras et al 2018](#a-style-based-generator-architecture-for-generative-adversarial-networks-[2])
- [Style transfer - Xun Huang et al 2017](#arbitrary-style-transfer-in-real-time-with-adaptive-instance-normalization-[3])



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


# A Style-Based Generator Architecture for Generative Adversarial Networks [2]


#### IMRAD
- Gap in GAN latent spaces properties understanding
- Generator comparison!
- Motivation from style transfer literature

Contribution:
- Generator adjusts at each layer
- Directly noise injection into the networks
- No change in GAN loss/ Discriminator
- Input latent code to intermediate latent space
- Argues - intermediate latent space doesn't have to follow the distribution of Input
- Two metrics propose -  Perceptual path length and Linear separability
- FFHQ dataset

Style based generator

### Prior Arts

#### Argument and Assumption/ Context


#### Problem statement

#### contribution


#### Approach and Experiments:


#### Evaluations


#### Results:



#### My thoughts

#### Key ideas & Piece of Pie


#### Reference


[2] Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

# Arbitrary Style transfer in Real-time with Adaptive Instance Normalization [3]

#### IMRAD
- Gap in finite styles and slower
- Motivated from Instance Normalization
- Extension of IN based on observation
- Encoder-decoder structure (In between proposes an AdaIN layer)

- Architectural and Loss function contribution
- Conceptual difference between earlier and this method
  - Directly changes statistics in feature space in One Shot - instead of pixel value manipulation  


### Prior Arts

- Style transfer
- Deep generative image model

#### Argument and Assumption/ Context
- IN performs style normalization by normalizing features stats that carries style information.
- Need to change the statistics of the contents to match the style
- Extension of IN - AdaIN
- Need something in between Encoder-Decoder
- Style transfer occurs in AdaIN layer

#### Problem statement

Can the generation be more depend on the statistical features instead of the pixel value itself? Can the features be transferred by statistically to match the style?

#### contribution (Piece of Pie)

AdaIN
 - Given contents and Style AdaIN adjusts the mean, variance of contents to the styles
 - Decoder to convert AdaIN output
 - User control and time
 - Style transfer happens here

#### Background
- Batch Normalization layer: Across individual feature channel

<img src="https://latex.codecogs.com/gif.latex?BN(x)=\gamma(\frac{x-\mu(x)}{\sigma(x)})+\beta">

Where
<img src="https://latex.codecogs.com/gif.latex?\mu_c(x)=\sum_{n=1}^N\sum_{h=1}^H\sum_{w=1}^Wx_{nchw}/(NHW)"> and SD accordingly.  

- Instance Normalization layer: The difference in computing mean and SD from the BN.

<img src="https://latex.codecogs.com/gif.latex?IN(x)=\gamma(\frac{x-\mu(x)}{\sigma(x)})+\beta">

<img src="https://latex.codecogs.com/gif.latex?\mu_{nc}(x)=\sum_{h=1}^H\sum_{w=1}^Wx_{nchw}/(HW)"> and SD accordingly.  

- Conditional Instance Normalization:

<img src="https://latex.codecogs.com/gif.latex?CIN(x;s)=\gamma^s(\frac{x-\mu(x)}{\sigma(x)})+\beta^s">




#### Approach and Experiments:

Preprocessing
- resize, Batch size- 8, Adam

Architecture:

Encoder f, Adaptive IN layer,  Decoder g

- Adaptive Instance Normalization: (AdaIN)

<img src="https://latex.codecogs.com/gif.latex?AdaIN(x,y)=\sigma(y)(\frac{x-\mu(x)}{\sigma(x)})+\mu(y)">

Features t,
<img src="https://latex.codecogs.com/gif.latex?t=AdaIN(f(c),f(s))">

And reconstructed image
<img src="https://latex.codecogs.com/gif.latex?T(c,s)=g(t)">


uses pretrained VGG19 models

<img src="https://github.com/xunhuang1995/AdaIN-style/raw/master/examples/architecture.jpg" width=600>

Figure: Mind the AdaIN layer in between enc-dec. [3]

Loss Function: Weighted sum of the two losses
- Content Loss
Euclidean distance between the AdaIN features and dec-enc(AdaIN features).
<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_c=\sqrt{||f(g(t))-t||}">

- Style loss
Measures the loss between the statistics of the generated image and the style image in the VGG19 Layer stages (Authors used relu 1 to 4).
<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_s=\sum_{i=1}^L(||\mu(\phi_i(g(t))) - \mu(\phi_i(s)||^{.5}+||\sigma(\phi_i(g(t))) - \sigma(\phi_i(s)||^{.5})">



#### Evaluations


#### Results:
Dataset
- MS-COCO dataset
- WikiArt

Metrics
- Quantitative Analysis
- Qualitative Analysis
- Speed analysis

Alternatives:
- Enc-AdaIn-Dec (this paper)
- Enc-concat-Dec (earlier method)
- Enc-AdaIN-BNDec (this)
- Enc-AdaIN-INDec (this)

Controls (No modify in the training) in test-time
- Content-style trade-Off:

Convex interpolation between output of encoder and the AdaIN.

<img src="https://latex.codecogs.com/gif.latex?T(c,s,\alpha)=g((1-\alpha)f(c)+\alpha AdaIN(f(c),f(s)))">

Example

<img src="https://github.com/xunhuang1995/AdaIN-style/raw/master/examples/style_weight.jpg" width=600>

Figure: Source [3]

To interpolate set of K styles, the authors performs convex weighted sum of the AdaIN features for the K styles


- Style interpolation

To interpolate set of K styles, the authors performs convex weighted sum of the AdaIN features for the K styles

<img src="https://latex.codecogs.com/gif.latex?T(c,s_{1,..,K},w_{a,..,K})=g(\sum_{i=1}^KAdaIN(f(c),f(s_i))">

Example

<img src="https://github.com/xunhuang1995/AdaIN-style/raw/master/examples/style_interp.jpg" width=600>

Figure: Source [3]

- Spatial and color control: Different style in different image portions.





#### My thoughts

#### Key ideas & Piece of Pie


#### Reference

[3] Huang, Xun, and Serge Belongie. "Arbitrary style transfer in real-time with adaptive instance normalization." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1501-1510. 2017.
