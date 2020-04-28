# Paper Review Part 1

I understand the hard-work a researcher has to go through to get a publication or continuing research. As a researcher I take ideas from multiple sources for my own research to contribute in a effective way towards my fields. It is absolutely devilish/unacceptable to knowingly plagiarize/steal ideas from people without properly crediting them.

This blog contains review of different interesting papers for my personal interest. I have tried to cite them in a correct way all the way. Please let me know If I have mistaken to cite something in the proper way. Moreover I would be glad if somebody wants to discuss/ argue with my understandings. I am always flexible to change my understanding based on facts. {(zhasan3@umbc.edu) subject: Missed citation in proper way/ Paper discussion from blog}

Paper List

- [Clevrer: Kexin et al 2019](#paper1)
- [Style Gan- Tero Karras et al 2018](#paper2)
- [Style transfer - Xun Huang et al 2017](#paper3)



# Paper1
Clevrer: Collision events for video representation and reasoning [1]

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


# Paper2
A Style-Based Generator Architecture for Generative Adversarial Networks [2]

[Interesting blog](https://towardsdatascience.com/explained-a-style-based-generator-architecture-for-gans-generating-and-tuning-realistic-6cb2be0f431)

#### IMRAD
- Gap in GAN latent spaces properties understanding
- Gap in measuring factor of variation without enc-dec.
- Generator comparison!
- Motivation from style transfer literature
- No change in loss function



### Prior Arts

#### Argument and Assumption/ Context
- AdaIN can be expanded beyond encoder-decoder settings.
- Disentanglement can be used as generator improvement measures.


#### Problem statement

Traditional GAN generator improvement via style learning layer by layer. Can the feature be better selected by introducing a trick before feeding to generator? Is it possible to entangle the styles and features?  

#### contribution

- Style based generator proposal
- No change in GAN loss/ Discriminator
- Argues - intermediate latent space doesn't have to follow the distribution of Input
- Two metrics propose -  Perceptual path length and Linear separability
- FFHQ dataset


#### Approach and Experiments:

Style based Generator:

- Architecture:

From random vector, z, to a vector space, w, of same size (I guess loss backpropagates here too). Affine transformation learn (how?) from w to y (Get y from random vector instead of style image!). Then the AdaIN layer.  

<img src="https://latex.codecogs.com/gif.latex?AdaIN(x_i,y)=y_{s,i}(\frac{x_i-\mu(x_i)}{\sigma(x_i)})+y_{b,i}">  

Noise (Gaussian noise) introduction cause stochastic variation in the generated images.

<img src="https://miro.medium.com/max/1400/0*ANwSHXJDmwqjNSxi.png">

Figure: [Source](https://towardsdatascience.com/explained-a-style-based-generator-architecture-for-gans-generating-and-tuning-realistic-6cb2be0f431)

Important Note about Style generator: Each layer overridden by next AdaIN operation. Each AdaIN controls one CNN. As normalization and rescaling happen after each layers.

- Style Mixing: Mixing regularization. Switch from one latent code to another at random points. Has option to add the style at different scale in the generator network. Two latent codes z1 and z1 generated for the two sets of images.
- stochastic variation: Hair line, number, eye color etc. - controlled by the noise.
- Separation of global effects from stochasticity: Pose, lightening, - controlled by the latent space.

Disentanglement studies: Latent space consists of linear subspaces each basis controls one factor of variation. Latent space W can be learned from the random vector by f(z).

- Perceptual path length: Pairwise image distance as a weighted difference between two VGG16 embeddings. The authors used R. Zhang et al 2018 metrics. A bit different from only taking layerwise distance in VGG16 layers, spherical interpolation between latent space vectors.

<img src="https://latex.codecogs.com/gif.latex?l_z=E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t)),G(slerp(z_1,z_2;\epsilon+t)))]">

where,

<img src="https://latex.codecogs.com/gif.latex?slerp(x,y;t=\frac{\sin[(1-t)\Omega]}{\sin\Omega}x+\frac{\sin[t\Omega]}{\sin\Omega}y">

and

<img src="https://latex.codecogs.com/gif.latex?l_z=E[\frac{1}{\epsilon^2}d(G(lerp(f(z_1),f(z_2);t)),G(lerp(f(z_1),f(z_2);\epsilon+t)))]">

- Linear separability: Distinguished set for distinguished features. How much entropy has reduce in H(Y|X); Where Y is the image label and X is the feature label.

#### Evaluations


#### Results:
- Experiment without any styles

<img src="https://miro.medium.com/max/1400/0*eKvFqsrzvHdc70dp.png">

Figure: Source [2]


#### My thoughts
Not the traditional style transfer from one image to another. Rather a group to another. There is no image input here, only the random vector like the original GAN.
#### Key ideas & Piece of Pie

AdaIN in generator. Feature transform network for the latent random variables.

#### Reference


[2] Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

(R. Zhang et al 2018) R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proc. CVPR, 2018. 6, 7




# Paper3
Arbitrary Style transfer in Real-time with Adaptive Instance Normalization [3]

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

<img src="https://github.com/xunhuang1995/AdaIN-style/raw/master/examples/architecture.jpg">

Figure: Mind the AdaIN layer in between enc-dec. [3]

Loss Function: Weighted sum of the two losses
- Content Loss: Euclidean distance between the AdaIN features and dec-enc(AdaIN features).

<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_c=\sqrt{||f(g(t))-t||}">


- Style loss: Measures the loss between the statistics of the generated image and the style image in the VGG19 Layer stages (Authors used relu 1 to 4).

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

<img src="https://github.com/xunhuang1995/AdaIN-style/raw/master/examples/style_weight.jpg">

Figure: Source [3]

To interpolate set of K styles, the authors performs convex weighted sum of the AdaIN features for the K styles


- Style interpolation

To interpolate set of K styles, the authors performs convex weighted sum of the AdaIN features for the K styles

<img src="https://latex.codecogs.com/gif.latex?T(c,s_{1,..,K},w_{a,..,K})=g(\sum_{i=1}^KAdaIN(f(c),f(s_i))">

Example

<img src="https://github.com/xunhuang1995/AdaIN-style/raw/master/examples/style_interp.jpg">

Figure: Source [3]

- Spatial and color control: Different style in different image portions.





#### My thoughts

#### Key ideas & Piece of Pie


#### Reference

[3] Huang, Xun, and Serge Belongie. "Arbitrary style transfer in real-time with adaptive instance normalization." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1501-1510. 2017.