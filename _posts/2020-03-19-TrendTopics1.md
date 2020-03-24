# Representation Learning
### Representation learning: A review and new perspective - (Bengio, Y. et al. 2014)
[original paper](https://arxiv.org/pdf/1206.5538.pdf)

Review paper on unsupervised learning covering *probabilistic model*, auto-encoders, manifold learning, and deep networks to learn the data Representation.

Representation learning; learning data Representation that make it easier to extract useful information when building classifiers and other predictors.

##### *Application:*
- Speech recognition and signal processing
- Object recognition
- NLP
- Multi-Task and transfer learning, Domain Adaptation.

##### *What makes Representation good?*
- Priors for Representation Learning in AI
  - Smoothness:
  - Multiple explanatory factors
  - Hierarchical organization of explanatory factors
  - Semi-supervised Learning
  - Shared factors across task - MT, DA, TL
  - Manifolds
  - Natural clustering
  - Temporal and spatial coherence
  - Sparsity
  - Simplicity of Factor Dependencies
- Smoothness and the curse of dimensionality  
- Distributed representations
- Depth and abstraction - this paper
  - Feature re-use
  - abstraction and invariance
- Disentangling factors of Variation: information preserving and disentangle factors simultaneously.
- Good criteria for learning representations:

##### *Building Deep Representations*

Greedy layerwise unsupervised pre-training- Resulting deep features used as input to another standard classifier. Another approach is greedy layerwise supervised pre-training. Layerwise stacking often leads better representations.
- Stack pre-trained RBM into DBN (wake sleep algorithm)
- Combine RBM parameters into DBM
- Stack RBMs or auto-encoders into a deep auto-encoders.
  - Encoder-decoder block <img src = "https://latex.codecogs.com/gif.latex?(f^{(i)}(.),&space;g^{(i)}(.))">
  - composition of encoder: <img src = "https://latex.codecogs.com/gif.latex?f^{(N)}(...f^{(2)}(f^{(1)}(.)))">
  - Decoder block: <img src ="https://latex.codecogs.com/gif.latex?g^{(1)}(g^{(2)}...(f^{(N)}(.)))">
- Iterative construction of *free energy function*

##### *single-layer learning modules*
Two approaches
- probabilistic graphical model
- Neural networks

PCA connected to probabilistic models, Encoder-decoder, Manifolds

#### Probabilistic Model

Learning joint distribution <img src = "https://latex.codecogs.com/gif.latex?p(x,&space;h)">  and posterior distribution <img src = "https://latex.codecogs.com/gif.latex?p(h|&space;x)">, where *h* is the latent variables for *x*.

##### 1. Directed Graphical Models

<img src = "https://latex.codecogs.com/gif.latex?p(x,&space;h)=p(x|&space;h)p(h)">

##### Probabilistic interpretation of PCA
Prior;
<img src = "https://latex.codecogs.com/gif.latex?p(h)=&space;\mathcal{N}(h;0,\sigma_h^2, I)">

And the likelihood
<img src = "https://latex.codecogs.com/gif.latex?p(x|h)=&space;\mathcal{N}(x;Wh+\mu_x,\sigma_x^2, I)">

##### Sparse Coding

Clever equation regarding the prior and likelihood. A variation of sparse coding leads to *Spike-and-slab sparse coding (S3C)*. S3C outperforms sparse coding shown by Goodfellow et al.

##### 2. Undirected Graphical Models

Markov random fields. Expressed joint probability as the multiplication of clique potentials. A spacial form of markov random field is Boltzmann distribution with positive clique potentials. Here comes the notion of energy function.

##### Restricted Boltzmann machines (RBMs)

A little modification in energy function in Boltzmann machines leads to Restricted Boltzmann machines.  

##### Generalizations of the RBM to real-valued data

### Directly Learning A parametric Map from input to Representation

Graphical models are intractable for multiple layers.
- Learn a direct encoding; a parametric map from input to their Representation

##### 1. Auto-encoders
encoder
<img src = "https://latex.codecogs.com/gif.latex?h=&space;f_\theta(x)">

For each data instances
<img src = "https://latex.codecogs.com/gif.latex?h^t=&space;f_\theta(x^t)">

Decoder
<img src = "https://latex.codecogs.com/gif.latex?r=&space;g_\theta(h)">

Now the objective function to minimize

<img src = "https://latex.codecogs.com/gif.latex?\mathcal{J}_{AE}(\theta)=\sum_tL(x^t, g_\theta(f_\theta(x^t)))">

##### 2. Regularized Auto-encoders

Use a bottleneck by forcing latent space dimension lower than the input space dimention.  

##### Sparse auto-encoders

Applying the sparsity regularization to force the latent representation to be close to zero.

### Representation learning as Manifold Learning

May be I will add later.

### Connection between probabilistic and Directed encoding Models

### Global training in Deep Models

### Building in invariance

#### Generating transformed Examples
#### Convolution and pooling
#### Temporal coherence and Slow features
#### Algorithms to disentangle factors of Variation
