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
