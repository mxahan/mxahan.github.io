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
<img src = "https://latex.codecogs.com/gif.latex?p(h)=&space;\mathcal{N}(h;0,\sigma_h^2,I)">

And the likelihood
<img src = "https://latex.codecogs.com/gif.latex?p(x|h)=&space;\mathcal{N}(x;Wh+\mu_x,\sigma_x^2,I)">

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





# Region based Object Detectors

[original post link](https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9)

3 Parts:

- Faster R-CNN, R-FCN, FPN
- SSD, YOLO, FPN and Focal loss
- Design choices, Lessons learned, and trend for object detection

#### Sliding-window Detectors


```Pseudocode
For window in windows
  patch = get_patch(image, window)
  results =  detector(patch)
```
![image](https://miro.medium.com/max/1453/1*BYSA3iip3Cdr0L_x5r468A.png)

#### Selective Search (SS)

Region proposal method to find Region of interest (ROIs).

#### R-CNN

Cares about 2000 ROIs by using a region proposal method. The regions are warped into fixed size images and feed into a CNN network individually. Then followed by fully connected layers to classify and refine the boundary box.

![img](https://miro.medium.com/max/2574/1*Wmw21tBUez37bj-1ws7XEw.jpeg)

The system flow

![image](https://miro.medium.com/max/1484/1*ciyhZpgEvxDm1YxZd1SJWg.png)

Pseudocode
```
ROIs = region_proposal(image)
for ROI in ROIs
    patch = get_patch(image, ROI)
    results = detector(patch)
```

#### Boundary box regressor

#### Fast R-CNN

R-CNN is slow. Instead use feature extraction for whole image. It proposes a feature extractor and region proposal method.

![image](https://miro.medium.com/max/2400/1*Dd3-sugNKInTIv12u8cWkw.jpeg)

Network flow:

![image](https://miro.medium.com/max/1642/1*fLMNHfe_QFxW569s4eR7Dg.jpeg)

Pseudocode
```
feature_maps = process(image)
ROIs = region_proposal(image)
for ROI in ROIs
    patch = roi_pooling(feature_maps, ROI)
    results = detector2(patch)
```
Here comes the multitask loss (Classification and localization loss)

#### ROI Pooling

#### Faster R-CNN

```
feature_maps = process(image)
ROIs = region_proposal(image)         # Expensive!
for ROI in ROIs
    patch = roi_pooling(feature_maps, ROI)
    results = detector2(patch)
```

Network flow

![image](https://miro.medium.com/max/1746/1*F-WbcUMpWSE1tdKRgew2Ug.png)

The region proposal is replaced by a Region Proposal network (convolutional network)

![image](https://miro.medium.com/max/3636/1*0cxB2pAxQ0A7AhTl-YT2JQ.jpeg)

#### Region Proposal network

ZF networks structure

![image](https://miro.medium.com/max/1210/1*z0OHn89t0bOIHwoIOwNDtg.jpeg)

#### Region-based Fully convolutional Networks (R-FCN)

Faster R-CNN

```
feature_maps = process(image)
ROIs = region_proposal(feature_maps)
for ROI in ROIs
    patch = roi_pooling(feature_maps, ROI)
    class_scores, box = detector(patch)         # Expensive!
    class_probabilities = softmax(class_scores)
```
R-FCN

```
feature_maps = process(image)
ROIs = region_proposal(feature_maps)         
score_maps = compute_score_map(feature_maps)
for ROI in ROIs
    V = region_roi_pool(score_maps, ROI)     
    class_scores, box = average(V)                   # Much simpler!
    class_probabilities = softmax(class_scores)
```

Networks

![image](https://miro.medium.com/max/2744/1*Gv45peeSM2wRQEdaLG_YoQ.png)

# Region based Fully Convolutional networks (R-FCN)

[Original post link](https://medium.com/@jonathan_hui/understanding-region-based-fully-convolutional-networks-r-fcn-for-object-detection-828316f07c99)

Two stage detection
- Generate region proposals (ROIs)
- Make Classification and localization predictions from ROIs

There is a notion of position, leads to position sensitive score maps. Example: For 9 nine features we get nine feature maps.
![imag](https://miro.medium.com/max/1336/1*HaOHsDYAf8LU2YQ7D3ymOg.png)

Now we take map from each feature map depending on the position. From each 9 feature maps we get select one box per feature map and select if for voting.


![image](https://miro.medium.com/max/949/1*K4brSqensF8wL5i6JV1Eig.png)

![image](https://miro.medium.com/max/1937/1*ZJiWcIl2DUyx1-ZqArw33A.png)

The creates total (C+1)x3x3 score maps. Further example: Person detectors. Total 9 features leads to 9 feature maps. And we select 9 feature maps and get one box from each map to get the voting.

![image](https://miro.medium.com/max/1603/1*DjPNw7AAIUh4OhHi3o4nlA.png)

The image of network R-FCN showed above and the following is network followed

![image](https://miro.medium.com/max/1718/1*nwGLuSFHkhJV3OELc76fBg.png)

#### Boundary Box Regression

Convolutional filter creates kxkx(C+1) scope maps. Another convolutional filter to create 4xkxk maps from same features maps. We apply the position based ROI pool to compute KxK array with each element containing boundary box and finally average them.
