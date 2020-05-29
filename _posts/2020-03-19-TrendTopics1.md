# Trending Topics

---
layout: post
title: "TrendingTopics1"
categories: Hotopics
---

So far In this writing I have covered

- [Representation](#representation-learning)
- [Region based object detectors](#region-based-object-detectors)
- [Region based Convolutional networks](#region-based-fully-convolutional-networks)
- [Deep learning for object detection P2](#deep-learning-for-object-detection-p2)
- [Neural Networks Pruning](#nn-prune)
- [Graph Neural Network](#graphnn)
- [Energy Based Learning](#energy-based-learning)
- [Video activity recognition](#video-activity-recognition)

# 1. Representation Learning
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





# 2. Region based Object Detectors

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

This is a very good point to look back and talk about region proposal methods. This is alternative and better version of sliding window detectors. In [opencv](https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/) tutorial there are several proposal methods
- Objectness
- Constrained parametric min-cuts for automatic object segmentation
- Category independent object proposals
- Randomized Prim
- Selective Search

As things stand, among these methods selective search in the most commonly used.

**What**
Fast and high recall. Computing Hierarchical grouping of similar region based on color, texture, size and shape.

- Start by graph based segmentation method. Over segmentation as seed.
  -  Add bounding box for region proposals
  - Group adjacent segments based on similarity
  - continue to graph based segmentation.

- similarity  
    - color similarity is the closeness of the histogram <img src ="https://latex.codecogs.com/gif.latex?S_{color}(r_i,r_j)=\sum_{k=1}min(c_i^k,c_j^k)">
    - Similar for texture similarity (color instead of texture  in the previous line)
    - Size similarity <img src ="https://latex.codecogs.com/gif.latex?S_{size}(r_i,r_j)=1-\frac{size(r_i)+size(r_j)}{size(im)}"> targets to smaller region merge early.
    - Shape compatibility measures how well they fit in each other.

Final equation stands, weighted sum of previous four findings.

Region proposal method to find Region of interest (ROIs).

#### R-CNN

Cares about 2000 ROIs by using a region proposal method. The regions are warped into fixed size images and feed into a CNN network individually. Then followed by fully connected layers to classify and refine the boundary box.

<img src ="https://miro.medium.com/max/2574/1*Wmw21tBUez37bj-1ws7XEw.jpeg" width="800" height = "200">

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

Overcome the problems of the computational complexity of the selective search by offering region proposal after the CNN layer instead of the original image. Three key steps
- *Anchor boxes*
Needs anchor point in feature maps. This is generated by the CNN network. From point we get boxes by defining aspect ratio and width (Determines number of boxes (no of AR* no of Width)). The boxes needs to be scaled with the original images. As we can see above picture network shrink the features. We can counted this by using stride in the original image. EX. if Feats are downsampled by 4 then from the anchor point use 4 stride in the original image.

![image](https://miro.medium.com/max/1256/1*FifNx4NCyynAZqLjVtB5Ow.png)
![img](https://miro.medium.com/max/1400/1*6JES9IbbxMKrc7NEarQiog.jpeg)

Figure: [Source](https://towardsdatascience.com/region-proposal-network-a-detailed-view-1305c7875853)

- *Classify the anchor boxes*
After defining the boxes we need the information about the box contents (background or foreground). The boxes also need to be fixed for the original objects. Again CNN comes to rescue with two output for both class score and boxes point.

![img](https://miro.medium.com/max/1400/1*hdny9cskat-RjUPuq5ze4A.jpeg)

Figure: [Source](https://towardsdatascience.com/region-proposal-network-a-detailed-view-1305c7875853)

- *Offset for the anchor box to bound the objects*
We still need to learn the size of the box from the anchor points. This is a regression problem with the true boxes by maximizing IoU with the ground truth. It learns the offset to get the actual boxes (kind of fix the mistake by the backbone CNN). This post processing is named Proposal Generation.


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

#### SSD

Inspired by [blog](https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06)

SSD eliminates the necessary for the region proposal networks.

- Faster in time (better FPS)
- Accuracy in lower resolution images
- Multiscale features and default boxes

2 parts
- Extract feature maps (VGG16)
- Apply CNN filter to detect objects (4 object predictions) - Each prediction composes of all score + no_object score.

<img src="https://miro.medium.com/max/1400/1*aex5im2aYcsk4RVKUD4zeg.jpeg">

Figure: [source](https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06) (38x38x512 to 38x38x4x(21+4)) The addition 4 are because of the box coordinates.

<img src="https://miro.medium.com/max/2000/1*mvF9g_zH2DaQK2KgutndFg.jpeg">

Figure: [source](https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06)

Multibox: Making multiple prediction containing boundary boxes with class score. (4 boxes each with class score)

The core advantages of SSD are the multiscale features.

<img src="https://miro.medium.com/max/2000/1*up-gIJ9rPkHXUGRoqWuULQ.jpeg">

Figure: [source paper](https://arxiv.org/pdf/1512.02325.pdf) (At some auxiliary stage SSD takes 6 box predictions)

Default boundary box are similar ideas like anchors in Faster R-CNN. This is the hard part. For this task the model needs to predict different type of boxes shapes. The boundary boxes are chosen manually.

To boxes are annotated as positive/negative based on the IoU of matching boxes with the ground truth.

<img src="https://miro.medium.com/max/1400/1*-KVIXjvBO5m2MQZrzWx-wg.png">

Figure: [source](https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06) Multiscale features and default boundary boxes. Higher resolution maps can detect small objects.



- SSD performs badly for small objects

Loss Function: sum of localization (mismatch between gt and predicted positive matches) loss and confidence loss (Classification loss).

<img src="https://miro.medium.com/max/1400/1*cIE7bbicMOokWQ6w41I-NA.png">

- data mining for negative match as the background selectors. Select a moderate negative example to balance between class imbalances.
- Data augmentation

# 3. Region based Fully Convolutional networks (R-FCN)

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


## Region Proposal Network (RPN) - Backbone of Faster R-CNN

[original post link](https://medium.com/egen/region-proposal-network-rpn-backbone-of-faster-r-cnn-4a744a38d7f9)

![images](https://miro.medium.com/max/413/1*WJsEQnhr4puR4k-29uD80Q.png)

![iamg](https://miro.medium.com/max/450/1*JDQw0RwmnIKeRABw3ZDI7Q.png)

## Feature Pyramid Networks (FPN)

![img](https://miro.medium.com/max/1400/1*UtfPTLB53cR8EathGBOT2Q.jpeg)

Key Idea: Multi-scale Feature map

Top-down and bottom up data structure:

![imga](https://miro.medium.com/max/1400/1*UvGM-OSoebgJDYAjNwX26w.png)
Figure: FPN with RPN (3x3 and 1x1 conv are RPN Head)

Total structure

![img](https://miro.medium.com/max/2000/1*Wvn0WG4XZ0w9Ed2fFYPrXw.jpeg)








# 4. Deep learning for object Detection P2

[original post](https://www.alegion.com/object-detection-part-2)



### R-CNN and Fast R-CNN

![image](https://www.alegion.com/hs-fs/hubfs/1*CP3X0CXeCIUki5NB4cjsTA.png?width=764&name=1*CP3X0CXeCIUki5NB4cjsTA.png)
Figure: R-CNN modules

![image](https://www.alegion.com/hs-fs/hubfs/1*GrqJeumA9QUSxyry6JLMaA.png?width=709&name=1*GrqJeumA9QUSxyry6JLMaA.png)
Figure: Fast R-CNN module

#### Faster R-CNN Architecture

- Region Proposal network
- Feature extraction using CNN
- ROI pooling layer - (Key part)
- Classification and localization

![ig](https://www.alegion.com/hs-fs/hubfs/1*ATyBsAsDQNqT4GYKLrO81w.png?width=759&name=1*ATyBsAsDQNqT4GYKLrO81w.png)
Figure: Building blocks of Fast R-CNN

### Spatial Pyramid pooling

- Pyramid Representation
- Bag-of-words

![img](https://www.alegion.com/hs-fs/hubfs/1*9C-HXo2-AfY-pF6bGwJeOQ.png?width=588&name=1*9C-HXo2-AfY-pF6bGwJeOQ.png)
Figure: network with Spatial pyramid pooling layer

Fixed size constraint comes only in the fully connected layers.

![img](https://www.alegion.com/hs-fs/hubfs/1*Z4MZlKHE0wpkTOk-OQ_U5A-1.png?width=812&name=1*Z4MZlKHE0wpkTOk-OQ_U5A-1.png)

### ROI pooling layer
interesting [blog](https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af)

![img](https://www.alegion.com/hs-fs/hubfs/1*FnDKNXBA0ZSSUczEOFbMFQ.png?width=1079&name=1*FnDKNXBA0ZSSUczEOFbMFQ.png)
*ROI vs SPP*

![img](https://www.alegion.com/hs-fs/hubfs/1*0BJEv0i2OCptYL0PWZve7g.png?width=941&name=1*0BJEv0i2OCptYL0PWZve7g.png)
*Zooming into the network*

ROI (Region of interest) converts proposal networks into a fixed shape required for the fully connected layers in classification.
<img src="https://miro.medium.com/max/1400/1*YZMAa60ycjCzLn5T_HkXgQ.png">
Figure: [Source](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)

ROI takes two inputs
- Feature map from CNN and pooling layers
- It takes indexes and corresponding coordinates for proposal of RPN

ROI pooling layers converts each ROI (regardless of the input feature map or proposal sizes) into a fixed dimension map.


# NN prune
[detail](https://towardsdatascience.com/pruning-deep-neural-network-56cae1ec5505)
TL-DR: original inspiration comes from biological synaptic pruning. In neural network, rank the individual weights and drop p% by setting smaller p's to zero. This is **weight pruning**. The neurons can also be dropped by dropping the neuron itself. This is done by deleting a whole column of weight matrix based on their L2 norms.

[detail](https://jacobgil.github.io/deeplearning/pruning-deep-learning)
TL-DR: Old ideas from Yan Lecun's. The main point concerns about ranking the neurons. The ranking is usually done by L1/L2 norm of weights, activation, or zero occurance of neuron in validation time, etc. After pruning the NN performance drops which is recovered by retraining iteratively.

Not Popular yet because pain of implemenation, Unstability of ranking method and some genius peoples unwillingness to share.

##### Some keypoints with reference
- For CNN the deeper the layer the more it gets pruned [paper from Nvidia](https://arxiv.org/pdf/1611.06440.pdf). Pruning the entire filter. Pruning in each filter or remove some filters entirely. Pruning works better in case of transfer learning.

- Prune the entire convolutional filter. The following layers also need to be cared. The paper used L1 for ranking and removed the lowest m filters and following layers. [Hao Li. et al, UMD and labs america](https://arxiv.org/pdf/1608.08710.pdf)

- The idea is similar to above but the ranking is complex. The validation set performance of the neurons are considered for the ranking assignment for pruning. [paper](https://arxiv.org/pdf/1512.08571.pdf)

- Formalize the combinatorial optimization problem. <img src = "https://latex.codecogs.com/gif.latex?min_w|\mathcal{C(D|W')-C(D|W')}|s.t.\mathcal{||W'||_0}<=B">. Where B is subset of weights. This introduces the notion of loss function in pruning to provide more stable results. [paper from NVIDIA](https://arxiv.org/pdf/1611.06440.pdf)
  - Oracle Pruning: Consider removing each filter and observe the effect. They come up with a Ranking method based on first order Taylor expansion of the cost function. Two subsequent point differ by presence of a filter. The ranking of a particular filer h can be expressed as
  <img src = "https://latex.codecogs.com/gif.latex?\Theta_{TE}(h_i)=|\Delta\mathcal{C}(h_i)|=|\Delta\mathcal{C(D,}h_i)-\frac{\delta\mathcal{C}}{\delta h_i}h_i-\Delta\mathcal{C(D,}h_i)|=|\frac{\delta\mathcal{C}}{\delta h_i}h_i|"> and <img src = "https://latex.codecogs.com/gif.latex?\Theta_{TE}(z_l^{(k)})=|\frac{1}{M}\sum_m\frac{\delta C}{\delta z_{l,m}^{(k)}}z_{l,m}^{(k)}|">. This would provide the rank of the layer after L2 norm.

# GraphNN

Very nice [survey](https://arxiv.org/pdf/1812.08434.pdf)

Motivation
  - CNN
  - Graph Embedding
  - Non Euclidean

Models:
  -

# Energy Based learning

[key tutorial](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
### Introduction

- Capture Dependencies between variables
- Associate scalar energy (measure of capability) to each variable configuration.
- Interferce: Finding remain variables values based on given observed variables to minimize energy Function
- Learning: Associate low energies to *correct values* of remaining variables.  
- Loss function: measure the quality of energy function.
- Unification between probabilitics and non-probabilitic methods.
- <img src = "https://latex.codecogs.com/gif.latex?Y^*=argmin_{Y\epsilon\mathcal{Y}}E(X,Y)"> - Compute energy for all possible Y!
- Energy function has many possible Form
- Appropriate Question
    - Which Y most compatible with X
    - is Y_1 or Y_2 more compatible with X
    - is Y compatible with X
    - what is conditional prob. dist. over space of Y given X.
- Cares only about lower energy for correct answer! problem with combining other variables. So, energy to probabilities again!
- <img src = "https://latex.codecogs.com/gif.latex?P(X|Y)=\frac{e^{-\beta E(X,Y))}}{\sum_{Y\epsilon \mathcal Y}e^{-\beta E(X,Y)}}"> ; beta akin to inverser temrature, Gibbs distribution.

### training
- <img src = "https://latex.codecogs.com/gif.latex?\mathcal E={E(W,X,Y):W\epsilon\mathcal W}"> Same old parameters, can be parameters to NN
- So the target becomes <img src = "https://latex.codecogs.com/gif.latex?W^*=min_{W\epsilon \mathcal W}\mathcal L(W,\mathcal S)"> Where S is the total dataset. Can be expanded to traditional instance based formula.

- IN summary Four components
  - Architecture: E(W,Y,X)
  - Interference algorithm: Method of finding Y to minimize the E(W,X,Y)
  - Loss function: Measure the quality of energy function based on the training data
  - Learning algorithm: fining W that minimize the loss from a family of energy functions <img src = "https://latex.codecogs.com/gif.latex?\mathcal E">
- Loss function
  - Energy loss - Value of E(X_i, Y_i, W)
  - Generalized Perceptron Loss - E(X_i, Y_i, W) - min_y E(W, Y, X_i)
  - Hinge loss - max(0, m + E(X_i, Y_i, W) - min_{Y_j/=Y_i}E(X_i, Y_i, W))
  - Log loss - log(1+ e^{E(X_i, Y_i, W - min_{Y_j/=Y_i}E(X_i, Y_i, W))}
  - MCE loss = f(E(X_i, Y_i, W - min_{Y_j/=Y_i}E(X_i, Y_i, W))) -  f may be step function
  - Square - Square Loss function: <img src = "https://latex.codecogs.com/gif.latex?E(W,X_i,Y_i)^2-(max(0,min_{Y_j\ne Y_i}E(W,X_i,Y_j)))">
  - Square exponential loss = <img src = "https://latex.codecogs.com/gif.latex?E(W,X_i,Y_i)^2-\gamma e^{min_{Y_j\ne Y_i}E(W,X_i,Y_j))}">

- Simple Architecture:
  - regressor
  - two class classifiers
  - Multiclass classifier
  - Implicit regression
- Latent Variable Architecture
  - <img src = "https://latex.codecogs.com/gif.latex?E(Y,X)=min_{Z\epsilon\mathcal(Z)}E(Z,Y,X)">

- Analysis of Loss function for EBM
  - Architecture and loss function connection: Some loss works with some architecture... Contrastive terms (hinge, log and MCE) helps in complicated architecture.


- Sufficient condition for Good Loss functions

- conditions on the Energy
  1. <img src = "https://latex.codecogs.com/gif.latex?E(W,Y^i,X^i)<E(W,Y,X^i), Y^i is GT and Y\neq Y^i">
  1. <img src = "https://latex.codecogs.com/gif.latex?E(W,Y^i,X^i)<E(W,\bar Y,X^i)-m, Y^i is GT and Y\neq Y^i"> Y bar is the minimum energy among the non GT solution.
  1. Existance of a point where energy for all other non GT solution are smaller than all the points outside the margin. (sufficient condition - without mentioning maths)

  <img src="https://i.stack.imgur.com/VHZzi.png">

- Efficient Interference

# Video Activity recognition

[informative resource](http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds08.pdf)
[blog post](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review)


Approaches

- Pre-deep Learning
  - Local features: HOG and Histogram of optical flow
  - Trajectory based: Motion boundary histogram
  - Feature aggregation: Bag of visual wordd and fisher vectors
  - Representing motion: Optical Flow and trajectory stacking
  - 3 key steps:
    - Local high dimensional feature, combine features, SVM classifiers

- Deep Learning
  - Fuse features from multiple frames: Single frame, late fusion, early fusion, slow fusion.
  - Single stream network: Single frame, late fusion, early fusion, slow fusion.
    - problem with Motion
    - Detailed features for the diverse dataset
  - Two stream Networks
    - Hypothesis: Video = Appearance + Motion
    - Special fusion, temporal fusion
    - Problems with long range features
    - Precomputed optical flow!
  - Multi resolution: High res fovea stream and low-res image context stream

- CNN+RNN
  - Video as sequence
  - Design choice: Modality (RGB and flow), features (CNN or hand crafted), Temporal aggregation (temporal pooling and RNN)
  - global discriminator

- 3D convolution
  - Spatio temporal features

Contemporary works based on single and two stream papers
1. LRCN (long term recurrent Convolutional network for visual recognition and description, Donahue et al 2014)
- Contribution
  - Based on RNN (not stream)
  - Encoder-decoder for video presentation
  - End2End training (but use flow!)
2. C3D (Learning spatiotemporal features with convolutional networks (Du tran et al 2014)
- Contributions
  - 3D CNN as feature extractors
  - Extensive search for best 3D cnn kernel and architecture.
  - Using deconvolutional layers for interpretation.
  - Factorized Spatio-temporal CN

3. Conv3D and attention (Yao et al, 2015)
- Contributions
  - Novel 3D CNN-RNN encoder-decoder for spatiotemporal
  - Use of attention within CNN-RNN encoder decoder frameworks.
- Not actually action recognition but cnn+lstm ..

4. TwoStreamFusion (Feichtenhofer et al. 2016)
- Contribution
  - Long range temporal modeling and better long range losses
  - Multi-level fuses architecture

5. TSN (Temporal segment Networks: Wang et al 2016)
- Contribution
  - Long range temporal Modeling
  - Bath norm, dropout and pretrained

6. ActionVLAD (Girdhar et al 2017)
- Contributions:
  - learnable video level aggregation of features (!) - Pooling from different regions
  - End2end training

7. HiddenTwoStream (Zhu et al 2017)
- Contributions
  - Novel architecture for optical flow input using separate network
  - Spatial stream CNN, parallel with MotionNet (for optical flow) and Temporal Stream CNN then late fusion.

8. I3D (Carreira et al. 2017)
- Contributions
  - 3D based model into two stream architecture
  - New dataset
  - Extension from C3D (2.)

9. T3D (Diba et al. 2017)
- Contributions
 - Combining temporal information across variable depth
 - Supervised transfer learning


Some Dataset
- Video classification
  - UCF101
  - Sports-1M
  - Youtube 8M

- Atomic action
  - Charades
  - Atomic Visual Actions
  - Moments in Time

- Movie Querying
  - M-VAD and MPII-MD
  - Large scale movie description Challenges (LSMDC)
