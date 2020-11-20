# Introduction

This blogs contains some of the commonly encountered terms in DL. Lets Move forward. The containing terms are

- [Embedding](#embedding)
- [Collaborative Filter](#collaborative-filter)
- [Boltzmann Machine](#boltzmann-machine)
- [Restricted Boltzmann Machine](#RBM)
- [Noise contrastive Estimation](#noise-contrastive-estimation)
- [Active Learning](#active-learning)

# Embedding

Very nice [post](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526) to start.

- Mapping of discrete variables to latent low dimensional continuous space instead of one hot encoding. 3 motivations
  - Nearest neighbors
  - Input to supervision task
  - Visualization
- Learning embedding- Supervised task to learn embedding.

# Collaborative Filter


# Boltzmann Machine

[courtesy: Hinton](https://www.cs.toronto.edu/~hinton/csc321/readings/boltz321.pdf)

- Search Problem
- Learning Problem

## Stochastic dynamics of Boltzmann Machine.

The probabilities of being a neuron to one is <img src="https://latex.codecogs.com/gif.latex?prob(s_i=1)=\frac{1}{1+e^{-z_i}} where z_i=b_i+\sum_js_jw_{ij}">

The interesting Collection: if the units are update sequentially independent of the total inputs, the network will reach Boltzmann/Gibbs distribution. (Google it.) Here comes the notion of energy of state vector. It's also connected to Hopfield network (1982): Described by undirected graph. The states are updated by the connection from the connected nodes. The neural attract or repel each other. The energy is connected with the node values, thresholds and edge weights. The update rule of S_i = 1 if sum of w_ij*s_j > threshold else -1. Following the update rule the energy either decreases or stays same. Different from Hopfield as for the probabilistic approach of weights.

There probabilities of states being in the  v energy states. The probabilities of being a neuron to one is <img src="https://latex.codecogs.com/gif.latex?prob(v)=\frac{\exp^{-E(v)}}{\sum_u\exp{-E(u)}}"> where E(v) represents energy function. The probabilistic state value helps to overcome energy barrier as it goes to new energy state sometimes.

Learning rule <img src="https://latex.codecogs.com/gif.latex?\sum_{v\epsilon data}\frac{\delta \log P(v)}{\delta w_{ij}}=<s_i,s_j>_{data}-<s_i,s_j>_{model}"> and update using gradient ascent. <> is expectation. This causes slow learning as the expectation operation.
Quick notes- data expectation means: Expected value of s_i*s_j in the data distribution and model expectation means sampling state vectors from the equilibrium distribution at a temperature of 1. Two step process firstly sample v and then sample state values.

Convex: When observed data specifies binary state for every unit in the Boltzmann machine.

### Higher order Boltzmann machine

Instead of only two nodes more terms in energy function.

### Conditional Boltzmann Machine

Boltzmann machine learns the data distribution.

### Restricted Boltzmann machine (Smolensky 1986)

A hidden layer and visible layers: Restriction: no hidden-hidden or visible-visible connection. Hidden units are conditionally independent of the visible layers. Now the data expectation can be found by one sweep but the model expectation needs multiple iteration. Introduced reconstruction instead of the total model expectation for s_i*s_j. Learning by contrastive divergence.

Related to markov random field, Gibbs sampling, conditional random fields.

# RBM

[Nice post](https://towardsdatascience.com/restricted-boltzmann-machines-simplified-eab1e5878976)
[by invertor](https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf)

Key reminders: Introducing probabilities in Hopfield network to get boltzman machine. Now one hidden layer makes it RBM with some restriction like no input-output direct connection or hidden-hidden connection. One single layer NN.

Learning rule: Contrastive divergence (approximation of learning rule of (data - model))
<img src = "https://miro.medium.com/proxy/1*cPYfytQ30HP-2rpe_NKqmg.png">

[More cool demonstration](http://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf) for contrastive divergence.

# Learning Vector Quantization

Fairly simple Idea. [Notes](https://towardsdatascience.com/learning-vector-quantization-ed825f8c807d)

Its a prototype-based learning method (Representation).  

- Create Mean to initialize the vector for each class as initial representation.
- Find Distance for each example
  - Update only for the closest prototype (representation) by some distance metrics.
  - If matches with original class (nearest to the actual class of representation) update the representation to go near the example
  - If not agrees than the representation is updated to move away from it.
  - Repeat for each examples to complete epoch1


# Noise Contrastive Estimation
[original Paper](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

## Contrastive loss:

[link 1](https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec)
The idea that same things (positive example) should stay close and negative examples should be in orthogonal position.
- SimCLR ideas
<img src = "https://files.speakerdeck.com/presentations/456108c05a1b416999213faeab0f652d/slide_23.jpg">

[link 2](https://towardsdatascience.com/contrastive-loss-for-supervised-classification-224ae35692e7)

[link 3](https://gombru.github.io/2019/04/03/ranking_loss/) Same ideas different applications.

[initial paper with algorithm](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) with a simpler explanation [resource](https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246).

[concise resource](https://mc.ai/contrasting-contrastive-loss-functions-2/)

[Triplet loss training tricks](https://sites.icmc.usp.br/moacir/p17sibgrapi-tutorial/2017-SIBGRAPI_Tutorial-DLCV-Part2-Regression-Deep-Learning-Siamese_and_Triplet_Nets.pdf)

[some loss function to formulate](https://towardsdatascience.com/contrasting-contrastive-loss-functions-3c13ca5f055e)
  - Max margin losses
  - Triplet loss
  - Multi-class N-pair loss
  - Supervised NT-Xent loss

[visual product similarity - Cornell](https://www.cs.cornell.edu/~kb/publications/SIG15ProductNet.pdf)

# Model Fairness

# Active Learning
[good Start](datacamp.com/community/tutorials/active-learning)
Three ways to select the data to annotate
- Least Confidence
- Margin sampling
- Entropy Sampling

[another source](https://algorithmia.com/blog/active-learning-machine-learning)
Three types
- Stream based selective sampling
  - Model chooses the data required (budget issue)
- Pool-based Sampling (most used)
  - Train with all and retrain with some cases
- Membership query synthesis
  - Create synthetic data to train

[common link resource](https://towardsdatascience.com/active-learning-in-machine-learning-525e61be16e5)
- very basic stuff on what and how to select data.

[very good resource](https://www.kdnuggets.com/2018/10/introduction-active-learning.html)
Which row to label?
- Uncertainty Sampling
  - Least confidence, Margin sampling and Entropy
- Query by committee (aka, QBC)
  - Multiple model trained with same data and find disagreement over the test data!
- Expected Impact
  - Which data samples addition would change the model most! (how!!) to reduce the generalized error
  - look into the input spaces!!
- Density Weighted methods
  - Representative and underlying data distribution consideration.

### Deep learning and Active learning

one of the nicest [introductory blog](https://jacobgil.github.io/deeplearning/activelearning)
- Connection to semi-supervised learning
- ranking image
  - Uncertainty sampling - images that model is not certain about
  - diversity sampling - most diverse example finding
- two issue to combine DL and AL
  - NN are not sure about their Uncertainty
  - It processes data as batch instead of single !
- [key paper- see paperlist](https://arxiv.org/pdf/1703.02910.pdf)
  - use of the dropout to create multiple models and check confidence.
- [BALD paper](https://papers.nips.cc/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf)
  - Additional loss for entropy
- [Learning loss for AL](https://arxiv.org/pdf/1905.03677.pdf)
- Batch aware method [link](https://arxiv.org/pdf/1906.08158.pdf)
- Active learning for the CNN [link](https://arxiv.org/pdf/1708.00489.pdf)
  - Diversity Sampling
-  
