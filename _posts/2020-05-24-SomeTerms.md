# Introduction

This blogs contains some of the commonly encountered terms in DL. Lets Move forward. The containing terms are

- [Embedding](#embedding)
- [Collaborative Filter](#collaborative-filter)
- [Boltzmann Machine](#boltzmann-machine)
- [Restricted Boltzmann Machine](#RBM)



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
