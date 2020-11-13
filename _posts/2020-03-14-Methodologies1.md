# Optical Flow
For frames in video, we assume object moves but the intensity of pixel remains same.


<img src="https://latex.codecogs.com/svg.latex?I(x, y, t) = I(x +dx, y+dy, t+dy)">


Now using taylors formula

<img src="https://latex.codecogs.com/svg.latex?I(x +dx, y+dy, t+dy) = I(x, y, t)+ \frac{\delta I}{\delta t} \delta x+ \frac{\delta I}{\delta t}\delta y +\frac{\delta I}{\delta t}\delta t + ...">

Combining the earlier two gives,

<img src="https://latex.codecogs.com/svg.latex?;\frac{\delta I}{\delta t} \delta x+ \frac{\delta I}{\delta t}\delta y +\frac{\delta I}{\delta t}\delta t = 0">

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\delta I}{\delta t} u+ \frac{\delta I}{\delta t}v +\frac{\delta I}{\delta t} = 0">

This show relation between image gradients alone x, y and time axis. The unknowns are *u* and *v*. This requires methods like Mean shift color histogram tracking, Lucas-Kanade methods. It's an optimization problem.

A distinction to keep in mind for recovering motion.

1. *Feature-tracking;* Extract visual features and track them
2. *Optical flow;*  Recover image motion at pixel from spatio-temporal image brightness variations (the brightness assumption, small motion and spatial coherence should maintain).

Solving equation, modified and matrix form from the earlier equation.

<img src="https://latex.codecogs.com/svg.latex?Au = b">

<img src="https://latex.codecogs.com/svg.latex?A^TAu = A^Tb">

Deep learning has some implementation of the optical flow: **FlowNet** and its variations.

1.	FlowNetS:
       -  Simple implementaion
       -  Encoder Decoder layeryer
2. 	EPE/APE
	- Euclidean distance between true and ground truth vector
3. 	FlowNetC
	- Correlated
	- Two similar structure
4.	FlowNet 2.0
	- 1st Layer of FlowNetC
	-  FlowNetCS (Combination of C and S)
	-  Warping
		- Training Dataset: Syntheic data


# Eulerian Video Maginification
A computational technique to visualize the *small* change in video.  A function approximation and magnify the function. Related to fluid mechanics in *Lagrangian Prospective*.  Transforming image into a complex steerable pyramid. Exaggerating the phase variation. Amplify the small motions.

1.	 **Linear Video Magnification:** First-Taylor arguments.

1.1	**1D Translation:** Goal is to motion magnification of the following signal.

  <img  src = "https://latex.codecogs.com/gif.latex?%5Chat%7BI%7D%28x%2C%20t%29%20%3D%20f%28x-%281&plus;%5Calpha%29%5Cdelta%20t%29">

<img src = "https://latex.codecogs.com/gif.latex?%5Chat%7BI%7D%28x%2C%20t%29%20%5Capprox%20f%28x%29-%20%281&plus;%5Calpha%29%5Cdelta%28t%29%20%5Cfrac%7Bf%28x%29%7D%7B%5Cdelta%20x%7D">

 The interesting part are the change

<img src ="https://latex.codecogs.com/gif.latex?B%28x%2C%20t%29%20%3A%3D%20I%28x%20%2Ct%29%20-%20I%28x%2C%200%29">

Using Taylor Expantion

<img src = "https://latex.codecogs.com/gif.latex?I%28x%2C%20t%29%20%5Capprox%20f%28x%29%20-%20%5Cdelta%28t%29%20%5Cfrac%7Bf%28x%29%7D%7B%5Cdelta%20x%7D">

<img src ="https://latex.codecogs.com/gif.latex?B%28x%2C%20t%29%20%5Capprox%20-%20%5Cdelta%28t%29%20%5Cfrac%7Bf%28x%29%7D%7B%5Cdelta%20x%7D">

Now the magnification:
<img src ='https://latex.codecogs.com/gif.latex?%5Chat%7BI%7D%28x%2C%20t%29%20%3D%20I%28x%2C%20t%29%20&plus;%5Calpha%20B%28x%2C%20t%29'>

Amplified factor (1 + <img src="https://latex.codecogs.com/svg.latex?\alpha"> )

1.2 **General case:** Similar like general taylor with amplification factor.

1.3.  **Limitation:**

- Overshoot or undershoot (too large motion causes artifacts)
- Noise amplificatin

Has another better alternative

2. 	**Phase based magnification:** Use of wavelet

2.1 **Simplified Global case:**: Assumption about the functional form of previous function

<img src = "https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Csum_%5Comega%20A_%5Comega%20e%5E%7Bi%5Cphi_%5Comega%7D%20e%5E%7Bi_%5Comega%20x%7D">

Now we get phase difference by using the change in time by <img src="https://latex.codecogs.com/svg.latex?\omega \delta (t)"> which get amplified by amplification factor. Breaking image into local sinusoid using *complex steerable pyramid*.

2.2 **Complex steerable pyramid:** Concept of wavelet and basis functon to localize frequency and space.

2.3 **Phase Shift and Translation:** Related to phase based optical flow.



# Model Compression

[survey 1](https://arxiv.org/pdf/1710.09282.pdf)

Focused on 4 key Contributions
- Parameter pruning/ quantization (drop redundant/uncritical information)
  - Quantization and Binarization
  - Network pruning
  - Structural matrix
- Low-rank Factorization (estimate informative params)
- Transferred or compact convolutional filters (training from scratch)
- Knowledge distillation (From scratch)

Also
- Dynamic capacity network
- Stochastic depth network
Table: Summary [source paper]
<img src ="https://d3i71xaburhd42.cloudfront.net/8dd85e38445a5ddb5dd71cabc3c4246de30c014f/2-TableI-1.png">

[blog 1](https://medium.com/zylapp/deep-learning-model-compression-for-image-analysis-methods-and-architectures-398f82b0c06f)

- [Deep Compression](https://arxiv.org/abs/1510.00149)
  - Network Pruning, Quantization, Huffman encoding

- Weight Quantization method
  - Binarized Neural network [link](https://arxiv.org/abs/1602.02830)
  - Trained ternary Quantization [link](https://arxiv.org/abs/1612.01064)

- SqueezeNet [link](https://arxiv.org/abs/1602.07360)

- MovileNet [v1](https://arxiv.org/abs/1704.04861) [v2](https://arxiv.org/abs/1801.04381)

- SepNet[link](https://arxiv.org/abs/1706.03912)


# Importance Sampling

[medium link](https://towardsdatascience.com/importance-sampling-introduction-e76b2c32e744)
