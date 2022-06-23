# Semi-Supervised

## 2021

1. Assran, Mahmoud, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Armand Joulin, Nicolas Ballas, and Michael Rabbat. "Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples." arXiv preprint arXiv:2104.13963 (2021).
  - PAWS (Predicting view assignments with support samples)
  - Minimize a consistency loss!! different view to get same pseudo labels
  - RQ: can we leverage the labeled data throughout training while also building on advances in self-supervised learning?
  - How it is different than augmentation (may be using some unlabeled counterparts)
  - How the heck the distance between view representation and labeled representation is used to provide weights over class labels (why is makes sense, and what benefits it offers??)
  - Related works: Semi-supervised learning, few shot learning, and self-supervised learning
  - Interesting ways to stop the collapse [sharpening functions] (section 3.2)
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)


## 2020

1. Sohn, Kihyuk, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. "Fixmatch: Simplifying semi-supervised learning with consistency and confidence." arXiv preprint arXiv:2001.07685 (2020).

1. Pham, Hieu, Zihang Dai, Qizhe Xie, Minh-Thang Luong, and Quoc V. Le. "Meta pseudo labels." arXiv preprint arXiv:2003.10580 (2020).
  - semi supervised learning (should cover the labeled and unlabeled at the beginning)
  - teacher network - to generate - pseudo labels [extension of pseudo label work]
    - TP: Contrary to the Pseudo label work: *The teacher network is not fixed* and constantly adapting [claim: teacher learn better pseudo labels]
    - Connected to Pseudo Labels/self-training (semi-supervised)
  - Issues with the confirmation bias of pseudo label? does TP solve this??
    - correct the confirmation bias using a systematic mechanism!!! (how pseudo-label affect the student network) [figure 1]
    - Parallel training [student's learning feedback goes to teacher]!! (dynamic target!)
  - assumption: Pseudo label of teacher can be adjusted
    - However, extremely complication optimization as it requires to unroll everything!
  - Sampling hard pseudo labels (modified version of REINFORCE algorithm!)
  - Nice experiment section: Dataset, network, and baseline
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Li, Junnan, Richard Socher, and Steven CH Hoi. "Dividemix: Learning with noisy labels as semi-supervised learning." arXiv preprint arXiv:2002.07394 (2020).
  - TP:  divide the training data into a labeled set with clean samples and an unlabeled set with noisy samples (co-training two networks), and trains the model on both data (?). Improved MixMatch
  - TP: Two diverged network (avoid confirmation bias of self-training) use each others data! GMM to find labeled and unlabeled (too much noisy) data.  Ensemble for the unlabeled.
  - Related strategy: MixUp (noisy sample contribute less to loss!), co-teaching?  Loss correction approach? Semi-supervised learning, MixMatch (unifies SSL and LNL [consistency regularization, entropy minimization, and MixUp])
  - Application: Data with noisy label (social media image with tag!) may result poor generalization (as may overfit)!
  - Hypothesis: DNNs tend to learn simple patterns first before fitting label noise Therefore, many methods treat samples with small loss as clean ones (discards the sample labels that are highly likely to be noisy! and leverages them as unlabeled data)
  - Algorithm is nice to work with

1. Xie, Qizhe, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. "Self-training with noisy student improves imagenet classification." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10687-10698. 2020.
  - Interesting way to improve the Classifier
  - (labeled data) -> Build classifier (T) -> (predict unlabeled data) -> Train Student using both labeled + model predicted unlabeled data. Repeat.. [algo 1]
  - Introduce noise for both T and S.
    - Data noise, model noise (dropout)


## 2019

1. Zhai, Xiaohua, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. "S4l: Self-supervised semi-supervised learning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1476-1485. 2019.
  - Pretext task of rotation angle prediction!!
    - Rotation, invariant across augmentation
  - Baseline: vitrural adversarial training [inject noise with the original images], EntMin
    - EntMin is bad: because the model can easily become extremely confident by increasing the weights of the last layer

1. Gupta, Divam, Ramachandran Ramjee, Nipun Kwatra, and Muthian Sivathanu. "Unsupervised Clustering using Pseudo-semi-supervised Learning." In International Conference on Learning Representations. 2019.

1. Berthelot, David, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin Raffel. "Mixmatch: A holistic approach to semi-supervised learning." arXiv preprint arXiv:1905.02249 (2019).
  - TP: guesses low-entropy labels for data-augmented unlabeled examples and mixes labeled and unlabeled data using MixUp (Algorithm 1)
  - Related works: Consistency Regularization, Entropy Minimization

## 2018 and earlier

1. Chang, J., Wang, L., Meng, G., Xiang, S., & Pan, C. (2017). Deep adaptive image clustering. In Proceedings of the IEEE international conference on computer vision (pp. 5879-5887).
  - adopts a binary pairwise classification framework for image clustering [DAC]
  - learned label features tend to be one-hot vectors (constraint into DAC)
  - single-stage named Adaptive Learning algorithm to  streamline the learning procedure for image clustering
    - dynamically learn the threshold for pairwise label selection [section 3.3]
  - *summary:* chose two adaptive th for pairwise labels. iterative fix the network and th to reduce number of unlabeled images.  
  - *metrics:* Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) and clustering Accuracy (ACC).

1. Rasmus, A., Berglund, M., Honkala, M., Valpola, H., & Raiko, T. (2015). Semi-supervised learning with ladder networks. Advances in neural information processing systems, 28.
  - trained to simultaneously minimize the sum of sup and unsup cost functions by backpropagation, avoiding the need for layer-wise pre-training
    - builds on top of the Ladder network [bit out of my interest for now]

1. Blum, Avrim, and Tom Mitchell. "Combining labeled and unlabeled data with co-training." In Proceedings of the eleventh annual conference on Computational learning theory, pp. 92-100. 1998.
  - Early semi-supervised learning (mathematical framework with graph setting)
    - Web page (and its augmentation)

1. Grandvalet, Yves, and Yoshua Bengio. "Semi-supervised learning by entropy minimization." In CAP, pp. 281-296. 2005.
  - Semi-supervised learning by minimun entropy regularization!
    - result compared with mixture models ! (entropy methods are better)
    - Connected to cluster assumption and manifold learning
  - Motivation behind supervised training for unlabeled data
    - Exhaustive generative search
    - More parameters to be estimation that leads to more uncertainty
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." arXiv preprint arXiv:1703.01780 (2017).
  - Improves Temporal Ensemble by average model weights (usual practice now!) instead of label prediction (WOW!)
    - Temporal and Pi model suffers from confirmation bias (requires better target) as self-teacher!
  - Two ways to improve: chose careful perturbation or chose careful teacher model
  - Result: Mean teacher is better! faster converge and higher accuracy
  - Importance of good architecture (ThisPaper: Residual networks):
  - TP: how to form better teacher model from students.
  - TP: Large batch, large dataset, on-line learning.

1. Laine, Samuli, and Timo Aila. "Temporal ensembling for semi-supervised learning." arXiv preprint arXiv:1610.02242 (2016).
  - A form of consistency regularization.
  - Self-ensemble (Exponential moving average), consensus prediction of unknown labels!!
    - Ensemble Utilization of outputs of different network-in-training (same network: different epochs, different regularization!!, and input augmentations) to predict unlabeled data.
    - The predicted unlabeled data can be used to train another network
  - One point to put a distinction between semi-supervised learning and representation learning-fine tuning. In Semi-sup the methods uses the label from the beginning.
  - Importance on the Data Augmentation and Regularization.
  - TP: Two self-ensemble methods: pi-model and temporal ensemble (Figure 1) based on consistency losses.
  - Pi Model: Forces the embedding to be together (Contrastive parts is in the softmax portion; prevents collapse)
  - Pi vs Temporal model:
    - (Benefit of temporal) Temporal model is faster.In case of temporal, training target is less noisy.
    - (Downside of temporal) Store auxiliary data! Memory mapped file.


1. Miyato, Takeru, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. "Virtual adversarial training: a regularization method for supervised and semi-supervised learning." IEEE transactions on pattern analysis and machine intelligence 41, no. 8 (2018): 1979-1993.
  - Find the distortion in allowed range for a given input to maximize the loss. (gradient ascent for the input spaces)
    - Network trains to minimize the loss on the distorted input. (Gradient descent in the network parameter spaces)
