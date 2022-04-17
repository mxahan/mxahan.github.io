# Self-Supervised Learning

## 2015

1. Dosovitskiy, Alexey, Philipp Fischer, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. "Discriminative unsupervised feature learning with exemplar convolutional neural networks." IEEE transactions on pattern analysis and machine intelligence 38, no. 9 (2015): 1734-1747.

1. Doersch, Carl, Abhinav Gupta, and Alexei A. Efros. "Unsupervised visual representation learning by context prediction." In Proceedings of the IEEE international conference on computer vision, pp. 1422-1430. 2015.
  - Crop position learning pretext!
  - Figure 2 (problem formulation) and 3 (architectures) shows the key contribution

1. Tishby, Naftali, and Noga Zaslavsky. "Deep learning and the information bottleneck principle." In 2015 IEEE Information Theory Workshop (ITW), pp. 1-5. IEEE, 2015.

1. Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." In ICML deep learning workshop, vol. 2. 2015.
  - Original contrastive approach with two (either similar or dissimilarity) images [algorithmic]

1. Wang, Xiaolong, and Abhinav Gupta. "Unsupervised learning of visual representations using videos." In Proceedings of the IEEE international conference on computer vision, pp. 2794-2802. 2015.
  - Visual tracking provides the supervision!!! [sampling method for CL]
  - Siamese-triplet networks: energy based max-margin loss
  - Experiments: VOC 2012 dataset (100k Unlabeled videos)
  - interesting loss functions (self note: please update the pdf files)



## 2016

1. Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis." In International conference on machine learning, pp. 478-487. PMLR, 2016.
  - Deep Embedded Clustering (DEC) Learns (i) Feature representation (ii) cluster assignments
  - Experiment: Image and text corpora
  - Contribution: (a) joint optimization of deep embedding and clustering; (b) a novel iterative refinement via soft assignment (??); (c) state-of-the-art clustering results in terms of clustering accuracy and speed
  - target distribution properties: (1) strengthen predictions (i.e., improve cluster purity), (2) put more emphasis on data points assigned with high confidence, and (3) normalize loss contribution of each centroid to prevent large clusters from distorting the hidden feature space.
  - Computational complexity as iteration over large data samples

1. Joulin, Armand, Laurens Van Der Maaten, Allan Jabri, and Nicolas Vasilache. "Learning visual features from large weakly supervised data." In European Conference on Computer Vision, pp. 67-84. Springer, Cham, 2016.

1. Sohn, Kihyuk. "Improved deep metric learning with multi-class n-pair loss objective." In Proceedings of the 30th International Conference on Neural Information Processing Systems, pp. 1857-1865. 2016.
  - Deep metric learning (solves the slow convergence for the contrastive and triple loss)
    - what is the penalty??
    - How they compared the convergences
  - This paper: Multi-class N-pair loss
    - developed in two steps (i) Generalization of triplet loss (ii) reduces computational complexity by efficient batch construction (figrue 2) taking (N+1)xN examples!!
  - Experiments on visual recognition, object recognition, and verification, image clustering and retrieval, face verification and identification tasks.
  - identify multiple negatives [section 3], efficient batch construction
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Noroozi, Mehdi, and Paolo Favaro. "Unsupervised learning of visual representations by solving jigsaw puzzles." In European conference on computer vision, pp. 69-84. Springer, Cham, 2016.
  - Pretext tasks (solving jigsaw puzzle) - self-supervised

1. Misra, Ishan, C. Lawrence Zitnick, and Martial Hebert. "Shuffle and learn: unsupervised learning using temporal order verification." In European Conference on Computer Vision, pp. 527-544. Springer, Cham, 2016.
  - Pretext Task: a sequence of frames from a video is in the correct temporal order (figure 1) [sampling method for CL]
    - Capture temporary variations
    - Fusion and classification [not the CL directly]
  - experiment Net: CNN Based network, data: UCF101, HMDB51 & FLIC, MPII (pose Estimation)
  - self note: There's more.

1. Oh Song, Hyun, Yu Xiang, Stefanie Jegelka, and Silvio Savarese. "Deep metric learning via lifted structured feature embedding." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4004-4012. 2016.
  - Proposes a different loss function (Equation-3)
    - Non-smooth and requires special data mining
    - Solution: This paper: optimize upper bound of eq3, instead of mining use stochastic approach!!
    - This paper: all pairwise combination in a batch!! O(m<sup>2</sup>)
      - uses mini batch
      - Not-random batch formation: Importance Sampling
      - Hard-negative mining
      - Gradient finding in the algorithm-1, mathematical analysis
  - Discusses one of the fundamental issues with contrastive loss and triplet loss!
    - different batches puts same class in different position
  - Experiment: Amazon Object dataset- multiview .
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

## 2017

1. Santoro, Adam, David Raposo, David GT Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and Timothy Lillicrap. "A simple neural network module for relational reasoning." arXiv preprint arXiv:1706.01427 (2017).

1. Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Split-brain autoencoders: Unsupervised learning by cross-channel prediction." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1058-1067. 2017.
  - Extension of autoencoders to cross channel prediction [algorithmic]
    - Predict one portion to other and vice versa + loss on full reconstruction.
    - Two disjoint auto-encoders.
  - Tried both the regression and classification loss
  - Section 3 sums it up
    - Cross-channel encoders
    - Split-brain autoencoders.

1. Fernando, Basura, Hakan Bilen, Efstratios Gavves, and Stephen Gould. "Self-supervised video representation learning with odd-one-out networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3636-3645. 2017.
  - Pretext tasks: Finding the odd-one (O3N) video using fusion. [sampling method for CL]
    - Temporal odd one! Target: Regression task
  - Network: CNN with fusion methods.
  - Experiments: HMDB51, UCF101

1. Denton, Emily L. "Unsupervised learning of disentangled representations from video." In Advances in neural information processing systems, pp. 4414-4423. 2017.
  - Encoder-Decoder set up for the disentangled [disentanglement representation]
  - Hypothesis: Content (time invariant) and Pose (time variant)
  - Two Encoders for the pose and content; Concatenate the output for single Decoder
  - Introduce adversarial loss
  - Video generation conditioned on context, and pose modeling via LSTM.


## 2018

1. Aljalbout, Elie, Vladimir Golkov, Yawar Siddiqui, Maximilian Strobel, and Daniel Cremers. "Clustering with deep learning: Taxonomy and new methods." arXiv preprint arXiv:1801.07648 (2018).
  - Three components: Main encoder networks (concerns with architecture, Losses, and cluster assignments)
  - Non-cluster loss: Autoencoder reconstruction losses
  - Various types of clustering loss (note)
  - Combine losses: Pretraining / jointly training / variable scheduleing
  - Cluster update: Jointly update with the network model / Alternating update with network models
  - Relevant methods: Deep Embedded Clustering (Xie et al, 2016), Deep Clustering Network (yang et al, 2016), Discriminatively Boosted Clustering (Li et al, 2017), ..

1. Belghazi, Mohamed Ishmael, Aristide Baratin, Sai Rajeshwar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and Devon Hjelm. "Mutual information neural estimation." In International Conference on Machine Learning, pp. 531-540. 2018.

1. Hjelm, R. Devon, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. "Learning deep representations by mutual information estimation and maximization." arXiv preprint arXiv:1808.06670 (2018).
  - locality of input knowledge and match prior distribution adversarially (DeepInfoMax)
    - Maximize input and output MI
  - Experimented on Images
    - Compared with VAE, BiGAN, CPC
  - Evaluate represenation by Neural Dependency Measures (NDM)
  - Global features (Anchor, Query) and Local features of the query (+), local feature map of random images (-)
  - [personal note](https://github.com/mxahan/PDFS_notes/blob/master/deepinfomax_paper.pdf)

1. Wu, Zhirong, Alexei A. Efros, and Stella X. Yu. "Improving generalization via scalable neighborhood component analysis." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 685-701. 2018.
 - SCNA (Modified version of the NCA approach)
 - Aim to reduce the computation complexity of the NCA by taking two assumptions
  - Partial gradient update making the mini-batch gradient update possible
  - TP: Device a mechanism called augmented memory for the generalization. (a version of momentum update!)

1. Caron, Mathilde, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. "Deep clustering for unsupervised learning of visual features." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 132-149. 2018.
  - Cluster Deep features and make them pseudo labels. [fig 1]
  - Cluster (k-means) for training CNN [Avoid trivial solution of all zeros!]
  - Motivation from Unsupervised feature learning, self-supervised learning, generative model
  - [More](https://github.com/facebookresearch/deepcluster)

1. Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. "Representation learning with contrastive predictive coding." arXiv preprint arXiv:1807.03748 (2018).
  - Predicting the future [self-supervised task design]
    - derive the concept of context vector (from earlier representation)
      - use the context vector for future representation prediction.
  - TP: Great works with some foundation of CL
  - probabilistic (AR) contrastive loss!!
    - in latent space
  - Experiments on the speech, image, text and RL
  - CPC (3 things)  - Aka- InfoNCE (coining the term)
    - compression, autoregressive and NCE
  - Energy based like setup
  - Figure 4: about what they did!
  - [more notes](https://github.com/mxahan/PDFS_notes/blob/master/cpc_2017.pdf)

1. Wu, Zhirong, Yuanjun Xiong, Stella X. Yu, and Dahua Lin. "Unsupervised feature learning via non-parametric instance discrimination." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3733-3742. 2018.
  - non-parametric classifier via feature representation **(Memory Bank)**
  - Memory bank stores instance features (used for kNN classifier)
    - Dimention reduction: one of the key [algorithmic] contribution
  - Experiments
    - obj detect and image classification
  - connect to
    - selfsupervised learning (related works) and Metric learning (unsupervised fashion)
    - NCE (to tackle class numbers) - [great idea, just contrast with everything else in E we get the classifier]
  - instance-level discrimination, non-parametric classifier.
    - compared with known example (non-param.)
  - interesting setup section 3
    - representation -> class (image itself) (compare with instance) -> loss function (plays the key role to distinguish)
    - NCE from memory bank
    - Monte carlo sampling to get the all contrastive normalizing value for denominator
    - proximal parameter to ensure the smoothness for the representations {proximal regularization:}

1. Sermanet, Pierre, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. "Time-contrastive networks: Self-supervised learning from video." In 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 1134-1141. IEEE, 2018.
  - Multiple view point [same times are same, different time frames are different], motion blur, viewpoint invariant
    - Regardless of the viewpoint [same time same thing , same representation]
    - Considered images [sampling method for CL]
    - Representation is the reward
    - TCN - a embedding {multitask embedding!}
  - imitation learning
  - PILQR for RL parts
  - Huber-style loss


## 2019


1. Dwibedi, Debidatta, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. "Temporal cycle-consistency learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1801-1810. 2019.
  - Temporal video alignment problem: the task of finding correspondences across multiple videos despite many factors of variation
  - TP: Temporal cycle consistency losses (complementary to other methods [TCN, shuffle and learn])
  - Dataset: Penn AR, Pouring dataset
  - TP: Focus on temporal reasoning!! (metrics)
    - learns representations by aligning video sequences of the same action
    - Requires differentiable cycle-consistency losses
  - Figure 2: interesting but little hard to understand

1. Li, Xueting, Sifei Liu, Shalini De Mello, Xiaolong Wang, Jan Kautz, and Ming-Hsuan Yang. "Joint-task self-supervised learning for temporal correspondence." arXiv preprint arXiv:1909.11895 (2019).

1. Xie, Qizhe, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le. "Unsupervised data augmentation for consistency training." arXiv preprint arXiv:1904.12848 (2019).
  - TP: how to effectively noise unlabeled examples (1) and importance of advanced data augmentation (2)
    - Investigate the role of noise injection and advanced data augmentation
    - Proposes better data augmentation in consistency training: Unsupervised Data Augmentation (UDA)
    - Experiments with vision and language tasks
  - Bunch of experiment with six language tasks and three vision tasks.
  - Consistency training as regularization.
  - UDA: Augment unlabeled data!! and quality of the noise for augmentations.
  - Noise types: Valid noise, Diverse noise, and Targeted Inductive biases
  - Augmentation types: RandAugment for image, backtranslating the language
  - Training techniques: confidence based masking, Sharpening Predictions, Domain relevance data filtering.
  - Interesting graph comparison under three assumption.
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Anand, Ankesh, Evan Racah, Sherjil Ozair, Yoshua Bengio, Marc-Alexandre Côté, and R. Devon Hjelm. "Unsupervised state representation learning in atari." arXiv preprint arXiv:1906.08226 (2019).

1. Alwassel, Humam, Dhruv Mahajan, Bruno Korbar, Lorenzo Torresani, Bernard Ghanem, and Du Tran. "Self-supervised learning by cross-modal audio-video clustering." arXiv preprint arXiv:1911.12667 (2019).

1. Sun, Chen, Fabien Baradel, Kevin Murphy, and Cordelia Schmid. "Learning video representations using contrastive bidirectional transformer." arXiv preprint arXiv:1906.05743 (2019).

1. 1. Han, Tengda, Weidi Xie, and Andrew Zisserman. "Video representation learning by dense predictive coding." In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pp. 0-0. 2019.
  - Self-supervised AR (DPC)
    - Learn dense coding of spatio-temporal blocks by predicting future frames (decrease with time!)
    - Training scheme for future prediction  (using less temporal data)
  - Care for both temporal and spatial negatives.
  - Look at their case for  - the easy negatives (patches encoded from different videos), the spatial negatives (same video but at different spatial locations), and the hard negatives (TCN)
  - performance evaluated by Downstream tasks (Kinetics-400 dataset (pretrain), UCF101, HMDB51- AR tasks)
  - Section 3.1 and 3.2 are core (contrastive equation - 5)

1. Ye, Mang, Xu Zhang, Pong C. Yuen, and Shih-Fu Chang. "Unsupervised embedding learning via invariant and spreading instance feature." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6210-6219. 2019.
  - Contrastive idea but uses siamese network.

1. Bachman, Philip, R. Devon Hjelm, and William Buchwalter. "Learning representations by maximizing mutual information across views." arXiv preprint arXiv:1906.00910 (2019).
  - Multiple view of shared context
  - Why MI (analogous to human representation, How??) - same understanding regardless of view
    - Whats the problem with others !!
  - Experimented with Imagenet
  - Extension of local DIM in 3 ways (this paper calls it - augmented multi-scale DIM (AMDIM))
    - Predicts features for independently-augmented views
    - predicts features across multiple views
    - Uses more powerful encoder
  - Methods relevance: Local DIM, NCE, Efficient NCE computation, Data Augmentation, Multi-scale MI, Encoder, mixture based representation

1. Goyal, Priya, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. "Scaling and benchmarking self-supervised visual representation learning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6391-6400. 2019.

1. Chen, Xinlei, Haoqi Fan, Ross Girshick, and Kaiming He. "Improved baselines with momentum contrastive learning." arXiv preprint arXiv:2003.04297 (2020).

1. Tian, Yonglong, Dilip Krishnan, and Phillip Isola. "Contrastive multiview coding." arXiv preprint arXiv:1906.05849 (2019).
  - Find the invariant representation
  - Multiple view of objects (image) (CMC) - multisensor view or same object!!]
    - Same object but different sensors (positive keys)
    - Different object same sensors (negative keys)
    - Experiment: ImageNEt, STL-10, two views, DIV2K cropped images
  - Positive pairs (augmentation)
  - Follow-up of Contrastive Predictive coding (no RNN but more generalized)
  - Compared with baseline: cross-view prediction!!
  - Interesting math: Section 3 and experiment 4
    - Mutual information lower bound Log(k = negative samples)- L<sub>contrastive</sub>
    -  Memory bank implementation

1. Zhuang, Chengxu, Alex Lin Zhai, and Daniel Yamins. "Local aggregation for unsupervised learning of visual embeddings." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6002-6012. 2019.

1. Tschannen, Michael, Josip Djolonga, Paul K. Rubenstein, Sylvain Gelly, and Mario Lucic. "On mutual information maximization for representation learning." arXiv preprint arXiv:1907.13625 (2019).

1. Löwe, Sindy, Peter O'Connor, and Bastiaan S. Veeling. "Putting an end to end-to-end: Gradient-isolated learning of representations." arXiv preprint arXiv:1905.11786 (2019).

1. Tian, Yonglong, Dilip Krishnan, and Phillip Isola. "Contrastive representation distillation." arXiv preprint arXiv:1910.10699 (2019).
  - Missed structural knowledge of the teacher network!!
  - Cross modal distillation!!
  - KD -all dimension are independent (intro 1.2)
    - Capture the correlation/higher-order dependencies in the representation (how??).
    - Maximize MI between teacher and student.
  - Three scenario considered [fig 1]
  - KD and representation learning connection (!!)
  - large temperature increases entropy  [look into the equation! easy-pesy]
  - interesting proof and section 3.1 [great!]
  - Student takes query - matches with positive keys from teacher and contrast with negative keys from the teacher network.
  - Equation 20 is cross entropy (stupid notation)
  - Key contribution: New loss function: 3.4 eq21
  - [notes](https://github.com/mxahan/PDFS_notes/blob/master/contrastive_representation_distillation.pdf)

1. Saunshi, Nikunj, Orestis Plevrakis, Sanjeev Arora, Mikhail Khodak, and Hrishikesh Khandeparkar. "A theoretical analysis of contrastive unsupervised representation learning." In International Conference on Machine Learning, pp. 5628-5637. 2019.
  - present a framework for analyzing CL (is there any previous?)
    - introduce latent class!! shows generalization bound.
  - Unsupervised representation learning
  - TP: notion of latent classes (downstream tasks are subset of latent classes), rademacher complexity of the architecture! (limitation of negative sampling), extension!
  - CL gives representation learning with plentiful labeled data! TP asks this question. \
  - Theoretical results to include in the works.

1. Anand, Ankesh, Evan Racah, Sherjil Ozair, Yoshua Bengio, Marc-Alexandre Côté, and R. Devon Hjelm. "Unsupervised state representation learning in atari." arXiv preprint arXiv:1906.08226 (2019).

1. Kolesnikov, Alexander, Xiaohua Zhai, and Lucas Beyer. "Revisiting self-supervised visual representation learning." In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 1920-1929. 2019.
  - Insight about the network used for learning [experimentation]
  - Challenges the choice of different CNNs as network for vision tasks.
  - Experimentation with different architectures [ResNet, RevNet, VGG] and their widths and depths.
  - key findings : hat (1) lessons from architecture design in the fully supervised setting do not necessarily translate to the self-supervised setting; (2) contrary to previously popular architectures like AlexNet, in residual architectures, the final prelogits layer consistently results in the best performance; (3) the widening factor of CNNs has a drastic effect on performance of self-supervised techniques and (4) SGD training of linear logistic regression may require very long time to converge
  - pretext tasks for self-supervised learning should not be considered in isolation, but in conjunction with underlying architectures.

1. Hendrycks, Dan, Mantas Mazeika, Saurav Kadavath, and Dawn Song. "Using self-supervised learning can improve model robustness and uncertainty." arXiv preprint arXiv:1906.12340 (2019).
  - TP: self-supervision can benefit robustness in a variety of ways, including robustness to adversarial examples, label corruption, and common input corruptions (how is this new!!)
    - Interesting problem and experiment setup for each of the problems
  - Besides a collection of techniques allowing models to catch up to full supervision, SSL is used two in conjunction of providing strong regularization that improves robustness and uncertainty estimation

## 2020

1. Sohoni, Nimit, Jared Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. "No subclass left behind: Fine-grained robustness in coarse-grained classification problems." Advances in Neural Information Processing Systems 33 (2020): 19339-19352.
  - Hidden stratification: unavailable subclass labels
  - TP: GEORGE, a method to both measure and mitigate hidden stratification even when subclass labels are unknown.
    - TP: estimate subclass labels for the training data via clustering techniques (Estimation)
    - use these approximate subclass labels as a form of noisy supervision in a distributionally robust optimization objective (exploiting)
  - Paper construction: generative background for data labeling process [figure 2]
  - Reason of hidden stratification: Inherent hardness and Dataset imbalance
  - Method overview [figure 4]: includes two step training (i. training with classification, dimentional reduction of last layer and ii. fine tune.)

1. Patacchiola, Massimiliano, and Amos Storkey. "Self-supervised relational reasoning for representation learning." arXiv preprint arXiv:2006.05849 (2020).
  - Relation network head instead of direct contrastivie loss [architectural][Pretext task design]
  - differentiate from previous one in several ways:
    - (i) TP use relational reasoning on unlabeled data (previously on unlabeled data);
    - (ii) here TP focus on relations between different views of the same object (intra-reasoning) and between different objects in different scenes (inter-reasoning); [previously withing scene]
    - (iii) in previous work training the relation head was the main goal, here is a pretext task for learning useful representations in the underlying backbone
  - Related works: pretext task, metric learning, CL, pseudo labeling, InfoMax,
  - Intra-inter reasoning increases mutual information.
  - Easy-pesy loss functions to minimize (care about the sharping of the weight of the loss function!)

1. Huang, Zhenyu, Peng Hu, Joey Tianyi Zhou, Jiancheng Lv, and Xi Peng. "Partially view-aligned clustering." Advances in Neural Information Processing Systems 33 (2020).

1. Zhu, Benjin, Junqiang Huang, Zeming Li, Xiangyu Zhang, and Jian Sun. "EqCo: Equivalent Rules for Self-supervised Contrastive Learning." arXiv preprint arXiv:2010.01929 (2020).
  - Theoretical paper: Challenges the large number of negative samples [algorithmic]
    - Though more negative pairs are usually reported to derive better results, Interpretation: it may be because the hyper-parameters in the loss are not set to the optimum according to different numbers of keys respectively
    - Rule to set the hyperparameters
    - SiMo: Alternate for the InfoNCE
  - EqCo: Concept of Batch Size (N) and Negative Sample number (K)
    - CPC work follow-up
    - the learning rate should be adjusted proportional to the number of queries N per batch
    - linear scaling rule needs to be applied corresponding to N rather than K
  - SiMo: cancel the memory bank as rely on a few negative samples per batch. Instead, use the momentum encoder to extract both positive and negative key embeddings from the current batch.

1. Miech, Antoine, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. "End-to-end learning of visual representations from uncurated instructional videos." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9879-9889. 2020.
  - Experiments with Dataset: HowTo100M (task progression) [video with narration]
  - This paper: (multi-instance learning) MIL-NCE (to address misalignment in video narration)!! HOW? Requires instructional videos! (video with text)
  - Video representation learning: shows effectiveness in four downstream tasks: AR (HMDB, UCF, Kinetics), action localization (Youtube8M, crosstask), action segmentation (COIN), Text-to-video retrieval (YouCook2, MSR-VTT).
  - TP: Learn video representation from narration only (instructional video)!
  - Related works: Learning visual representation from unlabeled videos. (ii) Multiple instance learning for video understanding (MIL) [TP: connect MCE with MIL]
  - Two networks in the NCE Calculation : [unstable targets!!]
  - Experiments: Network: 3D CNN

1. Xie, Zhenda, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, and Han Hu. "Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning." arXiv preprint arXiv:2011.10043 (2020).
  - Argue that instance level CL reaches suboptimal representation!
    - Can we go better.
  - Alternative to instance-level pretext learning - Pixel-level pretext learning!
    - Pixel level pretext learning! pixel to propagation consistency!!
      - Avail both backbone and head network! to reuse
      - complementary to instance level CL
      - How to define pixel level pretext tasks!
    - Why instance-label is sub-optimal? How? Benchmarking!
    - Dense feature learning
  - Application: Object detection (Pascal VOC object detection), semantic segmentation
  - Pixel level pretext tasks
    - Each pixel is a class!! what!!
      - Features from same pixels are same !
    - PixContrast: Training data collected in self-supervised manner
    - requires pixel feature vector !
    - Feature map is warped into the original image space
    - Now closer pictures together and .... contrastive setup
    - Learns spatially sensitive information
  - Pixel-to-propagation consistency !! (pixpro)
    - positive pair obtaining methods
    - asymmetric pipeline
    - Learns spatial smoothness information
      - Pixel propagation module
      - pixel to propagation consisency loss
    - PPM block (Equation: 3,4): Figure 3

1. Piergiovanni, A. J., Anelia Angelova, and Michael S. Ryoo. "Evolving losses for unsupervised video representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 133-142. 2020.
  - video representation learning! (generic and transfer) (ELo)
    - Video object detection
  - Zeroshot and fewshot AcRecog
  - introduce concept of loss function evolving (automatically find optimal combination of loss functions capturing many (self-supervised) tasks and modalities)
      - using an evolutionary search algorithm (!!)
  - Evaluation using distribution matching and Zipf's law!
  - only outperformed by fully labeled dataset
  - This paper: new unsupervised learning of video representations from unlabeled video data.
    - multimodal, multitask, unsupervised learning
      - Youtube dataset
    - combination of single modal tasks and multi-modal tasks
    - too much task! how they combined it!! Engineering problem
      - evolutionary algorithm to solve these puzzle
      - Power law constraints and KL divergence
    - Evolutionary search for a loss function that automatically combines self-supervised and distillation task
    - unsupervised representation evaluation metric based on power law distribution matching
  - Multimodal task, multimodal distillation
    - show their efficacy by the power distribution of video classes (zipf's law)
  - Figure 2:
    - overwhelming of tasks! what if no commonalities.
  - Hypo: Synchronized multi-modal data source should benefit representation learning
    - Distillation losses between the multiple streams of networks + self supervised loss
  - Methods
    - Multimodal learning
    - Evolving an unsupervised loss function
      - [0-1] constraints
      - zipfs distribution matching
        - Fitness measurement - k-means clustering
        - use smaller subset of data for representation learning
        - Cluster the learned representations
        - the activity classes of videos follow a Zipf distribution
        - HMDB, AVA, Kinetics dataset, UCF101
      - ELo methods with baseline weakly-supervised methods
      - Self-supervised learning
        - reconstruction (encoder-decoder) and prediction tasks (L2 distance minimization)
        - Temporal ordering
        - Multi-modal contrastive loss {maxmargin loss}
      - ELo and ELo+Distillation
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Chen, Ting, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. "Big self-supervised models are strong semi-supervised learners." arXiv preprint arXiv:2006.10029 (2020).
  - Empirical paper
  - unsupervised pretrain (task agnostic), semi-supervised learning (task dependent), fine-tuning
  - Impact of Big (wide and deep) network
    - More benefited from the unsupervised pretraining
    - Big Network: Good for learning all the Representations
      - Not necessarily require in one particular task - So we can leave the unnecessary information
    - More improve by distillation after fine tuneing
  - Wow: Maximum use of works: Big network to learn representations, and fine-tune then distill the task (supervised and task specific fine-tune) to a smaller network
  - Proposed Methods (3 steps) [this paper]
    - Unsupervised pretraining (SimCLR v2)
    - Supervised fine-tune using smaller labeled data
    - Distilling with unlabeled examples from big network for transferring task-specific knowledge into smaller network
  - This paper:
    - Investigate unsupervised pretraining and selfsupervised fine-tune
      - Finds: network size is important
      - Propose to use big unsupervised_pretraining-fine-tuning network to distill the task-specific knowledge to small network.
    - Figure 3 says all
  - Key contribution
    - Big networks works best (in unsupervised pretraining - very few data fine-tune) althought they may overfit!!!
    - Big model learns general representations, may not necessary when task-specific requirement
    - Importance of multilayer transform head (intuitive because of the transfer properties!! great: since we are going away from metric dimension projection, so more general features)
  - Section 2: Methods (details from loss to implementation)
  - Empirical study findings
    - Bigger model are label efficient
    - Bigger/Deeper Projection Heads Improve Representation Learning
    - Distillation using unlabeled data improves semi-supervised learning

1. Gordon, Daniel, Kiana Ehsani, Dieter Fox, and Ali Farhadi. "Watching the world go by: Representation learning from unlabeled videos." arXiv preprint arXiv:2003.07990 (2020).
  - Multi-frame multi-pairs positive negative (single imgae)- instance discrimination

1. Tao, Li, Xueting Wang, and Toshihiko Yamasaki. "Self-supervised video representation learning using inter-intra contrastive framework." In Proceedings of the 28th ACM International Conference on Multimedia, pp. 2193-2201. 2020.
  - Notion of intra-positive (augmentation, optical flow, frame differences, color of same frames)
  - Notion of intra-negative (frame shuffling, repeating)
  - Inter negative (irrelevant video clips - Different time or videos)
  - Two (Multi) Views (same time - positive keys) - and intra-negative
  - Figure 2 (memory bank approach)
  - Consider contrastive from both views.

1. Chen, Ting, and Lala Li. "Intriguing Properties of Contrastive Losses." arXiv preprint arXiv:2011.02803 (2020).
  - Generalize the CL loss to broader family of losses
    - weighted sum of alignment and distribution loss
      - Alignment: align under some transformation
      - Distribution: Match a prior distribution
  - Experiment with weights, temperature, and multihead projection
  - Study feature suppression!! (competing features)
    - Impacts of final loss!
    - Impacts of data augmentation
  - Suppression feature phenomena: Reduce unimportant features
  - Experiments
    - two ways to construct data
  - Expand beyond the uniform hyperspace prior
    - Can't rely on logSumExp setting in NT-Xent loss
    - Requires new optimization (Sliced Wasserstein distance loss)
      - Algorithm 1
    - Impacts of temperature and loss weights
  - Feature suppression
    - Target: Remove easy-to-learn but less transferable features for CL (e.g. Color distribution)
    - Experiments by creating the dataset
      - Digit on imagenet dataset
      - RandBit dataset
  -[github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Xiao, Tete, Xiaolong Wang, Alexei A. Efros, and Trevor Darrell. "What should not be contrastive in contrastive learning." arXiv preprint arXiv:2008.05659 (2020).
  - What if downstream tasks violates data augmentation (invariance) assumption!
    - Requires prior knowledge of the final tasks
  - This paper: Task-independent invariance
    - Requires separate embedding spaces!  (how much computation increases, redundant!)
      - Surely, multihead networks and shared backbones
      - new idea: Invariance to all but one augmentation !!
  - Pretext tasks: Tries to recover transformation between views
  - Contrastive learning: learn the invariant of the transformations
  - Is augmentation helpful: Not always!
    - rotation invariance removes the orientation senses! Why not keep both! disentanglement and the multitask!
  - This paper: Multi embedding space (transfer the shared backbones and task specific heads)
    - Each head is sensitive to all but one transformations
    - LooC: (Leave-one-out Contrastive Learning)
      - Multi augmentation contrastive learning
      - view generation and embedded space
      - Figure 2 (crack of jack)
    - Good setup to apply the ranking loss function
    - Careful with the notation (bad notation)
    -  instances and their augmentations!
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Tschannen, Michael, Josip Djolonga, Marvin Ritter, Aravindh Mahendran, Neil Houlsby, Sylvain Gelly, and Mario Lucic. "Self-supervised learning of video-induced visual invariances." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13806-13815. 2020.
  - Framework to learn image representations from non-curated videos in the wild by learning **frame-**, **shot-**, and **video-level invariances**
    -  benefit greatly from exploiting video-induced invariances,
  - Experiment with Youtube8M (3.7 M
  - Scale-able to image and video data. (however requires frame label encoding!! what about video AR networks)

1. Ma, Shuang, Zhaoyang Zeng, Daniel McDuff, and Yale Song. "Learning Audio-Visual Representations with Active Contrastive Coding." arXiv preprint arXiv:2009.09805 (2020).

1. Tao, Li, Xueting Wang, and Toshihiko Yamasaki. "Self-Supervised Video Representation Using Pretext-Contrastive Learning." arXiv preprint arXiv:2010.15464 (2020).
  - Tasks and Contrastive setup connection  (PCL)
  - hypersphere features spaces
  - Combine Pretext(some tasks, intra information)+Contrastive (similar/dissimilarity, inter-information) losses
  - Assumption: pretext and contrastive learning doing the same representation.
  - Loss function (Eq-6): Contrast (same body +head1)+pretext task (same body + head2)! - Joint optimization
  - Figure 1- tells the story
  - Three pretext tasks (3DRotNet, VCOP, VCP) - Experiment section
  - Both RGB and Frame difference
  - Downstrem tasks (UCF and HMDB51)
  - Some drawbacks at the discussion section.

1. Li, Junnan, Pan Zhou, Caiming Xiong, Richard Socher, and Steven CH Hoi. "Prototypical contrastive learning of unsupervised representations." arXiv preprint arXiv:2005.04966 (2020).
  - Addresses the issues of instance wise learning (?)
    - issue 1: semantic structure missing
  - claims to do two things
    - Learn low-level features for instance discrimination
    - encode semantic structure of the data
  - prototypes as latent variables to help find the maximum-likelihood estimation of the network parameters in an Expectation-Maximization framework.
    - E-step as finding the distribution of prototypes via clustering
    - M-step as optimizing the network via contrastive learning
  - Offers new loss function ProtoNCE (Generalized InfoNCE)
  - Show performance for the unsupervised representation learning benchmarking (?) and low-resolution transfer tasks
  - Prototype: a representative embedding for a group of semantically similar instances
    - prototype finding by standard clustering methods
  - Goal described in figure 1
  - EM problem?
    - goal is to find the parameters of a Deep Neural Network (DNN) that best describes the data distribution, by iteratively approximating and maximizing the log-likelihood function.
    - assumption that the data distribution around each prototype is isotropic Gaussian
  - Related works: MoCo
    - Deep unsupervised Clustering: not transferable?
  - Prototypical Contrastive Learning (PCL)
    - See the math notes from section 3
  - Figure 2 - Overview of methods
  -[github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers2.pdf)

1. Caron, Mathilde, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. "Unsupervised learning of visual features by contrasting cluster assignments." arXiv preprint arXiv:2006.09882 (2020).
  - SwAV (online algorithm) [swapped assignments between multiple vies of same image]
  - Contrastive learning, clustering
  - Predict cluster from different representation, memory efficiency!
  - 'code' consistency between image and its transformation {target}
    - similarity is formulated as a swapped prediction problem between positive pairs
    - no negative examples
    - the minibatch clustering methods implicitly prevent collapse of the representation space by encouraging samples in a batch to be distributed evenly to different clusters.
  - online code computation
  - Features and codes are learnt online
  - multi-crop: Smaller image with multiple views
  - validation: ImageNet linear evaluation protocol
  - Interested related work section
  - Key motivation: Contrastive instance learning
  - Partition constraint (batch wise normalization) to avoid trivial solution

1. Chen, Xinlei, and Kaiming He. "Exploring Simple Siamese Representation Learning." arXiv preprint arXiv:2011.10566 (2020).
  - Not using Negative samples, large batch or Momentum encoder!!
  - Care about Prevention of collapsing to constant (one way is contrastive learning, another way - *Clustering*, or *online clustering*, BYOL)
  - Concepts in figure 1 (SimSiam method)
  - Stop collapsing by introducing the stop gradient operation.
  - interesting section in 3
    - Loss SimCLR (but differently ) [eq 1 and eq 2]
    - Detailed Eq 3 and eq 4
    - Empirically shown to avoid the trivial solution

1. Caron, Mathilde, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. "Unsupervised learning of visual features by contrasting cluster assignments." Advances in Neural Information Processing Systems 33 (2020).

1. Morgado, Pedro, Nuno Vasconcelos, and Ishan Misra. "Audio-visual instance discrimination with cross-modal agreement." arXiv preprint arXiv:2004.12943 (2020).
  - learning audio and video representation [audio to video and video to audio!]
    - How they showed its better??
    - Exploit cross-modal agreement [what setup!] how it make sense!
  - consider in-sync audio video, proposed AVID (au-visu-instan-discrim)
  - Experiments with UCF-101, HMDB-51
  - Discussed limitation AVID & proposed improvement
    - Optimization method - [dimentional reduction by LeCunn and NCE paper]
      - Training procedure in section 3.2 AVID
    - Cross modal agreement!! it groups (how?) similar videos [both audio and visual]
  - Prior arts - binary task of audio-video alignment instance-based
    - This paper: matches in the representation embedding domain.
  - AVID calibrated by formulating CMA??
  - Figure 2: Variant of avids [summary of the papers]
    - This people are first to do it!!
    - Joint and Self AVID are bad in result! Cross AVID is the best for generalization in results!
  - CMA - Extension of the AVID [used for fine tuning]
    - section 4: Loss function extension with cross- AVID and why we need this?

1. Robinson, Joshua, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. "Contrastive Learning with Hard Negative Samples." arXiv preprint arXiv:2010.04592 (2020).
  - Sample good negative (difficult to distinguish) leads better represenation
    - challenges: No label! unsupervised method! Control the hardness!
    - enables learning with fewest instances and distance maximization.
  - Problem (1): what is true label? sol: Positive unlabeled learning !!!
  - Problem (2): Efficient sampling? sol:  efficient importance sampling!!! (consider lack of dissimilarity information)!!!
  - Section 3 most important!
  - section 4 interesting
  - [Notes](https://github.com/mxahan/PDFS_notes/blob/master/contrastive_sampling_debias_hardMining.pdf)

1. Chuang, Ching-Yao, Joshua Robinson, Yen-Chen Lin, Antonio Torralba, and Stefanie Jegelka. "Debiased contrastive learning." Advances in Neural Information Processing Systems 33 (2020).
  - Sample bias [negatives are actually positive! since randomly sampled]
  - need unbiased - improves vision, NLP and reinforcement tasks.
  - Related to Positive unlabeled learning
  - interesting results
  - [Notes](https://github.com/mxahan/PDFS_notes/blob/master/contrastive_sampling_debias_hardMining.pdf)

1. Zhao, Nanxuan, Zhirong Wu, Rynson WH Lau, and Stephen Lin. "What makes instance discrimination good for transfer learning?." arXiv preprint arXiv:2006.06606 (2020).

1. Metzger, Sean, Aravind Srinivas, Trevor Darrell, and Kurt Keutzer. "Evaluating Self-Supervised Pretraining Without Using Labels." arXiv preprint arXiv:2009.07724 (2020).

1. Bhardwaj, Sangnie, Ian Fischer, Johannes Ballé, and Troy Chinen. "An Unsupervised Information-Theoretic Perceptual Quality Metric." Advances in Neural Information Processing Systems 33 (2020).

1. Qian, Rui, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, and Yin Cui. "Spatiotemporal contrastive video representation learning." arXiv preprint arXiv:2008.03800 (2020).
  - unlabeled videos! video representation, pretext task! (CVRL)
  - simple idea: same video together and different videos differ in embedding space.  (TCN works on same video)
  - Video SimCLR?
  - Inter video clips (positive negatives)
  - method
    - Simclr setup for video (InfoNCE loss)
    - Video encoder (3D resnets)
    - Temporal augmentation for same video (good for them but not Ubiquitous)
    - Image based spatial augmentation (positive)
    - Different video (negative)
  - Downstream Tasks
    - Action Classification
    - Action Detection

1. Wang, Tongzhou, and Phillip Isola. "Understanding contrastive representation learning through alignment and uniformity on the hypersphere." In International Conference on Machine Learning, pp. 9929-9939. PMLR, 2020.
  - How to contraint on these and they perform better? weighted loss
    - Alignment (how close the positive features) [E<sub>pos</sub>[f(x)-f(y)]<sup>2</sup>]
    - Uniformly [take all spaces in the hyperplane] [little complex but tangible 4.1.2]
      - l_uniform loss definition [!!]
      - Interpretation of 4.2 see our future paper !!
  - cluster need to form spherical cap
  - Theoretical metric for above two constraints??
    - Congruous with CL
    - gaussing RBF kernel e^{[f(x) -f(y)]^2} helps on uniform distribution achieving.
  - Result figure-7 [interesting]
  - Alignment and uniform loss

1. Xiong, Yuwen, Mengye Ren, and Raquel Urtasun. "LoCo: Local contrastive representation learning." arXiv preprint arXiv:2008.01342 (2020).

1. Kalantidis, Yannis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, and Diane Larlus. "Hard negative mixing for contrastive learning." arXiv preprint arXiv:2010.01028 (2020).
  - TP (MoCHi): The effect of Hard negatives (how, definition?)
  - TP: feature level mixing for hard negatives (minimal computational overhead, momentum encoder) by synthesizing hard negatives (!!)
  - Related works: Mixup workshop

1. Grill, Jean-Bastien, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch et al. "Bootstrap your own latent: A new approach to self-supervised learning." arXiv preprint arXiv:2006.07733 (2020).
  - Unsupervised Representation learning in a discriminative method. (BOYL)
  - Alternative of contrastive learning methods (as CL depends on batch size, image augmentation method, memory bank, resilient). [No negative examples]
  - Online and Target network. [Augmented image output in online network should be close to main image in target network.] What about all zeros! (Empirically slow moving average helps to avoid that)
  - Motivation [section 3 method]
  - similarity constraint between positive keys are also enforced through a prediction problem from an online network to an offline momentum-updated network
  - BYOL tries to match the prediction from an online network to a randomly initialised offline
  network. This iterations lead to better representation than those of the random offline network.
  -By continually improving the offline network through the momentum update, the quality of the representation is bootstrapped from just the random initialised network
  - All about architecture! [encoder, projection, predictor and loss function]
  - Works only with batch normalization - else mode collapse
  - [More criticism](https://generallyintelligent.ai/understanding-self-supervised-contrastive-learning.html)

1. Qi, Di, Lin Su, Jia Song, Edward Cui, Taroon Bharti, and Arun Sacheti. "Imagebert: Cross-modal pre-training with large-scale weak-supervised image-text data." arXiv preprint arXiv:2001.07966 (2020).
  - vision pre-training /cross modal pretraining
  - New data collection (LAIT)
  - pretraining (see the loss functions)
    - Image/text from same context? (ITM)
    - Missing pixel detection?
    - Masked object Classification (MOC)
    - Masked region feature regression (MRFR)
    - Masked Language Models
  - Fine tune Tasks
    - Binary classification losses
    - Multi-class classification losses
    - Triplet loss
  - Multistage pretraining
  - Experimented with VQA and others. image language

1. Purushwalkam, Senthil, and Abhinav Gupta. "Demystifying contrastive self-supervised learning: Invariances, augmentations and dataset biases." arXiv preprint arXiv:2007.13916 (2020).
  - object detection and classification
  - quantitative experiment to Demystify CL gains! (reason behind success)
    - Observation1: MOCO and PIRL (occlusion invariant)
      - but Fails to capture viewpoint
    - gain from object-centric dataset - imagenet!
    - Propose methods to leverage learn from unstructured video (viewpoint invariant)
  - Utility of systems: How much invarinces the system encodes
  - most contrastive setup - occlusion invariant! what about viewpoint invariant?
  - Related works
    - Pretext tasks
    - Video SSL
    - Understanding SSRL
      - Mutual information
    - This work - Why CL is useful
      - study two aspects: (invariances encoding & role of dataset)
  - Demystifying Contrastive SSL
    - what is good Representation? Utilitarian analysis: how good the downstream task is?
      - What about the insights? and qualitative analysis?
    - Measuring Invariances
      - What invariance do we need? - invariant to all transformation!!
        - Viewpoint change, deformation, illumination, occlusion, category instance
      - Metrics: Firing representation, global firing rate, local firing rate, target conditioned invariance, representation invariant score.
      - Experimental dataset
        - occlusion (GOR-10K), viewpoint+instance invariance (Pascal3D+)
      - image and video careful augmentation
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers2.pdf)

1. Ermolov, Aleksandr, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. "Whitening for self-supervised representation learning." arXiv preprint arXiv:2007.06346 (2020).
  - New loss function (why? and where it works?)
    - Generalization of the BYOL approach?
    - No negative examples (the scatters are preserved)
  - Whitening operation (scattering effect)

1. Ebbers, Janek, Michael Kuhlmann, and Reinhold Haeb-Umbach. "Adversarial Contrastive Predictive Coding for Unsupervised Learning of Disentangled Representations." arXiv preprint arXiv:2005.12963 (2020).
  - video deep infomax: UCF101 dataset
  - Local and global features:
  - self note: go over this

1. Devon, R. "Representation Learning with Video Deep InfoMax." arXiv preprint arXiv:2007.13278 (2020).
  - DIM: prediction tasks between local and global features.
    - For video (playing with sampling rate of the views)

1. Liang, Weixin, James Zou, and Zhou Yu. "Alice: Active learning with contrastive natural language explanations." arXiv preprint arXiv:2009.10259 (2020).
  - Contrastive natural language!!
  - Experiments -  (bird classification and Social relationship classifier!!)
  - key steps
    - run basic Classifier
    - fit multivariate gaussian for all class (embedding!!), and find b pair of classes with lowest JS divergence.
    - contrastive query to machine understandable form (important and critical part!!). [crop the most informative parts and retrain.]
    - neural arch. morphing!! (heuristic and interesting parts) [local, super classifier and attention mechanism!]

1. Ma, Shuang, Zhaoyang Zeng, Daniel McDuff, and Yale Song. "Learning Audio-Visual Representations with Active Contrastive Coding." arXiv preprint arXiv:2009.09805 (2020).

1. Park, Taesung, Alexei A. Efros, Richard Zhang, and Jun-Yan Zhu. "Contrastive Learning for Unpaired Image-to-Image Translation." arXiv preprint arXiv:2007.15651 (2020).
  - Contrastive loss (Same patch of input - output are +ve and rest of the patches are -ve example) [algorithmic]
  - Trains the encoder parts more! (Fig 1, 2) ; Decoders train only on adversarial losses.
  - Contribution in loss (SimCLR) kinda motivation

1. Guo, Daniel, Bernardo Avila Pires, Bilal Piot, Jean-bastien Grill, Florent Altché, Rémi Munos, and Mohammad Gheshlaghi Azar. "Bootstrap Latent-Predictive Representations for Multitask Reinforcement Learning." arXiv preprint arXiv:2004.14646 (2020).
  - Notation Caution. Representation learning [latent space for observe and history]
  - States to future latent observation to future state.
  - Latent embedding of history.
  - Alternative for Deep RL
  - Experiments
    - DMLab-30
    - Compared for PopArt-IMPALA (RNN) with DRAW, Pixel-control, Contrastive predictive control.
  - Partially observable environments and Predictive representation.
  - Learn agent state by predictive representation.
  - RNN compresses history from the observations and actions; History as input for new decision making
  - Interesting section 3!

1. Tian, Yonglong, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. "What makes for good views for contrastive learning." arXiv preprint arXiv:2005.10243 (2020).
  - Multi-view in-variance
  - What is invariant?? (shared information between views)
  - balance to share the information we need in view!!
  - Questions
    - Knowing task what will be the view??!
    - generate views to control the MI
  - Maximize task related shared information, minimize nuisance variables. (InfoMin principle)
  - Contributions (4 - method, representation and task-dependencies, ImageNet experimentation)
  - Figure 1: summary.
  - Optimal view encoder.
    - Sufficient (careful notation overloaded! all info there), minimal sufficient (someinfo dropped), optimal representation (4.3)- only task specific information retrieved
  - InfoMin Principle: views should have different background noise else min encoder reduces the nuisance variable info. (proposition 4.1 with constraints.)
  - suggestion: Make contrastive learning hard
  - Figure 2: interesting. [experiment - 4.2]
  - Figure 3: U-shape MI curve.
  - section 6: different views and info sharing.

1. Lu, Jiasen, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. "12-in-1: Multi-task vision and language representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10437-10446. 2020.
  - MTL + Dynamic "stop and go" schedule. [multi-modal representation learning]
  - ViLBERT base architecture.

1. Misra, Ishan, and Laurens van der Maaten. "Self-supervised learning of pretext-invariant representations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6707-6717. 2020.
  - Pre-training method [algorithmic]
  - Pretext learning with transformation invariant + data augmentation invariant
  - See the loss functions
    - Tries to retain small amount of the transformation properties too !!
  - Use contrastive learning (See NCE)
      - Maximize MI
      - Utilizes extra head on the features.
  - Motivation from predicting video frames
  - Experiment of jigsaw pretext learning
  - Hypothesis: Representation of image and its transformation should be same
  - Use different head for image and jigsaw counterpart of that particular image.
      - Motivation for learning some extra things by different head network
  - Noise Contrastive learning (contrast with other images)
  - As two head so two component of contrastive loss. (One component to dampen memory update.)
  - Implemented on ResNet
  - PIRL

1. Srinivas, Aravind, Michael Laskin, and Pieter Abbeel. "Curl: Contrastive unsupervised representations for reinforcement learning." arXiv preprint arXiv:2004.04136 (2020).

1. Chen, Ting, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. "A simple framework for contrastive learning of visual representations." arXiv preprint arXiv:2002.05709 (2020).
  - Truely simple! (SimCLR) [algorithmic]
  - Two transfers for each image and representation
  - Same origin image should be more similar than the others.
  - Contrastive (negative) examples are from image other than that.
  - A nonlinear projection head followed by the representation helps.

1. Asano, Yuki M., Mandela Patrick, Christian Rupprecht, and Andrea Vedaldi. "Labelling unlabelled videos from scratch with multi-modal self-supervision." arXiv preprint arXiv:2006.13662 (2020).
  - clustering method that allows pseudo-labelling of a video dataset without any human annotations, by leveraging the natural correspondence between the audio and visual modalities
  - [Multi-Modal representation learning]

1. Patrick, Mandela, Yuki M. Asano, Ruth Fong, João F. Henriques, Geoffrey Zweig, and Andrea Vedaldi. "Multi-modal self-supervision from generalized data transformations." arXiv preprint arXiv:2003.04298 (2020).
  - [Multi-Modal representation learning]

1. Khosla, Prannay, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. "Supervised contrastive learning." arXiv preprint arXiv:2004.11362 (2020).
  - [Algorithmic]

1. He, Kaiming, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. "Momentum contrast for unsupervised visual representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729-9738. 2020.
  - Dynamic dictionary with MA encoder [Algorithmic] Contribution
  - (query) encoder and (key) momentum encoder.
  - The update of key encoder in a momentum fashion
      - Query updated by back propagation
  - Algorithm [1] is the Core: A combination of **end-end & the Memory bank**
  - key and query match but the queue would not match!
  - Momentum parametric dependencies
  - Start with the key and query encoder as the Same
    - key updates slowly, query updates with SGD.

1. Huynh, Tri, Simon Kornblith, Matthew R. Walter, Michael Maire, and Maryam Khademi. "Boosting Contrastive Self-Supervised Learning with False Negative Cancellation." arXiv preprint arXiv:2011.11765 (2020).
  - False negative Problem!! (detail analysis) [sampling method for CL]
    - Aim: Boosting results
  - Methods to Mitigate false negative impacts (how? what? how much impact! significant means?? what are other methods?)
  - Hypothesis: Randomly taken negative samples (leaked negative)
  - Overview
    - identify false negative (how?): Finding potential False negative sample [3.2.3]
    - Then false **negative elimination** and **false negative attraction**
  - Contributions
    - applicable on top of existing cont. learning

1. Lee, Jason D., Qi Lei, Nikunj Saunshi, and Jiacheng Zhuo. "Predicting what you already know helps: Provable self-supervised learning." arXiv preprint arXiv:2008.01064 (2020).
  - Highly theoretical paper.
  - TP:  is to investigate the statistical connections between the random variables of input features and downstream labels
    - Two important notion for the tasks: i) Expressivity (does the ssl good enough) ii) Sample complexity (reduce the complexity of sampling)
  - TP: analysis on the reconstruction based SSL
  - section 3 describes the paper summary. (connected to simsiam: Section 6
  - Discusses about conditional independence ([CI](!https://www.probabilitycourse.com/chapter1/1_4_4_conditional_independence.php)) condition of the samples w.r.t the labels

1. Dwibedi, Debidatta, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. "Counting out time: Class agnostic video repetition counting in the wild." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10387-10396. 2020.
  - Countix Dataset for video repeatation count. (part of Kinetics dataset)
    - annotated with segments of repeated actions and corresponding counts.
  - Per-frame embedding and similarity!! (RepNet)
  - Compared with benchmark:  PERTUBE and QUVA
  - Not really a self-supervised set up, rather a multitask setup.
  - Propose to create synthetic dataset by frame repeatation and reversal! Camera motion augmentation (Augmentation)

1. Appalaraju, Srikar, Yi Zhu, Yusheng Xie, and István Fehérvári. "Towards Good Practices in Self-supervised Representation Learning." arXiv preprint arXiv:2012.00868 (2020).
  - Tries to unravel the mysteries behind CL (!!)
  - Empirical analysis: provide practice/insight tips
  - Why data augmentation and adding projection head works in CL??
    - not true in the supervised setting!!
  - Design choice and good practice boost CL representation!
  - This paper: Focuses (empirical analysis) on three of the key points
  - (i) Importance of MLP heads (key factor)
    - Requires non-linear projection head on top (fc and ReLU)
  - (ii) semantic label shift problems by data augmentation:  breakdown of class boundaries in strong augmentations
  - (iii) Investigate on Negative Samples: Quality and Quantity


1. Han, Tengda, Weidi Xie, and Andrew Zisserman. "Self-supervised co-training for video representation learning." arXiv preprint arXiv:2010.09709 (2020).
  - visual only selfsupervised Representation learning!! [sampling method for CL]
  - new form of supervised CL (adding semantic positive to instance based Infor NCE loss)!! HOW? {explanation in figure 1}
  - co-training scheme with infoNCE loss!! (Explore complementary information from different views
  - evaluating the quality of learned representations (two downstream tasks: action recognition and video retrieval)
  - Question of "is instance discrimination is best??" - NO
    - hard positves! oracle experiment (UberNCE!)
    - CoCLR!! mining positive samples of data. (TCN extension by adding another modality, different network for each modality! how it is trained? Dynamic objective!!)
  - Focused more on sampling procedure
  - Experimented with RGB and FLOW network (aim to improve their representation)
  - Related works: Visual-only supervised learning, Multi-modal self-supervised learning, Co-training Paired networks and Video action recognition
  - InfoNCE and UberNCE *differs in sampling positives*
  - CoCLR algorithm (Initialization: flow and RGB net trained individually, Alternation: mine hard positive based on others Eq 3,4)
  - Dataset: UCF101, Kinetics-400, HMDB51

## 2021

1. Zimmermann, Roland S., Yash Sharma, Steffen Schneider, Matthias Bethge, and Wieland Brendel. "Contrastive learning inverts the data generating process." In International Conference on Machine Learning, pp. 12979-12990. PMLR, 2021.
  -  prove that *feedforward models* trained with objectives belonging to the commonly used *InfoNCE family* learn to implicitly *invert the underlying generative model* of the observed data
    - proofs make certain *statistical assumptions* about the generative model, however, hold empirically even if these assumptions are severely violated
  - highlights a fundamental connection between CL, generative modeling, and nonlinear ICA

1. Cui, Jiequan, Zhisheng Zhong, Shu Liu, Bei Yu, and Jiaya Jia. "Parametric contrastive learning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 715-724. 2021.
  - Application scenario: Unbalanced classes
  - tackle long-tailed recognition
    - Too much **complex paper** for then.


1. Kuang, Haofei, Yi Zhu, Zhi Zhang, Xinyu Li, Joseph Tighe, Sören Schwertfeger, Cyrill Stachniss, and Mu Li. "Video Contrastive Learning with Global Context." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3195-3204. 2021.
  - Why we require the global context though?? what is even global context??
  - TP:  video-level CL method based on *segments* to formulate positive pairs
  - Key contribution: uniformly divide the video into several segments, and randomly pick a clip from each segment (anchor) and randomly pick a clip from each segment again to form the positive tuple
    - a temporal order regularization term (enforce the inherent video sequential structure)
  - Video-level contrastive learning (VCLR)
  - Well! they contrast within contrast!! (although different head for losses)
  - applied to dataset with notion of global and local tasks.
  - Segment losses: Frame shuffling reformulated as classification problem

1. Zhong, Huasong, Jianlong Wu, Chong Chen, Jianqiang Huang, Minghua Deng, Liqiang Nie, Zhouchen Lin, and Xian-Sheng Hua. "Graph contrastive clustering." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9224-9233. 2021.
  - Well they use some sort of supervised information about latent classes.
    - Not directly the instance contrastive approach.
  - *representation:* a graph Laplacian based contrastive loss is proposed (discriminative and clustering-friendly features).
  - *assignment:* a novel graph-based contrastive learning strategy is proposed (compact clustering assignments.)
  - Overveiw in Figure 2
  - However Heavy computations
  - Math on note

1. Wang, Jingyu, Zhenyu Ma, Feiping Nie, and Xuelong Li. "Progressive self-supervised clustering with novel category discovery." IEEE Transactions on Cybernetics (2021).
  - Parameter-insensitive anchor-based graph obtained from balanced K-means and hierarchical K-means
  - a novel representative point selected strategy based on a semisupervised framework
  - Something to do with the laplacian!! and its decomposition.

1. Li, Yunfan, Peng Hu, Zitao Liu, Dezhong Peng, Joey Tianyi Zhou, and Xi Peng. "Contrastive clustering." In 2021 AAAI Conference on Artificial Intelligence (AAAI). 2021.
  - One stage online clustering method (CC) [instance and cluster label contrast]
    - TP: reveal that the row and column of the feature matrix intrinsically correspond to the instance and cluster representation
    - TP: Similar to **Multitask network** setup
      - Figure 1: Explains the idea
    - PQ: why don't they use the entropy loss!! (making single node go up)
  - Two different projection head (instance [row] and cluster label [column])
    - Figure 2: Methods overview.
  - NMI metric for clustering

1. HaoChen, Jeff Z., Colin Wei, Adrien Gaidon, and Tengyu Ma. "Provable guarantees for self-supervised deep learning with spectral contrastive loss." Advances in Neural Information Processing Systems 34 (2021).
  - Proposes graph theoretic based spectral contrastive losses
  - [background for graph clustering](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf)

1. Jahanian, Ali, Xavier Puig, Yonglong Tian, and Phillip Isola. "Generative models as a data source for multiview representation learning." arXiv preprint arXiv:2106.05258 (2021).
  - RQ: why bother using dataset when you have generator!! (used no real-data)
    - off-the-shelf image generator to get multiview data
    - Requires careful sampling and training method (!!!)
      - *Hypothesis*: Generator (a organized copy of compressed dataset)
  - We provide an exploratory study of representation learning in the setting of synthetic data sampled from pre-trained generative models:
  - **Assumption** Generator is able to get a good multiview data (sufficient capable generator)
  - **Findings**: i) CL can be naturally extended to learning from generative samples (different “views” of the data are created via transformations in the model’s latent space) ii) further can be combined with data augmentation iii) sub-logarithmic performance improvement with generator  
  - Interesting related work section
  - Figure 1: summary
  - Key analysis on how to make latent tx to get multiview data. i) gaussian approaches, ii) steered latent views

1. Islam, Ashraful, Chun-Fu Richard Chen, Rameswar Panda, Leonid Karlinsky, Richard Radke, and Rogerio Feris. "A broad study on the transferability of visual representations with contrastive learning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8845-8855. 2021.
  - TP: *comprehensive study* on the transferability of learned representations of different contrastive approaches for **linear evaluation, full-network transfer, and few-shot recognition**
  - Experiment: 12 downstream datasets from different domains, and object detection tasks on MSCOCO and VOC0712
  - Results shows the great transferability of the SSL (expected)
  - Good terminology: Self-SupCon (augmented version) and SupCon (different instances)

1. Graf, Florian, Christoph Hofer, Marc Niethammer, and Roland Kwitt. "Dissecting supervised constrastive learning." In International Conference on Machine Learning, pp. 3821-3830. PMLR, 2021.
  - Discusses the problem of class collapse when minimal loss is attained
  - TP address the question whether there are fundamental differences (between softmax and supcon loss) in the sought-for representation geometry in the output space of the encoder at minimal loss.
    - *Insight-1*. both losses attain their minimum once the representations of each class collapse to the vertices of a regular simplex, inscribed in a hypersphere. (theoretical)
    - reaching a close-to-optimal state typically indicates good generalization performance.
    - *Insight-2*. Supcon works **superlinearly** and Softmax works linearly. (empirical)

1. Fu, Daniel Yang, Mayee F. Chen, Michael Zhang, Kayvon Fatahalian, and Christopher Ré. "The Details Matter: Preventing Class Collapse in Supervised Contrastive Learning." (2021).
  - modification to supervised contrastive (SupCon) loss that prevents class collapse (keeps strata) by uniformly pulling apart individual points from the same class.
    - SupCon losses information. (collapse the strata information), not good for the downstream tasks.
      - Enforces one embedding per class: a regular simplex inscribed in hypersphere.
    - Proposes L_{spread} loss [a slight modification of L_{sc}] to preserve the strata in embedding space.
  - Hypothesis: Rarer and distinct strata are further away from common strata. (nice idea, in a unsupervised setup what is even important??) : has entropy flavour.

1. Shah, Anshul, Suvrit Sra, Rama Chellappa, and Anoop Cherian. "Max-Margin Contrastive Learning." arXiv preprint arXiv:2112.11450 (2021).
  - Addresses the slow convergence of contrastive methods (uses SVM objective)
    - by selecting negative examples using SVM methods (maximizes the boundary)
  - TP: simplification of SVM for alleviating computations and maximizing boundaries for hard negatives (MMCL)
    - Essentially a hard-negative mining problem!! (quality over quantity)
  - TP: Propose to separate the embedding using powerful SVM classifier.
    - One vs all fashion detection!!!
  - Experiment with vision, video, S3D network, tanh kernel.

1. Bardes, Adrien, Jean Ponce, and Yann LeCun. "Vicreg: Variance-invariance-covariance regularization for self-supervised learning." arXiv preprint arXiv:2105.04906 (2021).
  - TP: Variance-Invariance-Covariance Regularization (how to avoid collapse)
    - Applies two regularization term separately with the embeddings : term (1) maintains the variance of each embedding dimension above a threshold, term (2) decorrelates each pair of variables.
    - Key contribution: Loss function (triple objective)
  - Related works: prevent collapse by i) Contrastive methods / vector quantization (Simclr, MoCo, memory bank, etc) , ii) Information maximization (prevents information collapse).
  - Great intuition however, requires good sampling. [invariant mean between embeddings, variance of embeddings over a batch > th, covariance between a pair in batches &#8594; 0]
  - Requires asymmetric stop gradient (no weight sharing between two branches: allow mutlimodal)
  - As always interesting related work section. [related to decorrelation of barlow twin]
  - Network setup: Encoder and Expander [(1) eliminate the information by which the two representations differ, (2) expand the dimension in a non-linear fashion so that decorrelating the embedding variables will reduce the dependencies (not just the correlations) between the variables of the representation vector]

1. Ayush, Kumar, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell, and Stefano Ermon. "Geography-aware self-supervised learning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10181-10190. 2021.
  - training methods that exploit the spatio-temporal structure (!) of remote sensing data (!!). [application to satellite dataset]
    - [requires specialized dataset]: spatially aligned images over time to construct temporal positive pairs (temporal positive pairs) in i) contrastive learning and ii) geo-location to design pre-text tasks (predicting image location, utilizing geo-location data's metadata info)
      - MT: Can we also contrast based on that (disentangled)
    - Experiments: Functional Mop of the world (*fMoW*)benchmark, **Geo-tagged imagenet** dataset. Arch: ResNet
    - *Hypothesis*: Existance of remote sensing data's geo-located and multiple images of the same location over time.
    - representations to be invariant to subtle variations over time (object detection or semantic segmentation) [task 1: contrastive]
    - representations that reflect geographical information (useful in remote sensing tasks) [task 2: meta-data prediction]
  - TP: Combine two loss function specialized in remote sensing image dataset.

1. Das, Srijan, and Michael S. Ryoo. "ViewCLR: Learning Self-supervised Video Representation for Unseen Viewpoints." arXiv preprint arXiv:2112.03905 (2021).
  - View generator (3d geometric transformations): Learnable augmentation for pretext task (maximizing viewpoint similarities).
  - **Aim: Generalize over unseen camera viewpoints**. Camera invariant features
    - learnable augmentation to induce viewpoint (by VG) changes while for self-supervised representation.
  - Dataset: NTU RGB+D and NUCLA, MOCO for instance discrimination. (New PRETraining!!), arch: S3D, Evaluation: Cross subject and cross setting protocol.

1. Hua, Tianyu, Wenxiao Wang, Zihui Xue, Sucheng Ren, Yue Wang, and Hang Zhao. "On feature decorrelation in self-supervised learning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9598-9608. 2021.
  - analysis the collapse issues (looks detail of complete and dimensional collapse: figure 1)
  - Verification of collapse (!!) by standardizing variance.
  - Reveal connection between dimensional collapse and strong correlation. (where??) [dimension collapse is indicated by strong correlation among the features]
  - Performance gain by feature decorrelation
  - TP: Proposes decorrelated Batch normalization layer
    - Earlier findings: BN in projection layer avoids vanishing variances (complete collapse).
  - Main read: Section 3 is interesting to read (some key findings)- may be transcribed to work.

1. Desai, Karan, and Justin Johnson. "Virtex: Learning visual representations from textual annotations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11162-11173. 2021.

1. Chen, Kai, Lanqing Hong, Hang Xu, Zhenguo Li, and Dit-Yan Yeung. "MultiSiam: Self-supervised Multi-instance Siamese Representation Learning for Autonomous Driving." arXiv preprint arXiv:2108.12178 (2021).
  - TP: Two issues: (1) define positive samples for cross-view consistency ? (2) measure similarity in multi-instance circumstances ?
  - Experiments data: WayMo, SODA10M
  - Global consistency to local consistency?
  - basic assumption of instance discrimination: different views of the same image should be consistent in the feature space
    - what about multi-instance in a single image (realistic case)!!!
    - definition of positive samples is definitely needed to extend cross-view consistency framework to multi-instance circumstances. (multiple things in single image)
  - Methods: Uses IoU as proxy for data and Noise. (les IoU - More Noise)
    - Remove global pooling (counter the multiple instance collapse)
    - Well, very complex loss function!
  - Related works: VirTex, ConVIRT, ICMLM (proof of concept)

1. Yang, Ceyuan, Zhirong Wu, Bolei Zhou, and Stephen Lin. "Instance localization for self-supervised detection pretraining." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3987-3996. 2021.
  - Propose a new self-supervised tasks : Instance Localization. [self-supervised task design]
  - Put image crop into another and try to predict using RPN!!

1. Wang, Feng, and Huaping Liu. "Understanding the behaviour of contrastive loss." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2495-2504. 2021.
  - Studies the impact of the temperature in loss function (uniformity and temperature).
    - How to ensure tolerant for semantic similar examples [uniformity tolerance dilemma]
  - This paper: studies hardness aware properties (parameter in loss function).

1. Pan, Tian, Yibing Song, Tianyu Yang, Wenhao Jiang, and Wei Liu. "Videomoco: Contrastive video representation learning with temporally adversarial examples." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11205-11214. 2021.

1. Xu, Jiarui, and Xiaolong Wang. "Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective." arXiv preprint arXiv:2103.17263 (2021).

1. Liu, Xiao, Fanjin Zhang, Zhenyu Hou, Li Mian, Zhaoyu Wang, Jing Zhang, and Jie Tang. "Self-supervised learning: Generative or contrastive." IEEE Transactions on Knowledge and Data Engineering (2021).
  - Another Survey Paper

1. Akbari, Hassan, Linagzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. "Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text." arXiv preprint arXiv:2104.11178 (2021).

1. Sordoni, Alessandro, Nouha Dziri, Hannes Schulz, Geoff Gordon, Philip Bachman, and Remi Tachet Des Combes. "Decomposed Mutual Information Estimation for Contrastive Representation Learning." In International Conference on Machine Learning, pp. 9859-9869. PMLR, 2021.

1. Yang, Mouxing, Yunfan Li, Zhenyu Huang, Zitao Liu, Peng Hu, and Xi Peng. "Partially view-aligned representation learning with noise-robust contrastive loss." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1134-1143. 2021.
  - Partially view alignment Problem (PVP)???
  - Objective: aligning data and learning representation (why it is important??)
  - TP: Propose noise robust Contrastive loss to eliminate False negative {related to learning with noisy labels, any evidence on this?}
  - New definition of noisy labels (why, is it valid?) [false negative pair as noisy label! ]
  - Experiment: 10 SOTA classification and clustering tasks!
  - Two assumption on Data: Completeness (contains all views [partially data-missing problem]) and Consistency (no false negative/positive [partial view alignment problem]) of views.
  - Proposed modified distance loss for the negative pairs.

1. Dave, Ishan, Rohit Gupta, Mamshad Nayeem Rizve, and Mubarak Shah. "TCLR: Temporal Contrastive Learning for Video Representation." arXiv preprint arXiv:2101.07974 (2021).
  - temporal CL framework!!- (why?) - two novel loss functions
    - (local-local): non-overlapping of same videos!!
    - (global-local): increase temporal diversity!!
    - NCE based loss function formulation.
  - Interesting way to sample local and global (why is it necessary???) [figure 2, 3, and 4]
  - Architecture: 3D ResNet-18

1. Tian, Yuandong, Xinlei Chen, and Surya Ganguli. "Understanding self-supervised learning dynamics without contrastive pairs." arXiv preprint arXiv:2102.06810 (2021).
  - Theoretical: Why non-contrastive (without negative pairs) methods do not collapse (BYOL, SimSiam by using extra predictors/stop-gradient)
    - TP: DirectPred (Directly sets the linear predictor based on the statistics of its inputs, without gradient training)
    - motivated by theoretical study of the nonlinear learning dynamics of non-contrastive SSL in simple linear networks
      - yields conceptual insights into how non-contrastive SSL methods learn, how they avoid representational collapse, and impact of multiple factors, like *predictor networks, stop-gradients, exponential moving averages, and weight decay*
  - Empirical impacts of multiple hyperparams: i) EMA/momentum encoder ii) predictor optimality and LR iii) Weight decay (good ablation studies) [**section 3.2**]
  - [**Essential part of non-contrastive SSL: existance of the predictor and the stop-gradient**]
  - TP directPred: thereby avoiding complicated predictor dynamics and initialization issues by using PCA and setting predictor weight [*section 4*].
  - Th 1: ((Weight decay promotes balancing of the predictor and online networks), Th 2: (The stop-gradient signal is essential for success.)

1. Caron, Mathilde, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. "Emerging properties in self-supervised vision transformers." arXiv preprint arXiv:2104.14294 (2021).
  - DINO: knowledge Distillation with NO labels (Figure 2, Algorithm 1)
  - Self-supervised learning with vision transformer
  - Observations: (i) T features contain explicit information (!!!) about the semantic segmentation of an image
    - explicit information: scene layout, object boundaries (directly accessible)
    - Performs basic k-NN without any supervision.
  - Cross entropy loss (sharpening and centering requires to avoid collapse)
  - Experiments: Architecture (i) ViT and (ii) ResNet

1. Tan, Hao, Jie Lei, Thomas Wolf, and Mohit Bansal. "VIMPAC: Video Pre-Training via Masked Token Prediction and Contrastive Learning." arXiv preprint arXiv:2106.11250 (2021).

1. Tian, Yonglong, Olivier J. Henaff, and Aaron van den Oord. "Divide and Contrast: Self-supervised Learning from Uncurated Data." arXiv preprint arXiv:2105.08054 (2021).
  - Effects of contrastive learning from larger, less-curated image datasets such as YFCC
    - Finds a large difference in the resulting representation quality
    - because (hypothesis) distribution shift in image class (less relevant negative to learn)
  - TP: new approach DnC (divide and contrast) - alternate between CL and cluster based hard negative mining
    - Methods: Train individual models on subset and distill them into single model
    - Application Scope: less curated data to train! Aim: attempts to recover local consistency.
    - The distillation parts requires k+1 networks!
  - what if: the networks reaches different embedding in each running (from scratch!)
    - How come predicting both simultaneously makes sense?
  - Experiment dataset: JFT-300, YFCC100M, 95M Flickr

1. Zheltonozhskii, Evgenii, Chaim Baskin, Avi Mendelson, Alex M. Bronstein, and Or Litany. "Contrast to Divide: Self-Supervised Pre-Training for Learning with Noisy Labels." arXiv preprint arXiv:2103.13646 (2021).
  - **warm-up obstacle**: the inability of standard warm-up stages to train high quality feature extractors and avert memorization of noisy labels.!!
    - SoTA depends on warm-up stage where standard supervised training is performed using the full (noisy) training set
  - TP: contrast to divide (C2D)
    - benefit: drastically reducing the warm-up stage’s susceptibility to noise level, shortening its duration, and increasing extracted feature quality
  - warp-up stages!! Current works focus on warm up length only! requires optimal warm-up length! or relying on external dataset! TP: Self-supervised pretraining!
  - TP: firstly perform simclr, then proceed with standard LNL algorithm (: ELR+ and DivideMix!!)

1. Huang, Lianghua, Yu Liu, Bin Wang, Pan Pan, Yinghui Xu, and Rong Jin. "Self-supervised Video Representation Learning by Context and Motion Decoupling." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13886-13895. 2021.
  - a method that explicitly decouples motion supervision from context bias through a *carefully designed* pretext task
    - (i) Context matching (CL between key frame (how to find it?? - Random frame selection) and video clips) & Motion Prediction (estimate motion features in the future & also a regularizer)
    - Architecture: Shared backbone and separate head for the tasks
  - Figure 2: says all: Two target for the V-network (context {top, extracted from image} & motion {bottom, derived using the motion vectors?})
  - Related works: representation learning, AR in compressed videos, Motion prediction
  - Experiments:
    - Networks: Video backbone: C3D, R(2+1)D-26 and R3D-26 (V, video), shallow R2D-10 (I, context), and R3D-10 (video).
    - Data: UCF, Kinetics, HMDB51, with augmentation (same for one example), hard negatives,

1. Feichtenhofer, Christoph, Haoqi Fan, Bo Xiong, Ross Girshick, and Kaiming He. "A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3299-3309. 2021.
  - A large-scale study on unsupervised spatiotemporal representation learning from videos
    - Generalize image based method into space-time. (e.g. crop [image] to clip [video])
    - *Provides many empirical results*

1. Hénaff, Olivier J., Skanda Koppula, Jean-Baptiste Alayrac, Aaron van den Oord, Oriol Vinyals, and João Carreira. "Efficient Visual Pretraining with Contrastive Detection." arXiv preprint arXiv:2103.10957 (2021).
  - Tackles the computational complexity of the self-supervised learning
    - By providing new objective (extract rich information from each image!! ) named *Contrastive Detection* (Figure 2)
      - Two variants: SimCLR and the BYOL
      - Knowledge tx across the dataset
      - Heuristic mask on image and train!
        - Of the shelf unsupervised/human annotators [external methods]
      - Pull the features spaces close!!
    -  maximizes the similarity of object-level features across augmentations.
    - Result 5x less pretraining
    - Compared with SEER!!
  - Experiment with imagenet to COCO dataset
  - This paper: Special Data augmentation scheme

1. Goyal, Priya, Mathilde Caron, Benjamin Lefaudeux, Min Xu, Pengchao Wang, Vivek Pai, Mannat Singh et al. "Self-supervised pretraining of visual features in the wild." arXiv preprint arXiv:2103.01988 (2021).
  - SEER and connection to few shot learning
  - RQ: pretraining extremely large collection of uncurated, unlabeled images for good achievement?
    - Solution: Continuous learning in a self-supervised manner! (online fashion training)
  - TP: pretrain high capacity model (RegNet Architecture!! 700M params) on billions images! using SwAV approaches with large BS.
  - Results: One of the best model!! both on curated and uncarated data.
  - Related works: Scale on large uncurated images. (also scaling the network - Good Reads)
  - Results on finetuning large models, low shot learning and transfer learning.
  - Ablation studies: Model architecture, scaling the training data, scaling the self-supervised model head

1. Zbontar, Jure, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." arXiv preprint arXiv:2103.03230 (2021).
  - Tries to avoid trivial solution (by new objective function, cross-correlation matrix)!!
    - A new loss function
  - Same images ~ augmentation representations are distorted version of each other.
  - Motivation: Barlows Redundancy-reduction principle
  - Pair of identical neurons
  - Just WOW: The hell the idea is!!
    - Intriguingly avoids trivial solutions
    - Should require large batch size
  - Figure 1 & algorithm 1: Crack of all jack
  - BOYL follow up works
  - Contrastive learning
    - Either Negative examples
    - Or architectural constraint/ Asymmetric update
  - Motivated from Barlow-twin (redundancy-reduction principle) 1961!
    - H. Barlow hypothesized that the goal of sensory processing is to recode highly redundant sensory inputs into a factorial code (a code with statistically independent components).
    - propose new loss function:  tries to make the cross-correlation matrix computed from twin representations as close to the identity matrix as possible
      - Advantages: Not required Asymmetric update or large batch size
  - Methods description
    - What is Barlow twins
      - connection to Information bottleneck (IB)
    - Implementation details
  - Result
    - Linear and Semi-supervised Evalution of imagenet
    - Transfer learning
    - Object detection and segmentation
  - Ablation study
    - Variation of Loss function
    - Impacts of Batch Size
      - Outperformed by Large BS with BYOL and SimCLR
    - network selection impacts
      - projection head importance
    - Importance of the data augmentation
  - Discussion (interesting)
    - Comparison with Prior Art
      - InfoNCE
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers2.pdf)

1. Tsai, Yao-Hung Hubert, Martin Q. Ma, Muqiao Yang, Han Zhao, Louis-Philippe Morency, and Ruslan Salakhutdinov. "Self-supervised representation learning with relative predictive coding." arXiv preprint arXiv:2103.11275 (2021).
  -

1. Liu, Yang, Keze Wang, Haoyuan Lan, and Liang Lin. "Temporal Contrastive Graph for Self-supervised Video Representation Learning." arXiv preprint arXiv:2101.00820 (2021).
  - Graph Neural Network And Contrastive Learning
  - Video frame shuffling

1. Bulat, Adrian, Enrique Sánchez-Lozano, and Georgios Tzimiropoulos. "Improving memory banks for unsupervised learning with large mini-batch, consistency and hard negative mining." arXiv preprint arXiv:2102.04442 (2021).
  - Improvement for the memory bank based formulation (whats the problem??)
    - TP: (I) Large mini-batch: Multiple augmentation! (II) Consistency: Not negative enforce! The heck? how to prevent collapse? (III) Hard Negative Mining
    - Results: Improve the vanilla memory bank! Evidence!! Dataset experimentation!
  - Exploration:  With Batch Size and visually similar instances (is the argument 2 is valid?)
  - Contribution 2 seems important!
  - Each image is augmented k times: More data augmentation!
  - Interesting way to put the negative contrastive parts to avoid collapse (eq 3)
  - Experiments: Seen testing categories (CIFAR, STL), & unseen testing categories (Stanford Online Product). ResNet-18 as baseline model

1. Dwibedi, Debidatta, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. "With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations." arXiv preprint arXiv:2104.14548 (2021).
  - Positive from other instance (instead of augmented version of same image!)
  - positive sampling: Nearest neighbor in the latent space (NNCLR) [this covers both same samples and Nearest neighbor in the latent spaces]
  - Benefit: Less reliant on complex data augmentation (empirical results)
  - Experiments: semi-sup benchmark, tx-learning benchmark
  - Training: figure 1 (support set similar to memory bank but only provides positive samples)
  - Similarity across previously seen objects! (tricky implementation)! Initialization!!
  - Figure 2: details and key difference with others! (requires support set!)
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Radford, Alec, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry et al. "Learning transferable visual models from natural language supervision." arXiv preprint arXiv:2103.00020 (2021).
  - CLIP (contrastive learning from language image pretraining)
  - RQ: Can we reduce the necessity for requiring additional labelled data!
  - TP: predict caption and the images (!self-supervised task) [self-supervised task design]
    - Learning image representation from the text
    - Result matches SOTA
  - Related works: VirTex, ICMLM, ConVIRT

1. Tosh, Christopher, Akshay Krishnamurthy, and Daniel Hsu. "Contrastive learning, multi-view redundancy, and linear models." In Algorithmic Learning Theory, pp. 1179-1206. PMLR, 2021.
  - TP: Theory for contrastive learning in the multi-view setting, where two views of each datum are available.
  - learned representations are nearly optimal on downstream prediction tasks whenever the two views provide **redundant information** about the label.
    - what is redundant information!! := whenever the best linear prediction of the label on each individual view is nearly as good as the best linear prediction of the label when both views are used together.

1. Teng, Jiaye, Weiran Huang, and Haowei He. "Can pretext-based self-supervised learning be boosted by downstream data? a theoretical analysis." arXiv preprint arXiv:2103.03568 (2021).
  - Highly theoretical paper.
  - whether we can make the CI (conditional independence) condition hold by using downstream data to refine the unlabeled data to boost self-supervised learning [*Not always true though it seems intuitive*]
    - Shows some result to prove the counter intuitive results (Hence focusing the importance of the conditional independence)
    - Can we make the CI condition hold with the help of downstream data to boost self-supervised learning? (as CI rearly helds)
    - Validate self-supervised approach (not to used fine-tune data during pretraining)

1. Ren, Xuanchi, Tao Yang, Yuwang Wang, and Wenjun Zeng. "Do Generative Models Know Disentanglement? Contrastive Learning is All You Need." arXiv preprint arXiv:2102.10543 (2021).

1. Jing, Li, Pascal Vincent, Yann LeCun, and Yuandong Tian. "Understanding Dimensional Collapse in Contrastive Self-supervised Learning." arXiv preprint arXiv:2110.09348 (2021).
  - Discusses about dimension collapse ( the embedding vectors end up spanning a lower-dimensional subspace instead of the entire available embedding space.)
  - key findings.
    - along the feature direction where data augmentation variance is larger than the data distribution variance, the weight collapses. (strong augmentation along feature dimensions)
    - even if the covariance of data augmentation has a smaller magnitude than the data variance along all dimensions, the weight will still collapse due to the interplay of weight matrices at different layers known as [implicit regularization](http://old.ins.sjtu.edu.cn/files/paper/20200722191948_slide.pdf). (implicit regularization driving models toward low-rank solutions)
  - looks into the covariance matrix (their eigen decomposition; some eigenvalues become zeros) to measure dimensional collapse
  - Theoretically justify the importance of the projection layers.
  - TP: proposed DirectCLR (uses direct Optimization on the representation layer by selecting a subsection of the layer for loss calculation. remove the requirement of trainable projection. )

1. Sordoni, Alessandro, Nouha Dziri, Hannes Schulz, Geoff Gordon, Philip Bachman, and Remi Tachet Des Combes. "Decomposed Mutual Information Estimation for Contrastive Representation Learning." In International Conference on Machine Learning, pp. 9859-9869. PMLR, 2021.

1. Ryali, Chaitanya K., David J. Schwab, and Ari S. Morcos. "Leveraging background augmentations to encourage semantic focus in self-supervised contrastive learning." arXiv preprint arXiv:2103.12719 (2021).
  - This Paper: Image augmentation regarding the subject and background relationship - "background Augmentation" [sampling method for CL]
    - How they separate the subject background in the first places!! What prior knowledge!!
    - May use different existing methods!!
  - Augmentation Scheme: Another data engineering
    - Used with methods like BYOL, SwAV, MoCo to push SOTA forward
    - Figure 1: Shows all

1. Tosh, Christopher, Akshay Krishnamurthy, and Daniel Hsu. "Contrastive estimation reveals topic posterior information to linear models." Journal of Machine Learning Research 22, no. 281 (2021): 1-31.

1. Ericsson, Linus, Henry Gouk, Chen Change Loy, and Timothy M. Hospedales. "Self-Supervised Representation Learning: Introduction, Advances and Challenges." arXiv preprint arXiv:2110.09327 (2021).
  - Provides a good workflow for selfsupervised Learning
    - extractor function, classifer function, pretext output function (good terminology)
  - (Four pretext): masked prediction, transformation prediction, instance discrimination, and clustering
    - Masked prediction: context can be used to infer some types of missing information in the data if the domain is well-modeled.

## 2022

1. Hoffmann, David T., Nadine Behrmann, Juergen Gall, Thomas Brox, and Mehdi Noroozi. "Ranking Info Noise Contrastive Estimation: Boosting Contrastive Learning via Ranked Positives." arXiv preprint arXiv:2201.11736 (2022).
  - TP: New loss function: Ranking Info Noise Contrastive Estimation (RINCE) (modified InfoNCE losses)
    - preserved the ranked ordering of the InfoNCE loss. (not all negatives are equal)
      - Okay: Now how?: Does it use the supervised information!!!
        - Requires a strong notion of hierarchical positive sets!!
        - Very irritating formulation [equation 5] - learn to avoid such
        - Variation of temperature allows ranking (weight of the losses)
      - Application: Class similarities and gradual changes in video (!!)

1. Lee, Yoonho, Huaxiu Yao, and Chelsea Finn. "Diversify and Disambiguate: Learning From Underspecified Data." arXiv preprint arXiv:2202.03418 (2022).
    - DivDis: Two stage framework (learn diverse hypothesis [pretraining] and Disambiguate by selecting one of these hypothesis [fine tune])
    - Loss function of diversify is interesting

1. Goyal, Priya, Quentin Duval, Isaac Seessel, Mathilde Caron, Mannat Singh, Ishan Misra, Levent Sagun, Armand Joulin, and Piotr Bojanowski. "Vision Models Are More Robust And Fair When Pretrained On Uncurated Images Without Supervision." arXiv preprint arXiv:2202.08360 (2022).
  - Utilized Uncurated Image data across the globe [LOL! GPU please] (fair group)
    - Dense 10 Million model params
    - 50 benchmarks including fairness, robustness to distribution shift, geographical diversity, fine grained recognition, image copy detection and many image classification datasets
  - obvious result: the more data the better
  - key findings: self-supervised learning on random internet data leads to models that are more fair, less biased and less harmful

1. Saunshi, Nikunj, Jordan Ash, Surbhi Goel, Dipendra Misra, Cyril Zhang, Sanjeev Arora, Sham Kakade, and Akshay Krishnamurthy. "Understanding Contrastive Learning Requires Incorporating Inductive Biases." arXiv preprint arXiv:2202.14037 (2022).
  - Argues that requires to include **inductive biases of function class and training algorithm** to explain the success of CL with previous attempts (augmentation and loss function.) [algorithmic]
  - Key findings:  ignoring the architecture and the training algorithm can make the current theoretical analyses of contrastive learning vacuous
    - Theoretical result: incorporating inductive *biases of the function class (ResNet/VGGNet)* allows contrastive learning to work with less stringent conditions compared to prior analyses
    - different function classes and algorithms (SGD/ADAM) behave very differently on downstream tasks,
  - Some outcoms
    -  Is the contrastive loss indeed a good indicator of downstream performance? **No**
    - Do augmentations overlap sufficiently enough in practice to explain the success of contrastive learning? **NO**
    - Can contrastive learning succeed even when there is little to no overlap? **YES**
  - Three key terms:
    - i) Function class sensitivity (encoder architecture and optimization)
    - ii) Brittleness of transfer (sometime CL loss reduction cause detrimental impact)
    - iii) The disjoint augmentation regime: (when augmentation distribution do not overlap it may lead to false sense of security [may not work for downstream task, especially using linear classifier head]) [figure 1].
      - Details in section 3 example
    - Easy but interesting theory in lemma 4.1
      - if augmentation distribution is disjoint then even if contrastive loss is low, the downstream classification loss can be high.

1. Lee, Hyungtae, and Heesung Kwon. "Self-supervised Contrastive Learning for Cross-domain Hyperspectral Image Representation." arXiv preprint arXiv:2202.03968 (2022).
  - Method: Contrative learning for Hyperspectral image
    - uses a property of Hyperspectral: same area's image are in same class
  - Network: Cross-Domain CNN

1. Baevski, Alexei, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. "Data2vec: A general framework for self-supervised learning in speech, vision and language." arXiv preprint arXiv:2202.03555 (2022).
  - TP: target for generalized SSL
    - core idea is to predict latent representations of the full input data based on a masked view of the input in a self distillation (Transformer architecture)
    -  data2vec predicts contextualized **latent representations** that contain information from the entire input
  - Masked prediction with latent target representation.
  - Methods:  1. transformer architecture (Vit-B for vision tasks, ) 2. Masking input 3.  Training target
  - Interesing discussion and ablation studies section.

## 2014 and previous

1. Goldberger, Jacob, Geoffrey E. Hinton, Sam Roweis, and Russ R. Salakhutdinov. "Neighbourhood components analysis." Advances in neural information processing systems 17 (2004).
  - Non-parametric classification model (Early projection)
    - application: Dimensional reduction and metric learning.
  - Simple metric based learning (projection and check distance with the same group):
  - The neighborhood class instance should be closure.
  - The non-similar instance should be projected far away.

1. Divvala, Santosh K., Ali Farhadi, and Carlos Guestrin. "Learning everything about anything: Webly-supervised visual concept learning." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3270-3277. 2014.
  - a fully-automated approach for learning extensive models for a wide range of variations (e.g. actions, interactions, attributes and beyond) within any concept

1. Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality reduction by learning an invariant mapping." In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06), vol. 2, pp. 1735-1742. IEEE, 2006.
  - Algorithm 1: distance based loss optimization [algorithmic]

1. Bromley, Jane, J. W. Bentz, L. Bottou, I. Guyon, Y. LeCun, C. Moore, E. Sackinger, and R. Shah. "Signature Veriﬁcation using a “Siamese” Time Delay Neural Network." Int.]. Pattern Recognit. Artzf Intell 7 (1993).
  - Gold in old
  - Siamese network early application for hand writing

1. Becker, Suzanna, and Geoffrey E. Hinton. "Self-organizing neural network that discovers surfaces in random-dot stereograms." Nature 355, no. 6356 (1992): 161-163.

1. G. W. Taylor, I. Spiro, C. Bregler, and R. Fergus, ‘‘Learning invariance through imitation,’’ in Proc. CVPR, Jun. 2011, pp. 2729–2736, doi:10.1109/CVPR.2011.5995538

1. Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

1. Gutmann, Michael, and Aapo Hyvärinen. "Noise-contrastive estimation: A new estimation principle for unnormalized statistical models." In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pp. 297-304. 2010.

1. Weinberger, Kilian Q., John Blitzer, and Lawrence K. Saul. "Distance metric learning for large margin nearest neighbor classification." In Advances in neural information processing systems, pp. 1473-1480. 2006.
  - Triplet loss proposal