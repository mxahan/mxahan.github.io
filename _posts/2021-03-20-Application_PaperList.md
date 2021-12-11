# Introduction

This blog contains state of the art application and research on different applications. The applications will be stated in the subheadings. Although one paper can fall into multiple categories we will be mostly going with one tag per papers. However, the tag is subjected to change time-to-times.

# Multi View Application

# NN Pruning

1. Blalock, Davis, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. "What is the state of neural network pruning?." arXiv preprint arXiv:2003.03033 (2020).

# Few-Shot learning

1. Ramesh, Aditya, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. "Zero-shot text-to-image generation." arXiv preprint arXiv:2102.12092 (2021).

1. Chen, Wei-Yu, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, and Jia-Bin Huang. "A closer look at few-shot classification." arXiv preprint arXiv:1904.04232 (2019).
  - Consistent!  comparative analysis of few shot learning methods! (algorithm performance depends on the backbone networks) [deeper backbone reduces performance difference] **algo comparison varies as backbone network changes**
  - Modified baseline methods (imagenet and CUB dataset)!
  - new experimental setting for the cross domain generalization of FSL
  - Some empirical results (comments on backbone network size! small - then intra-class variation should be low [as expected])

1. Wang, Yaqing, Quanming Yao, James T. Kwok, and Lionel M. Ni. "Generalizing from a few examples: A survey on few-shot learning." ACM Computing Surveys (CSUR) 53, no. 3 (2020): 1-34.
  - Three approaches (i) Data (ii) Model (iii) Algorithm, all of them uses prior knowledge!
    - What is prior knowledge
  - Interesting section that distinguish between FSL with other learning problems (2.2)
    - FSL avails various options as prior knowledge. For example: Weak-supervised (semi-supervised, active learning) learning is a kind of FSL (when prior knowledge is the unlabeled data); alternatively, the prior knowledge may be the learned model (aka transfer learning.) [figure 3]
    - Unreliable empirical risk minimization
  - (i) data prior knowledge
    - Data augmentation - Transforming samples from the labeled training data
    - Transforming data from the unlabeled data (semi-supervised)
    - Transforming data from the similar dataset (GAN)
  - (ii) Model
    - Multitask learning
      - Parameter sharing
      - Parameter tying ? Pairwise difference is penalized
    - Embedding Learning
      - Task specific
      - Task invariant: Matching net, prototypical networks
      - hybrid embedding model
    - External memory: Key-value memory
      - Refining representation
      - Refining parameters
    - Generative modeling
      - Decomposable components
      - Groupwise Shared Prior
      - Parameters of inference networks
  - (iii) Algorithm
    - Refining existing parameters
      - Regularization
      - aggregation
      - Fine-tuning existing parameter
    - Refining Meta-learned parameters
    - Learning the optimizer
  - Future works: four possible directions : Problem setup, Techniques, applications, and Theories

1. Vinyals, Oriol, Charles Blundell, Timothy Lillicrap, and Daan Wierstra. "Matching networks for one shot learning." In Advances in neural information processing systems, pp. 3630-3638. 2016.
  - Metric learning and augmented memory network!!
  - Non-parametric setup for metric learning
  - Two novelty: Modeling & training

1. Munkhdalai, Tsendsuren, and Hong Yu. "Meta networks." In International Conference on Machine Learning, pp. 2554-2563. PMLR, 2017.

1. Dumoulin, Vincent, Neil Houlsby, Utku Evci, Xiaohua Zhai, Ross Goroshin, Sylvain Gelly, and Hugo Larochelle. "Comparing Transfer and Meta Learning Approaches on a Unified Few-Shot Classification Benchmark." arXiv preprint arXiv:2104.02638 (2021).

1. Reed, Scott, Zeynep Akata, Honglak Lee, and Bernt Schiele. "Learning deep representations of fine-grained visual descriptions." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 49-58. 2016.

1. Sung, Flood, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M. Hospedales. "Learning to compare: Relation network for few-shot learning." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1199-1208. 2018.
  - Relation network (RN) - simple! flexible! General! - end-2-end
  - Utilizes meta-learning approach! keeps distance between small set of supports (metric learning based approach), reduces complexity in the inference
  - Example based classification (FSL) and extension to ZSL
  - Experiments with 5 benchmark dataset [2 FS: omniglot, MiniImagenet], [3 ZS: AwA1, AwA2 and CUB]
  - Two branch Relation network for FSL: Embedding module, and relation module (nonlinear comparator, multi-layer NN) **Figure 1**
    - Both modules are meta-learner
  - Interesting related work section : learning to fine-tune (MAML), RNN memory based, Embedding and Metric learning approachs
    - Mostly related to prototypical network and siamese network
  - Problem definition and solution: Section 3
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Dinu, Georgiana, Angeliki Lazaridou, and Marco Baroni. "Improving zero-shot learning by mitigating the hubness problem." arXiv preprint arXiv:1412.6568 (2014).

1. Snell, Jake, Kevin Swersky, and Richard S. Zemel. "Prototypical networks for few-shot learning." arXiv preprint arXiv:1703.05175 (2017).
  - Prototypical networks learn a metric space; classification by computing distances to prototype representations of each class. (non-parametric learning!)
  - This paper: New algorithms!
  - simple design; sometimes (when??) better than the meta learning and complex architecture.
  - Extension to zero-shot??
  - Prior works: Matching networks and episode concepts
    - episodic learning to the meta learning
    - The learning is dependent on the episode selection
  - This paper: tries not to overfit; by using simple inductive bias ???
    - Prototypical Networks: single prototype for each class (classification by finding nearest prototype) - embed the meta-data into shared space
    - works for both few-shot and zero-shot learning
    - how it's differ from the cluster?
  - Episodic training: By selecting some instances randomly from the given example pool
    - Algorithm 1: interesting
  - This paper: uses Bregman Distance function (euclidean distance) to define embedding metric space
  - One shot learning Scenario (single support example): Prototypical network == matching network
    - matching network: weighted nearest neighborhood classifier.
  - Experiments: (i) omniglot few-shot data (ii) MiniImagenet few-shot classification (iii) CUB zero-shot classification.

1. Sun, Qianru, Yaoyao Liu, Tat-Seng Chua, and Bernt Schiele. "Meta-transfer learning for few-shot learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 403-412. 2019.

1. Ren, Mengye, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, and Richard S. Zemel. "Meta-learning for semi-supervised few-shot classification." arXiv preprint arXiv:1803.00676 (2018).
  - Extension (??) of Prototypical networks
    - Application scenario: partially labeled data, some classes are missing labeled section.
  - Experiment with omniglot and MiniImagenet , tieredImageNet (new proposed data!)
    - With and without distractor classes
    - Three novel extension of prototypical network!
      - (i) Prototypical net with soft k-Means
      - (ii) Prototypical net with soft k-Means with a distractor cluster
      - (iii) Prototypical net with soft k-Means and Masking
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." In International Conference on Machine Learning, pp. 1126-1135. PMLR, 2017.
  - learning and adapting quickly
    - The key idea: train the initial parameters to maximize performance on a new task after the parameters have been updated through few gradient steps computed with a small data from that new task.
    - Where it works! Why don't everybody used it afterwords!
    - Contribution: trains a model’s parameters such that a small number of gradient updates will lead to fast learning on a new task.
  - Prior arts: Learn update function!
    - MAML: more flexible (loss function and architectures)
  - MAML: Problem setup:

1. Weston, Jason, Sumit Chopra, and Antoine Bordes. "Memory networks." arXiv preprint arXiv:1410.3916 (2014).

# UAV

1. Narayanan, Priya, Christoph Borel-Donohue, Hyungtae Lee, Heesung Kwon, and Raghuveer Rao. "A real-time object detection framework for aerial imagery using deep neural networks and synthetic training images." In Signal Processing, Sensor/Information Fusion, and Target Recognition XXVII, vol. 10646, p. 1064614. International Society for Optics and Photonics, 2018.
  - (key point): use of synthetic images (video games)
    - Data augmentation
    - How much it covers the real scenario and cases
    - Are pixel distribution same as the real-time scenario!
    - How about coordinating between sensors & viewpoints
    - Requires all labeled !!
  - Detection approaches (YOLO, SSD)
  - Edge device implementation

1. Samaras, Stamatios, Eleni Diamantidou, Dimitrios Ataloglou, Nikos Sakellariou, Anastasios Vafeiadis, Vasilis Magoulianitis, Antonios Lalas et al. "Deep learning on multi sensor data for counter UAV applications—A systematic review." Sensors 19, no. 22 (2019): 4837.
  - Counter UAV (c-UAV): detect UAV to prevent criminal activities.
    - Discusses recent advancement of c-UAV using multi-sensors.
  - Figure 1: Good comparison among multiple individual sensors
  - Scope: Figure 3 {4 types of sensors and their fusion strategy}

# Action recognition and Progression

1. Ionescu, Catalin, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. "Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments." IEEE transactions on pattern analysis and machine intelligence 36, no. 7 (2013): 1325-1339.

1. Piergiovanni, A. J., and Michael S. Ryoo. "Recognizing Actions in Videos from Unseen Viewpoints." arXiv preprint arXiv:2103.16516 (2021).

1. van Amsterdam, Beatrice, Matthew J. Clarkson, and Danail Stoyanov. "Multi-task recurrent neural network for surgical gesture recognition and progress prediction." In 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 1380-1386. IEEE, 2020.

1. Zhu, Yi, Xinyu Li, Chunhui Liu, Mohammadreza Zolfaghari, Yuanjun Xiong, Chongruo Wu, Zhi Zhang, Joseph Tighe, R. Manmatha, and Mu Li. "A Comprehensive Study of Deep Video Action Recognition." arXiv preprint arXiv:2012.06567 (2020).
  - Popular dataset descriptions
  -

1. Jenni, Simon, and Paolo Favaro. "Self-Supervised Multi-View Synchronization Learning for 3D Pose Estimation." In Proceedings of the Asian Conference on Computer Vision. 2020.
  - Experiment Data: Human3.6M dataset, ResNet architectures
  - Self-supervised task: Transformation and synchronous prediction

1. Wang, Keze, Liang Lin, Chenhan Jiang, Chen Qian, and Pengxu Wei. "3D human pose machines with self-supervised learning." IEEE transactions on pattern analysis and machine intelligence 42, no. 5 (2019): 1069-1082.
  - Experiment data: HumanEva-I and Human3.6M

1. heng, Ce, Wenhan Wu, Taojiannan Yang, Sijie Zhu, Chen Chen, Ruixu Liu, Ju Shen, Nasser Kehtarnavaz, and Mubarak Shah. "Deep learning-based human pose estimation: A survey." arXiv preprint arXiv:2012.13392 (2020).

1. Kocabas, Muhammed, Salih Karagoz, and Emre Akbas. "Self-supervised learning of 3d human pose using multi-view geometry." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1077-1086. 2019.

# Disentangled Representation learning

1. Do, Kien, and Truyen Tran. "Theory and evaluation metrics for learning disentangled representations." arXiv preprint arXiv:1908.09961 (2019).

1. Zhu, Yizhe, Martin Renqiang Min, Asim Kadav, and Hans Peter Graf. "S3VAE: Self-supervised sequential VAE for representation disentanglement and data generation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6538-6547. 2020.
  -

1. Zamir, Amir R., Alexander Sax, William Shen, Leonidas J. Guibas, Jitendra Malik, and Silvio Savarese. "Taskonomy: Disentangling task transfer learning." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3712-3722. 2018.
  - Encoder decoder (VAE) set up for different self-supervised pretext tasks and experimentation  on the transfer learning.
  - Provide a good benchmark to transfer one task learning to other tasks. [relationship among tasks]
  - Provided their own dataset

1. Hu, Jie, Liujuan Cao, Tong Tong, Qixiang Ye, Shengchuan Zhang, Ke Li, Feiyue Huang, Ling Shao, and Rongrong Ji. "Architecture disentanglement for deep neural networks." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 672-681. 2021.

1. Arjovsky, Martin, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. "Invariant risk minimization." arXiv preprint arXiv:1907.02893 (2019).

1. Locatello, Francesco, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. "Challenging common assumptions in the unsupervised learning of disentangled representations." In international conference on machine learning, pp. 4114-4124. PMLR, 2019.
  - Requires inductive bias (implicit supervision)
  - (i) The role of inductive biases and implicit and explicit supervision should be made explicit: unsupervised model selection persists as a key question. (ii) practical benefits of enforcing a specific notion of disentanglement of the learned representations should be demonstrated.

1. Wang, Tan, Zhongqi Yue, Jianqiang Huang, Qianru Sun, and Hanwang Zhang. "Self-Supervised Learning Disentangled Group Representation as Feature." arXiv preprint arXiv:2110.15255 (2021).
  - Higgins definition of disentanglement representation
  - Group theoretic approach [Learn distinguished matrix for each tasks]

1. Higgins, Irina, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, and Alexander Lerchner. "Towards a definition of disentangled representations." arXiv preprint arXiv:1812.02230 (2018).
  - Formal *Definition(s)* of the disentangled representation learning
  - A vector representation is called a disentangled representation with respect to a particular decomposition of a symmetry group into subgroups, if it decomposes into **independent subspaces**, where each subspace is affected by the action of a single subgroup, and the actions of all other subgroups leave the subspace unaffected.

1. Wu, Zhenyu, Karthik Suresh, Priya Narayanan, Hongyu Xu, Heesung Kwon, and Zhangyang Wang. "Delving into robust object detection from unmanned aerial vehicles: A deep nuisance disentanglement approach." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1201-1210. 2019.
  - Utilization of meta-data
  - Multitask setup

1. Maziarka, Łukasz, Aleksandra Nowak, Maciej Wołczyk, and Andrzej Bedychaj. "On the relationship between disentanglement and multi-task learning." arXiv preprint arXiv:2110.03498 (2021).
  - TP conclusion: a hard-parameter sharing scenario multi-task learning indeed seems to encourage disentanglement, however, it is inconclusive whether disentangled representations have a clear positive impact on the models performance.

1. Yin, Xi, and Xiaoming Liu. "Multi-task convolutional neural network for pose-invariant face recognition." IEEE Transactions on Image Processing 27, no. 2 (2017): 964-975.


# Imitation Learning

1. Sermanet, Pierre, Kelvin Xu, and Sergey Levine. "Unsupervised perceptual rewards for imitation learning." arXiv preprint arXiv:1612.06699 (2016).
  - intermediate steps and sub-goals
  - Abstraction power of intermediate visualization!!
  - pretraining, unsupervised [results a reward function !! ]
  - Accumulate two ideas for imitation learning: usage of extensive prior knowledge!! & observation, trial-and-error learning !!
  - This paper: proposes reward learning method
    - Reward function (vision based learning) and discovering intermediate stages
    - learning visual representations can be used to represent the goal (no fine-tuning)
  - Related field: learning from demonstration, inverse reinforcement learning
  - video segmentation for notion of similarity
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Ziebart, Brian D., Andrew L. Maas, J. Andrew Bagnell, and Anind K. Dey. "Maximum entropy inverse reinforcement learning." In Aaai, vol. 8, pp. 1433-1438. 2008.
  - Framing problem as Imitation learning as solutions to MDP
  - Model navigation and driving behavior -> under noisy data
  - Issues in imitation learning during the imperfect demonstration
    - alternative solution: IRL : Matching feature count (algo 1)
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

# Architectural contributions

1. Chollet, François. "Xception: Deep learning with depthwise separable convolutions." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1251-1258. 2017.

1. Zhang, Xiangyu, Xinyu Zhou, Mengxiao Lin, and Jian Sun. "Shufflenet: An extremely efficient convolutional neural network for mobile devices.(2017)." arXiv preprint arXiv:1707.01083 (2017).

1. He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning for image recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

1. Feichtenhofer, Christoph, Haoqi Fan, Jitendra Malik, and Kaiming He. "Slowfast networks for video recognition." In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6202-6211. 2019.
  - Motivation: No need to treat spatial (semantics) and temporal dimension equally (Retinal ganglion cells)
  - TP: Two pathway model (Slow {low fps} and fast {high fps})
    - slow is the heavyweight computation (higher number of channel) and fast is the lightweight computation (lower number of channel)
    - Two pathway fused by lateral connections (fast to slow)
  - Experiments: kinetics, charades, AVA dataset
  - Nice ablation study sections

1. Wang, Xiaolong, Ross Girshick, Abhinav Gupta, and Kaiming He. "Non-local neural networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7794-7803. 2018.
  - Computes the response at a position as a weighted sum of the features at all positions.
    - Plugged in with may static image recognition system (pose, object detections).
    - Aim to capture long-term dependencies
  - Figure 2 sums all (design choice of f and g)

1. Feichtenhofer, Christoph. "X3d: Expanding architectures for efficient video recognition." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 203-213. 2020.
  - progressive forward expansion and backward contraction approaches!!

1. Neimark, Daniel, Omri Bar, Maya Zohar, and Dotan Asselmann. "Video transformer network." arXiv preprint arXiv:2102.00719 (2021).

1. Kondratyuk, Dan, Liangzhe Yuan, Yandong Li, Li Zhang, Mingxing Tan, Matthew Brown, and Boqing Gong. "Movinets: Mobile video networks for efficient video recognition." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16020-16030. 2021.

1. Howard, Andrew G., Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).

1. Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for mobilenetv3." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1314-1324. 2019.

1. Jaegle, Andrew, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. "Perceiver: General perception with iterative attention." arXiv preprint arXiv:2103.03206 (2021).

1. Jaegle, Andrew, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula et al. "Perceiver IO: A General Architecture for Structured Inputs & Outputs." arXiv preprint arXiv:2107.14795 (2021).

1. Qiu, Zhaofan, Ting Yao, and Tao Mei. "Learning spatio-temporal representation with pseudo-3d residual networks." In proceedings of the IEEE International Conference on Computer Vision, pp. 5533-5541. 2017.
  - TP: proposes computational efficient 3D CNN (and their extensions)
    - Decomposes 3D CNN as DD spatial filter and 1D temporal filter

1. Xie, Saining, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. "Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification." In Proceedings of the European conference on computer vision (ECCV), pp. 305-321. 2018.

1. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012): 1097-1105.

1. Srivastava, Nitish, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. "Dropout: a simple way to prevent neural networks from overfitting." The journal of machine learning research 15, no. 1 (2014): 1929-1958.

1. Radosavovic, Ilija, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. "Designing network design spaces." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10428-10436. 2020.
  -

1. Tolstikhin, Ilya, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung et al. "Mlp-mixer: An all-mlp architecture for vision." arXiv preprint arXiv:2105.01601 (2021).

1. Xie, Saining, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. "Aggregated residual transformations for deep neural networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1492-1500. 2017.
  - ResNeXt: highly modulated network (figure 1) [another network engineering with a very simple idea]
  - Name origin: as it adds **Next dimension: Cardinality** so it refers to ResNeXt
    - Not Inception as no downsampling in between.
  - Why? how is it better than the ResNet? Introduces new parameters " Cardinality (the size of set of transformation)"! [with addition of deeper and wider in ResNet]
  - TP: extends *split-transform-merge* and ResNet
  - TP: Transformation to Low dimension and outputs are *aggregated* by summation.
  - Related works: Multi-branch CN, grouped CN, compressed CN, Ensembling.
  - Experiments with ImageNet, COCO object detection. [outperforms ResNet, Inception, VGG.]

1. Radosavovic, Ilija, Justin Johnson, Saining Xie, Wan-Yen Lo, and Piotr Dollár. "On network design spaces for visual recognition." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1882-1890. 2019.

1. Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018).
  - Architectural search in Differentiable manner (why? How? )
    - Why: differential search space for gradient descent in architecture search (using some relaxations), faster than non-differential counterparts (RL based)!
  - Experiments with both images and NLP tasks.  

1. Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
  - CNN matching performance: large amount of data pretrain and transfer on mid/small sized classification tasks.

1. Tran, Du, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. "A closer look at spatiotemporal convolutions for action recognition." In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 6450-6459. 2018.
 - In github notes
 - factorizing the 3D convolutional filters into separate spatial and temporal components yields significantly gains in accuracy
 - Empirical studies (?? types of studies) leads to design choices R(2+1)D
 - 3D network with residual learning
 - Benefits:

1. Liu, Ze, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. "Video Swin Transformer." arXiv preprint arXiv:2106.13230 (2021).

1. Ke, Junjie, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. "MUSIQ: Multi-scale Image Quality Transformer." arXiv preprint arXiv:2108.05997 (2021).

1. Anonymous, . "Patches Are All You Need?." . In Submitted to The Tenth International Conference on Learning Representations .2022.

# Data augmentation

1. Liu, Zhuang, Tinghui Zhou, Hung-Ju Wang, Zhiqiang Shen, Bingyi Kang, Evan Shelhamer, and Trevor Darrell. "Transferable recognition-aware image processing." arXiv preprint arXiv:1910.09185 (2019).
  - optimizing the recognition loss directly on the image processing network or through an intermediate transforming model (transferable)
    -a neural network for image processing: maps an input image to an output image with some desired properties.
  - propose add a recognition loss optimized jointly with the image processing loss.
  - Figure 2: sums the work. (process loss + recognition task)

1. Talebi, Hossein, and Peyman Milanfar. "Learning to Resize Images for Computer Vision Tasks." arXiv preprint arXiv:2103.09950 (2021).
  - Does exactly what it's saying!! learn to resize using CNN layer for better downstream Tasks (proposes a resizer block )
    - Why in the first place!  
    - What will be the loss function for resize? (architecture: Figure 3)
    - Wait: are the extracting another feature! (calling it as resizer!)
  - ImageNet experiment, uses different model

1. Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).

1. Cubuk, Ekin D., Barret Zoph, Jonathon Shlens, and Quoc V. Le. "Randaugment: Practical automated data augmentation with a reduced search space." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 702-703. 2020.
  - TP tackles: (i) Reduce search spaces for automatic data augmentations! {by Simplifying the search spaces: avails grid search} (ii) adjust regularization strengths
  - Experimentation with the role of data augmentations.
  - Related works: Learning automatic data augmentations! AutoAugment RL basede methods! Population based augmentation,
  - Interesting empirical results: data augmentation depends on model sizes and dataset training set sizes. Experiments with cifar, SVHN and ImageNET, object detection.
  - Methods:

1. Cubuk, Ekin D., Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V. Le. "Autoaugment: Learning augmentation policies from data." arXiv preprint arXiv:1805.09501 (2018).

# Metrics and Losses

1. Sainburg, Tim, Leland McInnes, and Timothy Q. Gentner. "Parametric UMAP embeddings for representation and semi-supervised learning." arXiv preprint arXiv:2009.12981 (2020).
  - Good starting note for the UMAP and tSNE. Parametric extension for the UMAP
  - [link](https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668) May contain Some bias towards UMAP

1. Sun, Yifan, Changmao Cheng, Yuhan Zhang, Chi Zhang, Liang Zheng, Zhongdao Wang, and Yichen Wei. "Circle loss: A unified perspective of pair similarity optimization." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6398-6407. 2020.

1. Lin, Tsung-Yi, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. "Focal loss for dense object detection." In Proceedings of the IEEE international conference on computer vision, pp. 2980-2988. 2017.

# Unsorted Papers

1. Cheung, Brian, Alex Terekhov, Yubei Chen, Pulkit Agrawal, and Bruno Olshausen. "Superposition of many models into one." arXiv preprint arXiv:1902.05522 (2019).
  - Multiply weights to project them in orthogonal space and sum them.
