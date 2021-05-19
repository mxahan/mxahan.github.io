# Introduction

This blog contains state of the art application and research on different applications. The applications will be stated in the subheadings. Although one paper can fall into multiple categories we will be mostly going with one tag per papers. However, the tag is subjected to change time-to-times.

# Multi View Application

# Few-Shot learning

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

1. 1. Vinyals, Oriol, Charles Blundell, Timothy Lillicrap, and Daan Wierstra. "Matching networks for one shot learning." In Advances in neural information processing systems, pp. 3630-3638. 2016.
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

1. 1. Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." In International Conference on Machine Learning, pp. 1126-1135. PMLR, 2017.
  - learning and adapting quickly
    - The key idea: train the initial parameters to maximize performance on a new task after the parameters have been updated through few gradient steps computed with a small data from that new task.
    - Where it works! Why don't everybody used it afterwords!
    - Contribution: trains a model’s parameters such that a small number of gradient updates will lead to fast learning on a new task.
  - Prior arts: Learn update function!
    - MAML: more flexible (loss function and architectures)
  - MAML
    - Problem setup:

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
1. Piergiovanni, A. J., and Michael S. Ryoo. "Recognizing Actions in Videos from Unseen Viewpoints." arXiv preprint arXiv:2103.16516 (2021).

1. van Amsterdam, Beatrice, Matthew J. Clarkson, and Danail Stoyanov. "Multi-task recurrent neural network for surgical gesture recognition and progress prediction." In 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 1380-1386. IEEE, 2020.

# Imitation Learning

1. Sermanet, Pierre, Kelvin Xu, and Sergey Levine. "Unsupervised perceptual rewards for imitation learning." arXiv preprint arXiv:1612.06699 (2016).
  - intermediate steps and sub-goals
  - Abstraction power of intermediate visualization!!
  - pretraining, unsupervised [results a reward function !! ]
  - Accumulate two ideas for imitation learning: usage of extensive prior knowledge!! & observation, trial-and-error learning !!
  - This paper: proposes reward learning method
    - Reward function (vision based learning) and discovering intermediate stages
    - learning visual representations can be used to represent the goal (no fine-tuneing)
  - Related field: learning from demonstration, inverse reinforcement learning
  - video segmentation for notion of similarity

1. Ziebart, Brian D., Andrew L. Maas, J. Andrew Bagnell, and Anind K. Dey. "Maximum entropy inverse reinforcement learning." In Aaai, vol. 8, pp. 1433-1438. 2008.
  - Framing problem as Imitation learning as solutions to MDP
  - Model navigation and driving behavior -> under noisy data
  - Issues in imitation learning during the imperfect demonstration
    - alternative solution: IRL : Matching feature count (algo 1)
