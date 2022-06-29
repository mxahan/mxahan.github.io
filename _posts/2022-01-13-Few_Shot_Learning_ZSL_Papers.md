# Few-Shot learning and ZSL

## 2021

1. Chen, Da, Yuefeng Chen, Yuhong Li, Feng Mao, Yuan He, and Hui Xue. "Self-supervised learning for few-shot image classification." In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1745-1749. IEEE, 2021.
  - TP: apply a much larger embedding network with self-supervised learning (SSL) to incorporate with episodic task based meta-learning.
    - WTF: Pretraining and then Meta-learning!!

1. Ramesh, Aditya, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. "Zero-shot text-to-image generation." arXiv preprint arXiv:2102.12092 (2021).

1. Dumoulin, Vincent, Neil Houlsby, Utku Evci, Xiaohua Zhai, Ross Goroshin, Sylvain Gelly, and Hugo Larochelle. "Comparing Transfer and Meta Learning Approaches on a Unified Few-Shot Classification Benchmark." arXiv preprint arXiv:2104.02638 (2021).



## 2020

1. Du, Xianzhi, Tsung-Yi Lin, Pengchong Jin, Golnaz Ghiasi, Mingxing Tan, Yin Cui, Quoc V. Le, and Xiaodan Song. "SpineNet: Learning scale-permuted backbone for recognition and localization." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11592-11601. 2020.
  - Meta learning, NAS
  - how does Scale decreased model work as backbone in object detection?
  - Propose scale permuted network

1. Wang, Yaqing, Quanming Yao, James T. Kwok, and Lionel M. Ni. "Generalizing from a few examples: A survey on few-shot learning." ACM Computing Surveys (CSUR) 53, no. 3 (2020): 1-34.
  - Three approaches (i) Data (ii) Model (iii) Algorithm, all of them uses prior knowledge!
    - What is prior knowledge!!
  - Interesting section that distinguish between FSL with other learning problems (2.2)
    - FSL avails various options as prior knowledge. For example: Weak-supervised (semi-supervised, active learning) learning is a kind of FSL (when prior knowledge is the unlabeled data); alternatively, the prior knowledge may be the learned model (aka transfer learning.) [figure 3]
    - Unreliable empirical risk minimization
  - (i) data prior knowledge
    - Data augmentation - Transforming samples from the labeled training data
    - Transforming data from the unlabeled data (semi-supervised)
    - Transforming data from the similar dataset (GAN)
  - (ii) Model
    - Multitask learning [i. Parameter sharing ii. Parameter tying ? Pairwise] difference is penalized
    - Embedding Learning [i. Task specific ii. Task invariant: Matching net, prototypical networks iii. hybrid embedding model]
    - External memory: Key-value memory [i. Refining representation ii. Refining parameters]
    - Generative modeling [i. Decomposable components ii. Groupwise Shared Prior iii. Parameters of inference networks]
  - (iii) Algorithm
    - Refining existing parameters [i. Regularization ii. aggregation iii. Fine-tuning existing parameter]
    - Refining Meta-learned parameters
    - Learning the optimizer
  - Future works: four possible directions : Problem setup, Techniques, applications, and Theories

## 2019

1. Bennequin, Etienne. "Meta-learning algorithms for few-shot computer vision." arXiv preprint arXiv:1909.13579 (2019).
  - meta-learning algorithms, i.e. algorithms that learn to learn
  - TP: Meta learning review (N-way K-shot image classification) reviews
      - Support set (N number of classes, K example per class)
      - Query set, Q
      - Base-dataset, they are different from support set classes.
  - Solution: i) Memory augmented networks ii) Metric learning (there are some approach combined with meta-learning) iii) Gradient based meta-learner iv) data generation  
  - Meta-learning Definition: given a task, an algorithm is learning “if its performance at the task improves with experience”, while, *given a family of tasks*, an algorithm is learning to learn if “its performance at each task *improves with experience and with the number of tasks*”: referred as a meta-learning algorithm.
  -

1. Chen, Wei-Yu, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, and Jia-Bin Huang. "A closer look at few-shot classification." arXiv preprint arXiv:1904.04232 (2019).
  - Consistent!  comparative analysis of few shot learning methods! (algorithm performance depends on the backbone networks) [deeper backbone reduces performance difference] **algo comparison varies as backbone network changes**
  - Modified baseline methods (imagenet and CUB dataset)!
  - new experimental setting for the cross domain generalization of FSL
  - Some empirical results (comments on backbone network size! small - then intra-class variation should be low [as expected])

1. Sun, Qianru, Yaoyao Liu, Tat-Seng Chua, and Bernt Schiele. "Meta-transfer learning for few-shot learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 403-412. 2019.




## 2018

1. Ren, Mengye, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, and Richard S. Zemel. "Meta-learning for semi-supervised few-shot classification." arXiv preprint arXiv:1803.00676 (2018).
  - Extension (??) of Prototypical networks
    - Application scenario: partially labeled data, some classes are missing labeled section.
  - Experiment with omniglot and MiniImagenet , tieredImageNet (new proposed data!)
    - With and without distractor classes
    - Three novel extension of prototypical network!
      - (i) Prototypical net with soft k-Means
      - (ii) Prototypical net with soft k-Means with a distractor cluster
      - (iii) Prototypical net with soft k-Means and Masking
  - <embed src="https://mxahan.github.io/PDF_files/Meta_learning_SSFSL.pdf" width="100%" height="850px"/>

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
  - <embed src="https://mxahan.github.io/PDF_files/relation_net.pdf" width="100%" height="850px"/>

## 2017 and Earlier

1. Santoro, Adam, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. "Meta-learning with memory-augmented neural networks." In International conference on machine learning, pp. 1842-1850. PMLR, 2016.

1. Vinyals, Oriol, Charles Blundell, Timothy Lillicrap, and Daan Wierstra. "Matching networks for one shot learning." In Advances in neural information processing systems, pp. 3630-3638. 2016.
  - Metric learning and augmented memory network!!
  - Non-parametric setup for metric learning
  - Two novelty: Modeling & training

1. Munkhdalai, Tsendsuren, and Hong Yu. "Meta networks." In International Conference on Machine Learning, pp. 2554-2563. PMLR, 2017.

1. Reed, Scott, Zeynep Akata, Honglak Lee, and Bernt Schiele. "Learning deep representations of fine-grained visual descriptions." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 49-58. 2016.

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
  - <embed src="https://mxahan.github.io/PDF_files/Prototypical_net.pdf" width="100%" height="850px"/>


1. Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." In International Conference on Machine Learning, pp. 1126-1135. PMLR, 2017.
  - learning and adapting quickly (MAML)
    - The key idea: train the initial parameters to maximize performance on a new task after the parameters have been updated through few gradient steps computed with a small data from that new task.
    - Where it works! Why don't everybody used it afterwords!
    - Contribution: trains a model’s parameters such that a small number of gradient updates will lead to fast learning on a new task.
  - Prior arts: Learn update function!
    - MAML: more flexible (loss function and architectures)
  - MAML: Problem setup:

1. Weston, Jason, Sumit Chopra, and Antoine Bordes. "Memory networks." arXiv preprint arXiv:1410.3916 (2014).
