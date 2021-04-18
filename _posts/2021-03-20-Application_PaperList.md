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
  - Future works: four possible directions :
    - Problem setup
    - Techniques
    - applications
    - Theories

1. Munkhdalai, Tsendsuren, and Hong Yu. "Meta networks." In International Conference on Machine Learning, pp. 2554-2563. PMLR, 2017.

1. Dumoulin, Vincent, Neil Houlsby, Utku Evci, Xiaohua Zhai, Ross Goroshin, Sylvain Gelly, and Hugo Larochelle. "Comparing Transfer and Meta Learning Approaches on a Unified Few-Shot Classification Benchmark." arXiv preprint arXiv:2104.02638 (2021).

1. Reed, Scott, Zeynep Akata, Honglak Lee, and Bernt Schiele. "Learning deep representations of fine-grained visual descriptions." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 49-58. 2016.

1. Sung, Flood, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M. Hospedales. "Learning to compare: Relation network for few-shot learning." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1199-1208. 2018.
  - Relation network 

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

# Action recognition
1. Piergiovanni, A. J., and Michael S. Ryoo. "Recognizing Actions in Videos from Unseen Viewpoints." arXiv preprint arXiv:2103.16516 (2021).
