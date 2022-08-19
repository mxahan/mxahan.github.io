
# Introduction

This blog contains state of the art application and research on different applications. The applications will be stated in the subheadings. Although one paper can fall into multiple categories we will be mostly going with one tag per papers. However, the tag is subjected to change time-to-times.

### Table of contents

- [Multiview application](#Multi-View-Application)
- [Domain Adaptation](#Domain-Adaptation)


# Multi View Application

# Domain Adaptation

## 2022

1. Arnab, A., Xiong, X., Gritsenko, A., Romijnders, R., Djolonga, J., Dehghani, M., ... & Schmid, C. (2022). Beyond Transfer Learning: Co-finetuning for Action Localisation. arXiv preprint arXiv:2207.03807.
  - question the traditional two-step TL approach, and propose co-finetuning
  - co-finetuning significantly improves the performance on rare classes
  - has a regularising effect, and enables the network to learn feature representations that transfer between different datasets.
  - *TP*: Modify the training strategy.
    - 2 tasks consequently, [figure 2] on multiple datasets together.
      - classification and person detection bounding box.
      - Avoid catastrophic forgetting & Tx overfitting & Small dataset.
      - Helps improving mid and tail classes.
  - Experiment: ViViT, AVA, Kinetics, Moments in time, Something-something v2

1. Xie, B., Yuan, L., Li, S., Liu, C. H., & Cheng, X. (2022). Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 8068-8078).
  - TP: region-based active learning approach for semantic segmentation under a domain shift (RIPU)
    - aiming to automatically query a *small partition* of image regions to be labeled while maximizing segmentation performance
      - is to select the most diverse and uncertain regions
    - a *new* acquisition strategy characterizing the spatial adjacency of image regions along with the prediction confidence.
    - enforce local prediction consistency between a pixel and its nearest neighbors on a source image
    - develop a negative learning loss to make the features more discriminative
  - introduce two labeling mechanisms
    - what is not? reducing what is not
    - “Region-based Annotating (RA)”: every pixel in the selected regions—high annotation regime
    - “Pixel-based Annotating (PA)”: focus more on the labeling efficiency by selecting the center pixel within the region
  - Joint optimization: Pretty much like semi-supervised learning!

## 2020

1. Yuan, Y., Chen, X., & Wang, J. (2020, August). Object-contextual representations for semantic segmentation. In European conference on computer vision (pp. 173-190). Springer, Cham.
  - study the context aggregation problem in semantic segmentation.
  - TP: i) learn object regions under the supervision of the GT segmentation. ii) compute the object region representation by aggregating the representations of the pixels lying in the object region iii) compute the relation between each pixel and each object region, and augment the representation of each pixel with the *object-contextual representation* (weighted aggregation of all the object region representations)

1. Wilson, Garrett, and Diane J. Cook. "A survey of unsupervised deep domain adaptation." ACM Transactions on Intelligent Systems and Technology (TIST) 11, no. 5 (2020): 1-46.
  - unsupervised domain adaptation can handle situations where a network is trained on labeled data (source domain) and unlabeled target data (related but different domain) with the goal of performing well at target domain
  - 3 TL: i) inductive (target and source tasks are different) ii) transductive (tasks remain the same while the domains are different, domain adaptation) iii) Unsupervised (inductive with no labels)
  - Domain invariant feature learning
    - aligning distribution by divergence (i) Maximum Mean Discrepency ii) correlation alignment iii) Contrastive domain D. iv) Wasserstein
    - reconstruction
    - Adversarial
  - Domain mapping: Image-image mapping
  - Normalization Statistics
  - Ensemble methods
  - Target Discriminative methods

## 2016 and prior

1. Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." In International conference on machine learning, pp. 1180-1189. 2015.
  - Simple and great Idea [eq: 1,2,3].
    - Input to features (*f*), features to task (*y, loss<sub>y</sub>*), features to domain classifier (*d, loss<sub>d</sub>*).
    - *f* tries to minimize the *loss<sub>y</sub>* and maximize the *loss<sub>d</sub>*, and *d, y* tries to minimize their corresponding losses.
  - Final task need to be related (y) but the source may be different (f tries to find common ground).
  - Gradient Reversal layer to implement via SGD.
  - Reduces h delta h distance [eq 13]

1. Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. Machine learning, 79(1), 151-175.
  - Investigate
    - under what conditions can a classifier trained from source data be expected to perform well on target data?
      - bounding a classifier’s target error in terms of its source error and the divergence between the two domains
    - given a small amount of L target data, how should we combine it during training with the large amount of L source data to achieve best performance?
      - bounding the target error of a model which minimizes a convex combination of the empirical source and target errors.

# Model Compression

1. Blalock, Davis, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. "What is the state of neural network pruning?." arXiv preprint arXiv:2003.03033 (2020).

1. Khetan, Ashish, and Zohar Karnin. "PruneNet: Channel Pruning via Global Importance." arXiv preprint arXiv:2005.11282 (2020).
  - Importance score: Variance of input layer after filtering
  - New regularization scheme.

1. 1. Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).
  - Knowledge distillation
  - Distillation and the effect of temperature.
    - section 2 and 2.1 are important
  - Training small network to mimic the large network.
  - Train small network to learn the features and logits of the large network.
  - Softmax, temperature and the MSE with the prediction
  - Experimented with MNIST, speech and Specialist models.


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

# Action recognition, # progression, # 3D # Pose

## 2021

1. Liu, X., Pintea, S. L., Nejadasl, F. K., Booij, O., & van Gemert, J. C. (2021). No frame left behind: Full video action recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14892-14901).
  - TP: propose to do away with sub-sampling heuristics and argue for leveraging all video frames: Full video action recognition

1. Piergiovanni, A. J., and Michael S. Ryoo. "Recognizing Actions in Videos from Unseen Viewpoints." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4124-4132. 2021.
  - How to learn the camera matrix
  - Camera matrix background (pinhole camera model) [ref 1](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf), [ref 2](https://hedivision.github.io/Pinhole.html), [short course](https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/CV/index.html)
  - TP: i) approaches based on 3D representations ii) introduce a new geometric convolutional layer (neural projection layer: an architectural contribution, key contribution) to learn viewpoint invariant representations (target)  iii) new dataset (MLB-Youtube dataset)
    - Neural Projection Layer:
  - Invariant represtation: by learning global 3D pose
  - Hypothesis to obtain target: i) implicitly learning from large data ii) use 3D posture information (!! how)
  - TP: Testimating 3D pose directly from the videos (latent 3D representation and it's multiview 2d projection), then explores using different representations of it for AR.
  - TP: 3D representation by using 3D geometric transformation and projection.
  - Network: PoseNet (3D posture), and CalibNet (external camera matrix, however deems error-prone): Loss Function: sum of three losses (cross entropy, metric loss, and camera distribution loss !!!), Dataset: Human3.6M, MLB (baseball), TSH (toyota smart home)

1. Piergiovanni, A. J., and Michael S. Ryoo. "Recognizing Actions in Videos from Unseen Viewpoints." arXiv preprint arXiv:2103.16516 (2021).

## 2020

1. van Amsterdam, Beatrice, Matthew J. Clarkson, and Danail Stoyanov. "Multi-task recurrent neural network for surgical gesture recognition and progress prediction." In 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 1380-1386. IEEE, 2020.

1. Zhu, Yi, Xinyu Li, Chunhui Liu, Mohammadreza Zolfaghari, Yuanjun Xiong, Chongruo Wu, Zhi Zhang, Joseph Tighe, R. Manmatha, and Mu Li. "A Comprehensive Study of Deep Video Action Recognition." arXiv preprint arXiv:2012.06567 (2020).
  - Popular dataset descriptions
  -

1. Jenni, Simon, and Paolo Favaro. "Self-Supervised Multi-View Synchronization Learning for 3D Pose Estimation." In Proceedings of the Asian Conference on Computer Vision. 2020.
  - Experiment Data: Human3.6M dataset, ResNet architectures
  - Self-supervised task: Transformation and synchronous prediction

1. heng, Ce, Wenhan Wu, Taojiannan Yang, Sijie Zhu, Chen Chen, Ruixu Liu, Ju Shen, Nasser Kehtarnavaz, and Mubarak Shah. "Deep learning-based human pose estimation: A survey." arXiv preprint arXiv:2012.13392 (2020).

## 2019

1. Wang, Keze, Liang Lin, Chenhan Jiang, Chen Qian, and Pengxu Wei. "3D human pose machines with self-supervised learning." IEEE transactions on pattern analysis and machine intelligence 42, no. 5 (2019): 1069-1082.
  - Experiment data: HumanEva-I and Human3.6M

1. Kocabas, Muhammed, Salih Karagoz, and Emre Akbas. "Self-supervised learning of 3d human pose using multi-view geometry." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1077-1086. 2019.


## 2018

1. Sun, Xiao, Bin Xiao, Fangyin Wei, Shuang Liang, and Yichen Wei. "Integral human pose regression." In Proceedings of the European conference on computer vision (ECCV), pp. 529-545. 2018.
  - unifies heatmap representation and joint regression using integral operation
    - addresses the differentiable and quantization error of heatmap
    - Compatible with any heatmap based approach (!!), 3D pose estimation
  - Experiments: Architecture: ResNet backbone and simple head network, L2 loss, Dataset: Human3.6M, MPII, COCO keypoint detection

1. Choutas, Vasileios, Philippe Weinzaepfel, Jérôme Revaud, and Cordelia Schmid. "Potion: Pose motion representation for action recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7024-7033. 2018.
  - Process appearance and motion (movement of few relevant keypoints, human joints, over an entire video clip) jointly (Figure 1, aggregation of features)
    - TP:  novel representation (just aggregation and coloring!!!) that gracefully encodes the movement of some semantic keypoints
  - Interesting related works: pose detection and beyond
  - Two major steps i) PoTion representation [extracting joint heatmap, time dependent heatmap colorization, aggregation of colorized heatmap (three ways, unit, intensity, and normalized)] ii) CNN on PoTion representation
  - Experiment  i) Architecture: I3D, both 1 and 2 streams, Temporal segment network (TSN) ii) Dataset: JHMDB, HMDB and UCF101

1. Ionescu, Catalin, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. "Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments." IEEE transactions on pattern analysis and machine intelligence 36, no. 7 (2013): 1325-1339.



# Disentangled Representation learning

## 2022
1. Hou, Wenjie, Zheyun Qin, Xiaoming Xi, Xiankai Lu, and Yilong Yin. "Learning Disentangled Representation for Self-supervised Video Object Segmentation." Neurocomputing (2022).

## 2021

1. Hu, Jie, Liujuan Cao, Tong Tong, Qixiang Ye, Shengchuan Zhang, Ke Li, Feiyue Huang, Ling Shao, and Rongrong Ji. "Architecture disentanglement for deep neural networks." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 672-681. 2021.

1. Wang, Tan, Zhongqi Yue, Jianqiang Huang, Qianru Sun, and Hanwang Zhang. "Self-Supervised Learning Disentangled Group Representation as Feature." arXiv preprint arXiv:2110.15255 (2021).
  - Higgins definition of disentanglement representation
  - Group theoretic approach [Learn distinguished matrix for each tasks]

1. Maziarka, Łukasz, Aleksandra Nowak, Maciej Wołczyk, and Andrzej Bedychaj. "On the relationship between disentanglement and multi-task learning." arXiv preprint arXiv:2110.03498 (2021).
  - TP conclusion: a hard-parameter sharing scenario multi-task learning indeed seems to encourage disentanglement,
  - however, it is inconclusive whether disentangled representations have a clear positive impact on the models performance.
  - A synthetic dataset!


## 2020
1. Zhu, Yizhe, Martin Renqiang Min, Asim Kadav, and Hans Peter Graf. "S3VAE: Self-supervised sequential VAE for representation disentanglement and data generation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6538-6547. 2020.

1. Dittadi, A., Träuble, F., Locatello, F., Wüthrich, M., Agrawal, V., Winther, O., ... & Schölkopf, B. (2020). On the transfer of disentangled representations in realistic settings. arXiv preprint arXiv:2010.14407.
  - Provide new dataset (1M simulated and real-world 1800 images)
  - TP: propose new architectures to scale disentangled representation learning in realistic high-resolution settings and conduct a large-scale empirical study
    - disentanglement is a good predictor for out-of-distribution (OOD) task performance.

## 2019

1. Do, Kien, and Truyen Tran. "Theory and evaluation metrics for learning disentangled representations." arXiv preprint arXiv:1908.09961 (2019).

1. Arjovsky, Martin, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. "Invariant risk minimization." arXiv preprint arXiv:1907.02893 (2019).

1. Locatello, Francesco, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. "Challenging common assumptions in the unsupervised learning of disentangled representations." In international conference on machine learning, pp. 4114-4124. PMLR, 2019.
  - Requires inductive bias (implicit supervision)
  - (i) The role of inductive biases and implicit and explicit supervision should be made explicit: unsupervised model selection persists as a key question. (ii) practical benefits of enforcing a specific notion of disentanglement of the learned representations should be demonstrated.

## 2018

1. Hsieh, Jun-Ting, Bingbin Liu, De-An Huang, Li F. Fei-Fei, and Juan Carlos Niebles. "Learning to decompose and disentangle representations for video prediction." Advances in neural information processing systems 31 (2018).

1. Zamir, Amir R., Alexander Sax, William Shen, Leonidas J. Guibas, Jitendra Malik, and Silvio Savarese. "Taskonomy: Disentangling task transfer learning." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3712-3722. 2018.
  - Encoder decoder (VAE) set up for different self-supervised pretext tasks and experimentation  on the transfer learning.
  - Provide a good benchmark to transfer one task learning to other tasks. [relationship among tasks]
  - Provided their own dataset

1. Higgins, Irina, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, and Alexander Lerchner. "Towards a definition of disentangled representations." arXiv preprint arXiv:1812.02230 (2018).
  - Formal *Definition(s)* of the disentangled representation learning
  - A vector representation is called a disentangled representation with respect to a particular decomposition of a symmetry group into subgroups, if it decomposes into **independent subspaces**, where each subspace is affected by the action of a single subgroup, and the actions of all other subgroups leave the subspace unaffected.

1. Wu, Zhenyu, Karthik Suresh, Priya Narayanan, Hongyu Xu, Heesung Kwon, and Zhangyang Wang. "Delving into robust object detection from unmanned aerial vehicles: A deep nuisance disentanglement approach." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1201-1210. 2019.
  - Utilization of meta-data
  - Multitask setup

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
  - <embed src="https://mxahan.github.io/PDF_files/Perceptual_reward.pdf" width="100%" height="850px"/>

1. Ziebart, Brian D., Andrew L. Maas, J. Andrew Bagnell, and Anind K. Dey. "Maximum entropy inverse reinforcement learning." In Aaai, vol. 8, pp. 1433-1438. 2008.
  - Framing problem as Imitation learning as solutions to MDP
  - Model navigation and driving behavior -> under noisy data
  - Issues in imitation learning during the imperfect demonstration
    - alternative solution: IRL : Matching feature count (algo 1)
  - <embed src="https://mxahan.github.io/PDF_files/Maximum_entropy_IRL.pdf" width="100%" height="850px"/>

# Data augmentation

## 2019 and Earlier

1. Liu, Zhuang, Tinghui Zhou, Hung-Ju Wang, Zhiqiang Shen, Bingyi Kang, Evan Shelhamer, and Trevor Darrell. "Transferable recognition-aware image processing." arXiv preprint arXiv:1910.09185 (2019).
  - optimizing the recognition loss directly on the image processing network or through an intermediate transforming model (transferable)
    -a neural network for image processing: maps an input image to an output image with some desired properties.
  - propose add a recognition loss optimized jointly with the image processing loss.
  - Figure 2: sums the work. (process loss + recognition task)

1. Cubuk, Ekin D., Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V. Le. "Autoaugment: Learning augmentation policies from data." arXiv preprint arXiv:1805.09501 (2018).

1. Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).
  - supervised setup, linear interpolation between input and corresponding labels
  - a regularization method: design choices in ablation studies
  - A connection with ERM (interesting read)
  - MT (my thoughts): mixing 1 and 2 still gives 1/2 of both of it? whats the significance of the weights between mixing. I get it still can it be improved?

## 2020

1. Lee, Kibok, Yian Zhu, Kihyuk Sohn, Chun-Liang Li, Jinwoo Shin, and Honglak Lee. "i-mix: A domain-agnostic strategy for contrastive representation learning." arXiv preprint arXiv:2010.08887 (2020).
  - i-mix: regularization strategy for improving contrastive representation learning (data augmentation approaches)
    - assigning a unique virtual class to each data in a batch and mixing them in both the input and virtual label spaces
      - Weighted Mixing two images and weighted sum of their virtual labels (self-supervision.) with N-Pair contrastive loss
    - Application: image, speech, and tabular data
    - Compatible with different CL frameworks (simclr, moco, boyl) [requires modification (nice!): Section 3]

1. Cubuk, Ekin D., Barret Zoph, Jonathon Shlens, and Quoc V. Le. "Randaugment: Practical automated data augmentation with a reduced search space." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 702-703. 2020.
  - TP tackles: (i) Reduce search spaces for automatic data augmentations! {by Simplifying the search spaces: avails grid search} (ii) adjust regularization strengths
  - Experimentation with the role of data augmentations.
  - Related works: Learning automatic data augmentations! AutoAugment RL basede methods! Population based augmentation,
  - Interesting empirical results: data augmentation depends on model sizes and dataset training set sizes. Experiments with cifar, SVHN and ImageNET, object detection.
  - Methods:

## 2021

1. Talebi, Hossein, and Peyman Milanfar. "Learning to Resize Images for Computer Vision Tasks." arXiv preprint arXiv:2103.09950 (2021).
  - Does exactly what it's saying!! learn to resize using CNN layer for better downstream Tasks (proposes a resizer block )
    - Why in the first place!  
    - What will be the loss function for resize? (architecture: Figure 3)
    - Wait: are the extracting another feature! (calling it as resizer!)
  - ImageNet experiment, uses different model
  -  designing novel regularizers without class-dependent bias remains an open research question

## 2022

1. Alaa, A., Van Breugel, B., Saveliev, E. S., & van der Schaar, M. (2022, June). How faithful is your synthetic data? sample-level metrics for evaluating and auditing generative models. In International Conference on Machine Learning (pp. 290-306). PMLR.
  - Devising domain- and model-agnostic evaluation for generative Models  
    - 3 metrics, alpha-precision, beta-recall, authenticity, to characterize the fidelity, diversity and generalization

1. Balestriero, Randall, Leon Bottou, and Yann LeCun. "The Effects of Regularization and Data Augmentation are Class Dependent." arXiv preprint arXiv:2204.03632 (2022).
  - Aims to understand the impact of regularization
  - Discuss the impacts of two regularizers (implicit [model independent] regu: DA and explicit regu: weight decay [L2 norm on weights]) [very specific research problem]
    - Key finding:  DA or weight decay produce a model with a reduced complexity that is unfair across classes. The optimal amount of DA or weight decay found from cross-validation leads to disastrous model performances on some classes.
      - demonstrate that when employing regularization such as DA or weight decay, a significant bias is introduced into the trained model
      - Regu introduces bias (mostly good): HOWEVER, the bias introduced by regularization treats classes differently, including on transfer learning
    - Search for ever increasing generalization performance (averaged over all classes and samples) has left us with models and regularizers that silently sacrifice performances on some classes.
      - the regularized model exhibits strong per-class favoritism i.e. while the average test accuracy over all classes improves when employing regularization, it is at the cost of the model becoming arbitrarily inaccurate on some specific classes
    - DA introduce bias but not necessarily balanced throughout all classes
      - DA that is not label preserving introduce bias.
      - Unchecked Uniform DA the average test accuracy for overall classes, and decrease in some per-class test accuracies
    - Weight decay also create class dependent bias models
      - even for uninformative regularizers such as weight decay a per-class bias is introduced, reducing performances for some of the classes
    - The Class-Dependent Bias Transfers to Other Downstream Tasks

1. Balestriero, Randall, Ishan Misra, and Yann LeCun. "A Data-Augmentation Is Worth A Thousand Samples: Exact Quantification From Analytical Augmented Sample Moments." arXiv preprint arXiv:2202.08325 (2022).
  - Theoretically analyze the effect of DA by studying:
    - how many augmented samples are needed to correctly estimate the information encoded by that DA?
    - How does the augmentation policy impact the final parameters of a model?
  - TP: Close form derivation of the expectation and variance of an image, loss, and model’s output given a DA (quantify the benefits and limitations of DA)
  - training loss to be stable under DA sampling, the model’s saliency map (gradient of the loss with respect to the model’s input) must align with the smallest eigenvector of the sample variance under the considered DA
    - explanation on why models tend to shift their focus from edges to textures!
  - TP: proposes **data space transform** instead of *coordinating space tx*
  - <embed src="https://mxahan.github.io/PDF_files/DA_worth_thousand.pdf" width="100%" height="850px"/>
