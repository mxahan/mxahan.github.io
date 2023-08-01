---
tags: Papers
---

# Metrics and Losses

## 2022

## 2021

1. Liu, W., Lin, R., Liu, Z., Xiong, L., Schölkopf, B., & Weller, A. (2021, March). Learning with hyperspherical uniformity. In *International Conference On Artificial Intelligence and Statistics* (pp. 1180-1188). PMLR.

     - to achieve good generalization on unseen data, a suitable inductive bias is of great importance for neural networks. 
     -  hyperspherical uniformity is proposed as a novel family of **relational regularizations** that impact the interaction among neurons
       - several geometrically distinct ways to achieve hyperspherical uniformity and justified by theoretical insights and empirical evaluations.
       - neuron inhibit properties!!!
       - Uniform in the hypersphere! more generalization than orthogonal conditions
     - Current prevailing weight regularizations can be divided into two major categories: Individual regularization and relational regularization.
     - In order to promote hyperspherical uniformity with an explicit regularization, we formulate several distinct geometric learning objectives: minimum hyperspherical energy (MHE), maximum hyperspherical separation (MHS), maximum hyperspherical polarization (MHP), minimum hyperspherical covering (MHC), and maximum Gram determinant (MGD).
       - inspirated from statistical uniformity testing on the hypersphere and provide a novel and unified view on understanding those objectives
     - Motivation
       - hyperspherical uniformity leads to better optimization and generalization
       - hyperspherical uniformity can remove neuron redundancy and encourage the neurons to be diverse on the hypersphere
       - hyperspherical uniformity has a clear geometric interpretation and theoretical merits and close connection between orthogonality.
     - Related works: Relational regularization and hypersperical learning. **Very interesting**
       - orthogonality regularization, unitary constraint, decorrelation, spectral regularization, low-rank regularization, angular constraint
       - angular information in neural networks, in contrast to magnitude, preserves the key semantics and is very crucial to generalization
     - MHE: most related works. one seeks to find an equilibrium state with minimum potential energy that distributes N electrons on a unit sphere as evenly as possible (NCD!)
     - MHS: sphere packing problem: one packs a given number of circles on the sphere surface i.e. the minimum distance between circles can be maximized
       -  MHE has a global regularization effect (updates all vectors in each iteration), MHS focuses on the local separation (updates two vectors)
       - When the vectors are relatively diverse on the hypersphere, MHS tends to have stronger regularization effects than MHE. 
         - MHS gradient resulted from the two closest vectors has only one direction component and will not be cancelled out. In contrast, the MHE gradient comes from all the vectors has many direction components and may be largely cancelled out. Intuitively, MHS only activates the repulsive force from the closest vectors at a time while MHE simultaneously activates the repulsive force from all pairwise vectors
       - Optimizing MHS is straightforward and efficient: rank all the pairwise distances and obtain the vectors with the smallest distance (i.e., maximal similarity) and maximize the minimal distance by updating these two closest vectors via gradient ascent.
     - Interesting discussion section

2. Liu, D., Ning, J., Wu, J., & Yang, G. (2021). Extending Ordinary-Label Learning Losses to Complementary-Label Learning. IEEE Signal Processing Letters, 28, 852-856.

     - Weak supervision, learning from complementary label. (related works in intro is nice)

     - Again distribute complementary output to all others.

3. Kim, Y., Yun, J., Shon, H., & Kim, J. (2021). Joint negative and positive learning for noisy labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9442-9451).

     - *Learning from complementary labels.*
       
         - **TP**: JNPL (Improvement over NLNL)
           - trains CNN via two losses, NL+ and PL+ (improved NL and PL and also addressing their issues)
         
     - Related work: *Design noise-robust loss*, weighting samples, correction methods, selecting clean labels, **use of complementary labels**
         - Problem: Underfitting of NL (section 3)
           - focal loss flavor solution

     - Bad and unclear notation!!

     - <embed src="https://mxahan.github.io/PDF_files/joint_pos_neg_learn.pdf" width="100%" height="850px"/>

4. Leng, Z., Tan, M., Liu, C., Cubuk, E. D., Shi, J., Cheng, S., & Anguelov, D. (2021, September). PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions. In International Conference on Learning Representations.

   - New framework for loss function (taylor series expansion of log function)
       - PolyLoss allows the adjustment of polynomial bases depending on the tasks and datasets (subsumes cross-entropy loss and focal loss as special cases)
         - Experiment to support the requirement of adjustment.
         - Introduces an extra hyperparameters
   - Contributions: i) insight of common losses ii)  intuitive understanding of requirement to design different loss functions tailored to different imbalanced datasets!!
      - Proposes to manipulate the weight so the polynomial components
        - modifying the polynomial weights helps to go beyond the CE loss accuracy?
          - is it individual class dependent? For all classes or biased setup?
          - *the idea is to provide extra weights for the initial terms*


  - <embed src="https://mxahan.github.io/PDF_files/polyloss.pdf" width="100%" height="850px"/>

1. Kornblith, S., Chen, T., Lee, H., & Norouzi, M. (2021). Why do better loss functions lead to less transferable features?. Advances in Neural Information Processing Systems, 34.

     - how the choice of training objective affects the transferability of the hidden representations of CNNs trained on ImageNet
       - **Experimental paper**
       - Vanilla Cross-entropy loss as the final layers with different regularizers
     
         - Too much class separation it bad!
           - How did they measure the transfer accuracy?
             - transferability of the fixed features of our ImageNet-pretrained models by training linear or k-nearest neighbors (kNN) classifiers to classify 8 different natural image datasets (section 3.1)
     
     - Linear centered kernel alignment (CKA) provides a way to measure similarity of neural network representations that is invariant to rotation and isotropic scaling in representation space
         - Important findings
           - Better objectives improve accuracy, but do not transfer better
           - The choice of objective primarily affects hidden representations close to the output
           - Regularization and alternative losses increase class separation
           - Greater class separation is associated with less transferable features
     
     - limited to moderately sized datasets with moderately sized models, and our conclusions are limited to supervised classification settings
     
         

## 2020

1. Ren, J., Yu, C., Ma, X., Zhao, H., & Yi, S. (2020). Balanced meta-softmax for long-tailed visual recognition. *Advances in neural information processing systems*, *33*, 4175-4186.
     - Balanced cross entropy loss: Equation 3 is the proposed softmax operation. weighted by each sample number. 
1. Feng, L., Kaneko, T., Han, B., Niu, G., An, B., & Sugiyama, M. (2020, November). Learning with multiple complementary labels. In International Conference on Machine Learning (pp. 3072-3081). PMLR.
     - design two wrappers that decompose MCLs into many single CLs

     - Reverse the prediction and apply cross entropy loss.
1. Boudiaf, Malik, Jérôme Rony, Imtiaz Masud Ziko, Eric Granger, Marco Pedersoli, Pablo Piantanida, and Ismail Ben Ayed. "A unifying mutual information view of metric learning: cross-entropy vs. pairwise losses." In European Conference on Computer Vision, pp. 548-564. Springer, Cham, 2020.

     - TP: a theoretical analysis to link the cross-entropy to several well-known and recent pairwise losses from two perspective
       - i) explicit optimization insight and ii) discriminative and generative views of the mutual information between the labels and the learned features

     - Experimented with four different DML benchmarks (CUB200, Cars-196, Stanford Online Products (SOP) and In-Shop Clothes Retrieval (In-Shop)) [also used by MS loss]

     - TP proves that minimizing cross-entropy can be viewed as an approximate bound optimization of a more complex pairwise loss. [interesting section 4.3]

  - <embed src="https://mxahan.github.io/PDF_files/A_unifying_MI_view_metric_learning_CE.pdf" width="100%" height="850px"/>

1. Sainburg, Tim, Leland McInnes, and Timothy Q. Gentner. "Parametric UMAP embeddings for representation and semi-supervised learning." arXiv preprint arXiv:2009.12981 (2020).

     - Good starting note for the UMAP and tSNE. Parametric extension for the UMAP

     - [link](https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668) May contain Some bias towards UMAP

1. Sun, Yifan, Changmao Cheng, Yuhan Zhang, Chi Zhang, Liang Zheng, Zhongdao Wang, and Yichen Wei. "Circle loss: A unified perspective of pair similarity optimization." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6398-6407. 2020.

## 2019

1. Mettes, P., Van der Pol, E., & Snoek, C. (2019). Hyperspherical prototype networks. *Advances in neural information processing systems*, *32*.
     - unify classification and regression with prototypes on hyperspherical output spaces
     - propose to use hyperspheres as output spaces, with class prototypes defined a priori with large margin separation
       -  Training and inference is achieved through cosine similarities between examples and their fixed class prototypes.
     - position prototypes through data-independent optimization, with an extension to incorporate priors from class semantics (**how**)
       - Prototypes with privileged information (pairwise cross-entropy setting)
     -  generalize to regression, by optimizing outputs as an interpolation between two prototypes on the hypersphere [figure 1]
1. Ishida, T., Niu, G., Menon, A., & Sugiyama, M. (2019, May). Complementary-label learning for arbitrary losses and models. In International Conference on Machine Learning (pp. 2971-2980). PMLR.
     - derive a novel framework of complementary-label learning

     - Idea of gradient ascend.
1. Kim, Y., Yim, J., Yun, J., & Kim, J. (2019). Nlnl: Negative learning for noisy labels. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 101-110).

     - “input image belongs to this label” (Positive Learning; PL)

     - Negative learning: Randomly select other label than the not label !!!???
1. Zhao, S., Wang, Y., Yang, Z., & Cai, D. (2019). Region mutual information loss for semantic segmentation. Advances in Neural Information Processing Systems, 32.
     - Joint distribution for neighborhood pixels.
1. Wang, Xun, Xintong Han, Weilin Huang, Dengke Dong, and Matthew R. Scott. "Multi-similarity loss with general pair weighting for deep metric learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5022-5030. 2019.

     - (TP) Establish a General Pair Weighting (GPW) framework: casts the sampling problem of deep metric learning into a unified view of pair weighting through gradient analysis, (tool for understanding recent pair-based loss functions)

     - (TP) Various existing pair-based methods are compared and discussed comprehensively, with clear differences and key limitations identified;
         - (TP) Proposes new loss called multi-similarity loss (MS loss) under the GPW,
           - implemented in two iterative steps (i.e., **mining**: involves P and **weighting**: Involves S,N): consider three similarities for pair weighting, providing a more principled approach for collecting and weighting informative pairs
             - 3 similarities (self-similarity: S, Negative relative similarity: N, Positive relative similarity: P) [figure 2]
             - state-of-the-art performance on four image retrieval benchmarks (CUB200, Cars-196, Stanford Online Products (SOP) and In-Shop Clothes Retrieval (In-Shop))

  - <embed src="https://mxahan.github.io/PDF_files/msgpw.pdf" width="100%" height="850px"/>

1. Cheung, Brian, Alex Terekhov, Yubei Chen, Pulkit Agrawal, and Bruno Olshausen. "Superposition of many models into one." arXiv preprint arXiv:1902.05522 (2019).
     - Multiply weights to project them in orthogonal space and sum them.

## 2018

1. Golts, A., Freedman, D., & Elad, M. (2018). Deep-energy: unsupervised training of deep neural networks. *arXiv preprint arXiv:1805.12355*.

     - This work offers an unsupervised alternative that relies on the availability of task-specific energy functions, replacing the generic supervised loss
     - Experimentations: three different tasks – seeded segmentation, image matting and single image dehazing

1. Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the loss landscape of neural nets. *Advances in neural information processing systems*, *31*.

     - certain network architecture designs (e.g., skip connections) produce loss functions that train easier
     - well-chosen training parameters (batch size, learning rate, optimizer) produce minimizers that generalize better
       -  the reasons for these differences, and their effect on the underlying loss landscape, are not well understood
     - TP: explore the structure of neural loss functions, and the effect of loss landscapes on generalization, using a range of visualization methods
       -  introduce a simple **“filter normalization”** method that helps us visualize loss function curvature and make meaningful comparisons 
       - Section 4: filter normalization
     - Explore how network architecture affects the loss landscape, and how training parameters affect the shape of minimizers
     -  when networks become sufficiently deep, neural loss landscapes quickly transition from nearly convex to highly chaotic, causing a dramatic drop in generalization error
     - quantitatively measure non-convexity by calculating the smallest (most negative) eigenvalues of the Hessian around local minima, and visualizing the results as a heat map
     - Experimentation with convolution network. 
     - Some Insights:
       - The effect of Network Depth: has a dramatic effect on the loss surfaces of neural networks when skip connections are not used
       - Shortcut Connections to the Rescue: 
       - Wide Models vs Thin Models: wider models have loss landscapes less chaotic behavior
       - Implications for Network Initialization: loss landscapes  seem to be partitioned into a well-defined  low loss value and convex contours region, surrounded by a well-defined region of high and non-convex contours. This  may explain the importance of good initialization strategies, and also the easy training behavior of “good” architectures.
       - Landscape Geometry Affects Generalization

1. Liu, W., Lin, R., Liu, Z., Liu, L., Yu, Z., Dai, B., & Song, L. (2018). Learning towards minimum hyperspherical energy. *Advances in neural information processing systems*, *31*.

     - how to regularize the network to avoid undesired representation redundancy becomes an important issue
     - draw inspiration from physics-based Thomson problem: seeks to find a state that distributes N electrons on a unit sphere as evenly as possible with minimum potential energy
     - reformulate redundancy regularization problem to generic energy minimization, and propose a minimum hyperspherical energy (MHE) objective as generic regularization for neural network (also some variants of MHE: Angular MHE [A-MHE])
     - Related works: Diversity learning (enforce diversity in the prediction space), diversity regularization, ensemble learning, self-paced learning, metric learning, 
       -  formulating the diversity of neurons on the entire hypersphere, promoting diversity from a more global, top-down perspective
     - feature the significance of angular learning at both loss and convolution levels based on the observation that the angles in deep embeddings learned by CNNs tend to encode semantic difference
       -  key intuition is that angles preserve the most abundant and discriminative information for visual recognition
         - Keep the projection uniformly in the hypersphere. (global equidistance) 
       - MHE faces different situations when it is applied to hidden layers and output layers
     - characterize the diversity for a group of neurons by defining a generic hyperspherical potential energy using their pairwise relationship. Higher energy implies higher redundancy, while lower energy indicates that these neurons are more diverse and more uniformly spaced.

1. Kim, Wonsik, Bhavya Goyal, Kunal Chawla, Jungmin Lee, and Keunjoo Kwon. "Attention-based ensemble for deep metric learning." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 736-751. 2018.

     - Target:  learners should be diverse in their feature embeddings.
       - Divergence loss for diversity in attention map on M-head
       - attention-based ensemble, which uses multiple attention masks, to attend different parts of the object.
       - ABE-M (M way attention based Ensemble)

     - Section 3 describe it all [setup, and divergence loss to enforce different attention (section 3.4)]

     - Experiments: 4 dataset (CUB200, ... ), Arch: GoogLeNet (pretrained on ILSVRC dataset), M = 8.

1. He, Xinwei, Yang Zhou, Zhichao Zhou, Song Bai, and Xiang Bai. "Triplet-center loss for multi-view 3d object retrieval." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1945-1954. 2018.

     - variants of deep metric learning losses for 3D object retrieval (proposes a new loss: Combination of center and triplet losses)

     - learns a center for each class and the distances between samples and centers from the same class are closer than those from different classes.

     - related works: i. View based ii. Model based

     - Experiment: Dataset - ModelNet40 and ShapeNet Core55,

     - Joint optimization with softmax losses

     - Benefits: require less samples

     - <embed src="https://mxahan.github.io/PDF_files/triplet_center_loss.pdf" width="100%" height="850px"/>

## 2017

1. Ishida, T., Niu, G., Hu, W., & Sugiyama, M. (2017). Learning from complementary labels. Advances in neural information processing systems, 30.

   - PDF formulation, risk minimization

   - *complementary loss*: incurs a large loss if a predicted complementary label is not correct

   - terminology: *Ordinary Label* and *complementary label*

   - A bit strong assumption that the complementary examples are from all other classes!
       - **I think** this is where the gradient flows to everybody
         - Causes underfitting

   - <embed src="https://mxahan.github.io/PDF_files/learning_from_comple_label.pdf" width="100%" height="850px"/>

     

1. Lin, Tsung-Yi, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. "Focal loss for dense object detection." In Proceedings of the IEEE international conference on computer vision, pp. 2980-2988. 2017.

1. Wang, Feng, Xiang Xiang, Jian Cheng, and Alan Loddon Yuille. "Normface: L2 hypersphere embedding for face verification." In Proceedings of the 25th ACM international conference on Multimedia, pp. 1041-1049. 2017.

   - Training using normalization features.
     - modification of softmax and optimize cosine losses
     - Metric learning

   - Research gap of necessity of normalization

   - Four contributions?
     - why cosine doesn't converge? buy normalized dot succeed.
       - different loss option explore? why!!

## 2016 and Earlier

1. Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G. (2005, August). Learning to rank using gradient descent. In *Proceedings of the 22nd international conference on Machine learning* (pp. 89-96).
     - propose a probabilistic cost function: RankNet: an implementation of these ideas using a NN to model the underlying ranking function
1. Vapnik, V., & Izmailov, R. (2015). Learning using privileged information: similarity control and knowledge transfer. *J. Mach. Learn. Res.*, *16*(1), 2023-2049.
     - TP provides two mechanisms to significantly accelerate the speed of Student’s learning using privileged information: (1) correction of Student’s concepts of similarity between examples, and (2) direct Teacher-Student knowledge transfer.
       - Early form of student-teacher representation learning!!
1. Wen, Yandong, Kaipeng Zhang, Zhifeng Li, and Yu Qiao. "A discriminative feature learning approach for deep face recognition." In European conference on computer vision, pp. 499-515. Springer, Cham, 2016.
     - TP: To enhance discriminative power proposes a new loss: Center loss (distance loss from center of class clusters)
       - Inter-class dispension and intra-class compactness

     - Experiment: Joint estimation of Cross-entropy and center-loss

     - Dataset: LFW, YTF, tasks: Face recognition and verification.

     - Interesting discussion section (kinda ablation study)
1. Hoffer, Elad, and Nir Ailon. "Deep metric learning using triplet network." In International Workshop on Similarity-Based Pattern Recognition, pp. 84-92. Springer, Cham, 2015.

     - Triplet networks

     - Experimented on the MNIST dataset.
1. Yi, Dong, Zhen Lei, Shengcai Liao, and Stan Z. Li. "Deep metric learning for person re-identification." In 2014 22nd international conference on pattern recognition, pp. 34-39. IEEE, 2014.

