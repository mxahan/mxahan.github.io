# Open-Set Recognition problem (OSR)

Here, we will review papers regarding novel class detection (NCD), Out of distribution detection (OOD) mostly in computer vision.

- L for labeled
- cl for contrastive learning
- u for unlabeled or unknown
- gt - ground truth  

# Out-of-Distribution (OOD)

## 2022
1. Joseph, K. J., Paul, S., Aggarwal, G., Biswas, S., Rai, P., Han, K., & Balasubramanian, V. N. (2022). Novel Class Discovery without Forgetting. arXiv preprint arXiv:2207.10659.
  - identify and formulate a new, pragmatic problem setting of NCDwF: Novel Class Discovery without Forgetting
  - propose 1) a method to generate pseudo-latent representations for previously available L to alleviate forgetting 2) a MI based regularizer to enhance unsupervised NCD, and 3) a simple Known Class Identifier for generalized inference form L and U.
  - Related works: Incremental learning: to alleviate the catastrophic forgetting of model when learning across a sequence of tasks (*requires all labels*) by some regularization, memory based approaches, dynamically expanding and parameter isolation.
  - *TP*: labeled data can't be accessed during NCD time

1. Yang, M., Zhu, Y., Yu, J., Wu, A., & Deng, C. (2022). Divide and Conquer: Compositional Experts for Generalized Novel Class Discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14268-14277).
  - focus on this generalized setting of NCD (GNCD) by challenging two-step setup for L and U.
  - propose to divide and conquer it with two groups of Compositional Experts (ComEx).
  - propose to strengthen ComEx with *global-to-local and local-to-local regularization*.
  - Unsup clustering enforce neighborhood consistency and average entropy maximization: achieve clustering and avoid collapse.
  - two group of experts (lol: final layers MTL)! batch and class-wise ![image](https://amingwu.github.io/assets/images/novelty.png)

1. Zheng, J., Li, W., Hong, J., Petersson, L., & Barnes, N. (2022). Towards Open-Set Object Detection and Discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 3961-3970).
  - present a new task, namely Open-Set Object Detection and Discovery (OSODD)
    - propose a two-stage method that first uses an open-set object detector to predict both known and unknown objects
  - propose a category discovery method using *domain-agnostic augmentation*, CL and *semi-supervised clustering*.
  - approach: Open-set object detector with memory module, object category discovery with representation learning,

1. Joseph, K. J., Paul, S., Aggarwal, G., Biswas, S., Rai, P., Han, K., & Balasubramanian, V. N. (2022). Spacing Loss for Discovering Novel Categories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 3761-3766).
  - *Spacing loss* that enforces separability in the latent space using cues from multi-dimensional scaling
    - an either operate as a standalone method or can be plugged into existing methods to enhance them
  - characterize existing NCD approaches into single-stage and two-stage methods based on if they require access to L and U data together while discovering NC
    - Single-stage NCD models can access L and U together
  - common NCD methodologies: learn a feature extractor using the L and use clustering, psuedo-labelling or CL
  - Experiment with CIFAR dataset
  - Two characteristics: 1) the ability to transport similar samples to locations equidistant from other dissimilar samples in the latent manifold, 2) the datapoints to refresh their associativity to a group as the learning progresses
  - Spacing loss summary: i) finding equidist point

1. Zhao, Y., Zhong, Z., Sebe, N., & Lee, G. H. (2022). Novel Class Discovery in Semantic Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4340-4349).
  - Three stage learning.
    - LABELED data, Saliency map (another model dependent), ranking based MA training.

1. Vaze, S., Han, K., Vedaldi, A., & Zisserman, A. (2022). Generalized Category Discovery. arXiv preprint arXiv:2201.02609.
  - leverages the CL trained vision transformers to assign labels directly through clustering.
  - Existing recognition methods have several restrictive assumptions,  
    - the unlabelled instances only coming from known — or unknown
    - classes and the number of unknown classes being known a-priori.
    - TP: **Challenges these** and propose GCD. (improved NCD)
  - Approaches: Baseline, ViT, CL, Semi-supervised setup.
    - Dataset: CIFAR10, CIFAR100 and ImageNet-100
  - OSR: aims detect test-time images which do not belong to one of the labeled classes, does not require any further classification
  - NCD: aim to discover new classes in the unlabelled set, prone to overfit.
  - key insight is to leverage the strong ‘NN’ classification property of vision transformers along with CL
    - TP: use of contrastive training and a semi-supervised k-means clustering
    - TP: *estimating the number of categories* in unlabelled data
  - Related works: Semi-supervised, OSR,
  - how that existing NCD methods are prone to overfit the labelled classes in this generalized setting
  - CL and a semi-supervised k-means clustering to recognize images without a parametric classifier
  - Approach overview:
    - CL pretraining (ViT, DiNo pretrained) [kinda SCL setup]
    - Label assignment with semi-supervised k-means (use a non-parametric method)
      - Appendices [figure 4]
      - utilization of [k++](https://towardsdatascience.com/understanding-k-means-k-means-and-k-medoids-clustering-algorithms-ad9c9fbf47ca#:~:text=K%2DMeans%2B%2B%20is%20a%20smart%20centroid%20initialization%20technique%20and,dataset%20from%20the%20selected%20centroid.) for smart initialization and clustering methods. [elbow for finding K?]

1. Yang, H. M., Zhang, X. Y., Yin, F., Yang, Q., & Liu, C. L. (2020). Convolutional prototype network for open set recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence.
 - CNN for representation learning but replaces the closed-world assumed softmax with an open-world oriented prototype model. [CPN]
 - design several discriminative losses [OVA loss]
 - propose a generative loss (maximizing the log-likelihood) to act as a latent regularization. [is that as vague as their earlier paper??]
  - Nice but very easy: It bounds the class distance by some distance (eventually increases the **log(distance)** increases log likelihood)
  - Discusses two rejection rules (distance based and probability based)
    - Pretty straight forward

1. Zhou, Y., Liu, P., & Qiu, X. (2022, May). KNN-Contrastive Learning for Out-of-Domain Intent Classification. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 5129-5141).
  - Modified contrastive loss KNN-CL (T)
    - NLP works
    - KNN clustering and contrastive learning.
      - apply loss in different labels.

1. Dietterich, Thomas G., and Alexander Guyer. "The Familiarity Hypothesis: Explaining the Behavior of Deep Open Set Methods." arXiv preprint arXiv:2203.02486 (2022).
  - Research GAP: Detecting such “novel category” objects is formulated as an anomaly detection problem
  - TP demonstrate: the Familiarity Hypothesis that these methods succeed because they are detecting the absence of familiar learned features rather than the presence of novelty
    - reviews evidence from the literature (how to show them!!) and presents additional evidence and  suggest some promising research directions.
    - Looked into the penultimate layer activation norm (low for unseen classes): as Network was not activated enough [no feature found!!]
  - Claim: computer vision systems should master two functions: (a) detecting when an object belongs to a new category [TP] (b) learning to recognize that new category
  - The Familiarity Hypothesis (FH): **The standard model succeeds by detecting the absence of familiar features in an image rather than by detecting the presence of novel features in the image.**
  - interesting ways to find feature activation [validity!!]
  - Discussion section is a **gem**!!


## 2021

1. Choudhury, S., Laina, I., Rupprecht, C., & Vedaldi, A. (2021). Unsupervised part discovery from contrastive reconstruction. Advances in Neural Information Processing Systems, 34, 28104-28118.
  - Res.Gap.: representation learning at part level has received significantly less attention (most work focus on object and scene level)
  - Propose an unsup approach to object part discovery and segmentation
  - three contributions
    - construct a proxy task through a set of objectives (encourages the model to learn a meaningful decomposition of the image into its parts) [*CL*]
    - prior work argues for reconstructing or clustering pre-computed features as a proxy to parts
      - this paper shows that: this alone is unlikely to find meaningful parts;
      - because of their low resolution and the tendency of classification networks to spatially smear out information
      - image reconstruction at the level of pixels can alleviate this problem, acting as a complementary cue
      - the standard evaluation based on keypoint regression does not correlate well with segmentation quality
      - introduce different metrics, NMI and ARI (better characterize the decomposition of objects into parts)
    - given a collection of images of a certain object category (e.g., birds) and corresponding object masks, we want to learn to decompose an object into a collection of repeatable and informative parts.
      - no universally accepted formal definition for what constitutes a “part”, the nature of objects and object parts is accepted as different
    -  (a) consistency to transformation (equivariance), (b) visual consistency (or self-similarity), and (c) distinctiveness among different parts.

1. Jia, X., Han, K., Zhu, Y., & Green, B. (2021). Joint representation learning and novel category discovery on single-and multi-modal data. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 610-619).
  - a generic, end-to-end framework to jointly learn a reliable representation and assign clusters to unlabelled data.
  - Propose to overcluster than the original unknown classe (U Cardinality is known) [**Well! Gives something to work with!!!!**]
  - Joint optimization of many Losses
    - CL (both instance and cluster [for known label])
    - BCE (siamese network setup) [pseudo label]
    - Consistent MSE loss (different view of same data)
    - CE loss

1. Fini, E., Sangineto, E., Lathuilière, S., Zhong, Z., Nabi, M., & Ricci, E. (2021). A unified objective for novel class discovery. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 9284-9292).
  - depart from this traditional multi-objective and introduce a UNified Objective function [UNO] for NCD
    - favoring synergy between supervised and unsupervised learning
    - multi-view self-labeling strategy generate pseudo-labels homogeneously with GT
    - overview figure 2 [multihead network (L and U data)]
      - replace multi-objective using the multitask setting.
      - look at the **gradient flow strategy**
    - similar idea of **swav**
    - *dimension mismatch* in eq 4 and 5  
      - can be fixed by altering Y and L in the eq 4
  - <embed src="https://mxahan.github.io/PDF_files/UNO.pdf" width="100%" height="850px"/>

1. Zhong, Z., Fini, E., Roy, S., Luo, Z., Ricci, E., & Sebe, N. (2021). Neighborhood Contrastive Learning for Novel Class Discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10867-10875).
  - New framework for NCD [NCL]
    - i) a encoder trained on the L to generates representations (a generic query sample and its neighbors are likely to share the same class)
      - retrieve and aggregate pseudo-positive pairs with CL
    - ii) propose to generate hard negatives by mixing L and U samples in the *feature space*.
  - first idea: the local neighborhood of a query in the embedding space will contain samples most likely belong to the same semantic category (pseudo-+ve)
    - *numerous positives* obtain a much stronger learning signal compared to the traditional CL with only two views
  - second idea: address the better selection of -ve to further improve CL
  - related works: negative mining.
    - Their approach [figure 3]
  - Well: add bunch of losses together for joint optimization.
  - kind of avoid false-ve in CL
  - Assumption of L intersection U = {}
  - <embed src="https://mxahan.github.io/PDF_files/Neighborhood_CL.pdf" width="100%" height="850px"/>

1. Zhong, Z., Zhu, L., Luo, Z., Li, S., Yang, Y., & Sebe, N. (2021). Openmix: Reviving known knowledge for discovering novel visual categories in an open world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9462-9470).
  - mix the unlabeled examples from an open set and the labeled examples from known classes
    - non-overlapping labels and pseudo-labels are simultaneously mixed into a joint label distribution
    - kinda *data augmentation* approach like MixUp
      - generates training samples by incorporating both labeled and unlabeled samples
    - Assumption: : 1) labeled samples of old classes are exactly clean, and 2) L intersection U = empty set.
    - prevent the model from fitting on wrong pseudo-labels
    - proposes simple baseline.
    - effectively leveraging the labeled data during the unsupervised clustering in unlabeled data [unsupervised step described in section 3.1]
    - compounds examples in two ways: 1) mix U examples with L samples; and 2) mix U examples with reliable anchors.
  - <embed src="https://mxahan.github.io/PDF_files/Openmix.pdf" width="100%" height="850px"/>

1. Zhao, B., & Han, K. (2021). Novel visual category discovery with dual ranking statistics and mutual knowledge distillation. Advances in Neural Information Processing Systems, 34.
  - semantic partitions of unlabelled images (new classes) by leveraging a labelled dataset (contains different but relevant categories of images) [RS]
  - two branch learning (one branch focusing on local part-level information and the other branch focusing on overall characteristics)
  - dual ranking statistics on both branches to generate pseudo labels for training on the unlabelled data
    - transfer knowledge from labelled data to unlabelled data
  - introduce a mutual KD method to allow information exchange and encourage agreement between the two branches for discovering new categories
  - *TP*: Joint optimization of many many losses (eq 10)

1. Han, K., Rebuffi, S. A., Ehrhardt, S., Vedaldi, A., & Zisserman, A. (2021). Autonovel: Automatically discovering and learning novel visual categories. IEEE Transactions on Pattern Analysis and Machine Intelligence.
  - self-supervised learning to train the representation from scratch on the union of labelled and unlabelled data (avoid bias of labelled data) [low-level features]
  - ranking statistics to transfer the model’s knowledge of the labelled classes [high level features]
  - optimizing a joint objective function on the labelled and unlabelled subsets of the data
  - Enable estimating the number of classes
  - Utilization of average clustering accuracy (ACC) and Cluster validity index (CVI) [silohouette index]

1. Schott, L., von Kügelgen, J., Träuble, F., Gehler, P., Russell, C., Bethge, M., ... & Brendel, W. (2021). Visual representation learning does not generalize strongly within the same domain. arXiv preprint arXiv:2107.08221.
  - Empirical paper to test if representation learning approaches correctly infer the generative factors of variation in simple datasets (visual tasks).
  - To learn effective statistical relationships, the training data needs to cover most combinations of factors of variation (like shape, size, color, viewpoint, etc.) [exponential issues]
    - large factor variation leads to out-of distribution.
      - As soon as a factor of variation is outside the training distribution, models consistently predict previously observed value
    - learning the underlying mechanisms behind the factors of variation should greatly reduce the need for training data and scale more with factors.
    - underlying data generation process
  - TP: Four dataset with various factors of variations. [dSprites, Shapes3D, MPI3D, celebglow]
    -  shape, size, color, viewpoint, etc
  - TP: models mostly struggle to learn the underlying mechanisms regardless of supervision signal and architecture.
    - Experimented with different controllable factor of variations.
  - Thoughts on assumption (inductive biases) for learning generalization
    - Representation format: PCA
    - Architectural bias: invariance and equivalence.
  - Demonstrate empirical results by varying FoVs (6 in totals)
    - Check for composition, interpolation, extrapolation, and decomposition (4.2)
      - Modular performance (good on the in-distribution data)
  - Good insights for different cases (section 5 conclusion)
    - Disentanglement helps on downstream task but not necessarily in the OOD cases
    - **empirically show that among a large variety of models, no tested model succeeds in generalizing to all our proposed OOD settings (extrapolation, interpolation, composition)**
    - Instead of extrapolating, all models regress the OOD factor towards the mean in the training set
    - The performance generally decreases when factors are OOD regardless of the supervision signal and architecture
  - Reiterate the importance of the data. (Even gan fails to learn that)

1. Chen, G., Peng, P., Wang, X., & Tian, Y. (2021). Adversarial reciprocal points learning for open set recognition. arXiv preprint arXiv:2103.00953.
  - Target: reduce the empirical classification risk on the labeled known data and the open space risk on the potential unknown data simultaneously.
  - TP formulate the open space risk problem from multi-class integration perspective, and model the unexploited extra-class space with a novel concept **Reciprocal Point**
    - ARPL: minimize the overlap of known and unknown distributions without loss of known classification accuracy **by**
      - RP is learned by the extra-class space with the corresponding known class
      - the confrontation among multiple known categories are employed to reduce the empirical risk.!!
      - an adversarial margin constraint to reduce the open space risk by limiting the latent open space constructed by RP!!
      - an instantiated adversarial enhancement method generate diverse and confusing training samples (To estimate the unknown distribution from open space) [using RP and known classes]
  - SOTA problems: Figure 1
  - argue that not only the known classes but also the potential unknown deep space should be modeled in the training
    - Well how? - By RP: - RP: Whats not something. (Reciprocal of the prototypical learning) [figure 3]
      - **key Idea**: finds the non-catness and tell cat if otherwise happens (nice), however, is it that easy?? computationally possible
      - Example based solutions!!
      - kinda one vs all setting!! (centre for all other classes)
  - Related work: Classifier with rejection option. OOD, Prototype learning
  - Good problem formulation: The adversarial section constrain open space.
  - Algo 1 (IDEA), **Algo 2 (implemntation details)**
  - **Gems** in 3.4 section
  - Adversarial setup for generating confusion samples. [Architecture in figure 5]
  - TP: adds an extra entropy based terms with GAN G maximization.
  - Look into the experimentation of the batch normalization.
  - Somehow connected to the disentanglement settings.
  - <embed src="https://mxahan.github.io/PDF_files/Adv_RPL.pdf" width="100%" height="850px"/>

1. Vaze, Sagar, Kai Han, Andrea Vedaldi, and Andrew Zisserman. "Open-set recognition: A good closed-set classifier is all you need." arXiv preprint arXiv:2110.06207 (2021).
  - demonstrate that the ability of a classifier to make the ‘none-of-above’ decision is highly correlated with its accuracy on the closed-set classes
  - RQ: whether a well-trained closed-set classifier can perform as well as recent algorithms
  - TP: show that the open-set performance of a classifier can be improved by enhancing its closed-set accuracy
     - TP: simentic shift benchmark??
  - Interested related works: Out-of-Distribution (OOD) detection, novelty detection, anomaly detection, novel category discovery, novel feature discovery
  - Different Baseline: Maximum Softmax probability (MSP), ARPL: Reciprocal point learning, (varies on how to calculate the confidence score)
  - <embed src="https://mxahan.github.io/PDF_files/osr_good_close.pdf" width="100%" height="850px"/>

1. Kodama, Yuto, Yinan Wang, Rei Kawakami, and Takeshi Naemura. "Open-set Recognition with Supervised Contrastive Learning." In 2021 17th International Conference on Machine Vision and Applications (MVA), pp. 1-5. IEEE, 2021.
  - TP: Explicitly uses distance learning (CL!!) to obtain the feature space for the open-set problem
    - Supcon, EVT to find the normality score.

## 2020

1. Han, K., Rebuffi, S. A., Ehrhardt, S., Vedaldi, A., & Zisserman, A. (2020). Automatically discovering and learning new visual categories with ranking statistics. arXiv preprint arXiv:2002.05714.
  - hypothesize that a general notion of what constitutes a “good class” can be extracted from labeled to Unlabeled
  - later paper worked on various ranking methods for unlabeled data.
  - utilize the metrics of deep transfer clustering.
  - very good visualization but kind of build on previous works.
  - <embed src="https://mxahan.github.io/PDF_files/Ncd_ranking_loss.pdf" width="100%" height="850px"/>

1. Chen, Guangyao, Limeng Qiao, Yemin Shi, Peixi Peng, Jia Li, Tiejun Huang, Shiliang Pu, and Yonghong Tian. "Learning open set network with discriminative reciprocal points." In European Conference on Computer Vision, pp. 507-522. Springer, Cham, 2020.
  - Reciprocal Point (RP), a potential representation of the extra-class space corresponding to each known category.
    - sample is classified to known or unknown by the otherness with RP

1. Geng, Chuanxing, Sheng-jun Huang, and Songcan Chen. "Recent advances in open set recognition: A survey." IEEE transactions on pattern analysis and machine intelligence 43, no. 10 (2020): 3614-3631.
  - Very good terminologies to get
  - Four types of class categories: Known known class (KKC), K Unknown C (KUC), UKC: provided side information, UUC
    - Figure 2 demonstrate goal for OSR

## 2019 and Earlier
1. Asano, Y. M., Rupprecht, C., & Vedaldi, A. (2019). Self-labelling via simultaneous clustering and representation learning. arXiv preprint arXiv:1911.05371.

1. Quintanilha, I. M., de ME Filho, R., Lezama, J., Delbracio, M., & Nunes, L. O. (2018). Detecting Out-Of-Distribution Samples Using Low-Order Deep Features Statistics.
  - a simple ensembling of first and second order deep feature statistics (mean and standard deviation within feature) can differentiate ID and OOD.
  - Figure 1: Plug-and-play propose solution. ![image](https://d3i71xaburhd42.cloudfront.net/6e1f7b326dd795377a631cf76fc5e5df05f1dce2/3-Figure1-1.png)
  - linear classifier over the neural activation stats. 

1. Liu, Z., Miao, Z., Zhan, X., Wang, J., Gong, B., & Yu, S. X. (2019). Large-scale long-tailed recognition in an open world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2537-2546).
  - *Aim:* classify among majority and minority classes, generalize from a few known instances, and acknowledge novelty upon a never seen instance.
  - TP: OLTR learning from naturally distributed data and optimizing accuracy over a balanced test set of head, tail, and open classes
  - methodologies: 1. dynamic Meta Embedding. .. conected to Self-attention
  - overall figure 2

1. Bendale, A., & Boult, T. (2015). Towards open world recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1893-1902).

1. Oza, P., & Patel, V. M. (2019). C2ae: Class conditioned auto-encoder for open-set recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2307-2316).
  - TO: an OSR algorithm using class conditioned auto-encoders with novel training and testing methodologies
    - 2 steps: 1. closed-set classification and, 2. open-set identification
    - utilize EVT to find the threshold for known/unknown.
    - *Encoder* learns the first task following the closed-set classification training pipeline, *decoder* learns the second task by reconstructing conditioned on class identity


1. Scheirer, W. J., Rocha, A., Micheals, R. J., & Boult, T. E. (2011). Meta-recognition: The theory and practice of recognition score analysis. IEEE transactions on pattern analysis and machine intelligence, 33(8), 1689-1695.
  - Utilize EVT for OSR (rough note to start)
  - figure 3 summarizes:
    - matches the distribution with EVT distribution and check for tail cases.
  -  EVT is analogous to a CTL, but tells us what the distribution of extreme values should look like as we approach the limit
    - Extreme value distributions are the limiting distributions that occur for the maximum (or minimum, depending on the data representation) of a large collection of random observations from an arbitrary distribution
    - falls in one of the three exponential family format.
  - *observation*: most recognition systems, the distance or similarity scores are bounded from both above and below
  - takes the tail of these scores, which are likely to have been sampled from the extrema of their underlying portfolios, and fits a Weibull distribution to that data.

1. Bendale, A., & Boult, T. E. (2016). Towards open set deep networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1563-1572).
  - Introduce OpenMax layer **[alternative to softmax]** to incorporate open set setting.
    - Modify the activation weights before softmax function [eq 2, aglo 1,2]
    - estimates the probability of an input being from an unknown class
    -  greatly reduces the number of obvious errors made by a deep network!!!
    -  provides bounded open space risk, thereby formally providing OSR solution
  - *key element in detecting unknown probability is to adapt Meta-Recognition concepts in the networks' penultimate layer activation patterns*
  - Utilize EVT to incorporate rejection probability.


1. Hsu, Y. C., Lv, Z., Schlosser, J., Odom, P., & Kira, Z. (2019). Multi-class classification without multi-class labels. arXiv preprint arXiv:1901.00544.
  - a new strategy for multi-class classification (no class-specific labels) using pairwise similarity between examples
  - present a probabilistic graphical model for it, and derive a loss function for DL
  - generalizes to the supervised, unsupervised cross-task, and semi-supervised settings
  - reduce the problem of classification to a meta problem (siamese network)
    - has the vibe of student teacher model [MCL]
    - pretty standard approach for forming a binary class from multi-class.

1. Hsu, Yen-Chang, Zhaoyang Lv, and Zsolt Kira. "Learning to cluster in order to transfer across domains and tasks." arXiv preprint arXiv:1711.10125 (2017).
  - perform tx learning across domains and tasks, formulating it as a problem of learning to cluster [KCL]
  - TP: i) design a loss function to regularize classification with a constrained clustering loss (learn a clustering network with the transferred similarity metric)!!
  - TP: ii) for cross-task learning (unsupervised clustering with unseen categories) propose a framework to reconstruct and estimate the no of semantic clusters
  - utilize the pairwise information in a fashion similar to constrained clustering
    - LCO: pairwise similarity (pre-contratstive set up: matching network)
  - <embed src="https://mxahan.github.io/PDF_files/Learn_cluster.pdf" width="100%" height="850px"/>

1. Han, K., Vedaldi, A., & Zisserman, A. (2019). Learning to discover novel visual categories via deep transfer clustering. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 8401-8409).
  - problem of discovering novel object categories in an image collection [DTC]
  - assumption: prior knowledge of related but different image classes
  - use such prior knowledge to reduce the ambiguity of clustering, and improve the quality of the newly discovered classes (how??)
  - TP: i) Extend DEC ii) improve the algorithm by introducing a representation bottleneck, temporal ensembling, and consistency (how??) [a method to estimate the number of classes in the unlabelled data]
    - ii) **modification:** account unlabeled data, include bottleneck, incorporate temporal ensemble and consistency.
  - TP: o transfers knowledge from the known classes, using them as probes to diagnose different choices for the number of classes in the unlabelled subset.
    - transfers knowledge from the known classes, using them as probes to diagnose different choices for the number of classes in the unlabelled subset.
  - learn representation from labeled data and fine-tune using unlabeled data!!!
  - Algorithm 1 [warm-up training, final training], algo 1 [estimate class no]
    - learning model params and centre simultaneously.
  - <embed src="https://mxahan.github.io/PDF_files/deep_tx_cluster.pdf" width="100%" height="850px"/>

1. Scheirer, Walter J., Anderson de Rezende Rocha, Archana Sapkota, and Terrance E. Boult. "Toward open set recognition." IEEE transactions on pattern analysis and machine intelligence 35, no. 7 (2012): 1757-1772.
  - “open set” recognition: incomplete world knowledge is present at training, and unknown classes can be submitted during testing
  - TP:  “1-vs-Set Machine,” which sculpts a decision space from the marginal distances of a 1-class or binary SVM with a linear kernel
  - In classification, one assumes there is a given set of classes between which we must discriminate. For recognition, we assume there are some classes we can recognize in a much larger space of things we do not recognize

1. Yang, H. M., Zhang, X. Y., Yin, F., & Liu, C. L. (2018). Robust classification with convolutional prototype learning. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3474-3482).
  - lack of robustness for CNN is caused by the softmax layer (discriminative and based on the assumption of closed world)
    - TP: Proposes convolutional prototype learning (CPL)
      - design multiple classification criteria to train
      - prototypical loss as regularizers
      - Looks like requires a lot of computations!
      - Put a constraint: classes need to be inside a circle [prototype loss]!!
        - How the heck it got connected to generative model !!
    - <embed src="https://mxahan.github.io/PDF_files/conv_proto_net.pdf" width="100%" height="850px"/>

1. Kim, Y., Yim, J., Yun, J., & Kim, J. (2019). Nlnl: Negative learning for noisy labels. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 101-110).
  - “input image belongs to this label” (Positive Learning; PL)
  - Negative learning: Randomly select other label than the not label !!!???
