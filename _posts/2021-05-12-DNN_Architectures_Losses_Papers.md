# Architectural contributions

## 2021

1. Ren, S., Gao, Z., Hua, T., Xue, Z., Tian, Y., He, S., & Zhao, H. (2021). Co-advise: Cross inductive bias distillation. arXiv preprint arXiv:2106.12378.

1. Li, Duo, Jie Hu, Changhu Wang, Xiangtai Li, Qi She, Lei Zhu, Tong Zhang, and Qifeng Chen. "Involution: Inverting the inherence of convolution for visual recognition." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12321-12330. 2021.
  - inherent principles of standard convolution for vision tasks, specifically spatialagnostic and channel-specific.
  - TP: Involution: a novel atomic operation for deep neural networks by inverting the aforementioned design principles of convolution
    - Claim: Involution is a simple instantiation of the attention.
  - involution operator could be leveraged as fundamental bricks to build the new generation of neural networks for visual recognition

1. Liu, Ze, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. "Video Swin Transformer." arXiv preprint arXiv:2106.13230 (2021).

1. Ke, Junjie, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. "MUSIQ: Multi-scale Image Quality Transformer." arXiv preprint arXiv:2108.05997 (2021).

1. Anonymous, . "Patches Are All You Need?." . In Submitted to The Tenth International Conference on Learning Representations .2022.

1. Neimark, Daniel, Omri Bar, Maya Zohar, and Dotan Asselmann. "Video transformer network." arXiv preprint arXiv:2102.00719 (2021).
  - long term information in the videos. (unlike 3D CNN that focus on 2s of video)[full video decision]
  - Focus on global attention.
  - Related works: i) spatial-temporal network: Slowfast, X3d, 3d convs ii) Transformer in CV: ViT, DeiT!, DETR!, VisTR!, iii) Transformer for long sequences:
  - VTN description: i) Spatial Backbone (2D image network). ii) temporal attention based encoder (Longformer) iii) Classification MLP head
  - Experimented with Three networks i) ViT-B-VTN (Imagenet 21K backbone, transformer) ii) R50/101 VTN (ResNET -50 /101 as backbone) iii) DeiT-B/BD/Ti-VTN (distilled as the backbone for VTN)
  - Ablation Study: Section 5

1. Kondratyuk, Dan, Liangzhe Yuan, Yandong Li, Li Zhang, Mingxing Tan, Matthew Brown, and Boqing Gong. "Movinets: Mobile video networks for efficient video recognition." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16020-16030. 2021.

1. Jaegle, Andrew, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. "Perceiver: General perception with iterative attention." arXiv preprint arXiv:2103.03206 (2021).

1. Jaegle, Andrew, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula et al. "Perceiver IO: A General Architecture for Structured Inputs & Outputs." arXiv preprint arXiv:2107.14795 (2021).

1. Tolstikhin, Ilya, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung et al. "Mlp-mixer: An all-mlp architecture for vision." arXiv preprint arXiv:2105.01601 (2021).

## 2020


1. Radosavovic, Ilija, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. "Designing network design spaces." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10428-10436. 2020.
  -

1. Beltagy, Iz, Matthew E. Peters, and Arman Cohan. "Longformer: The long-document transformer." arXiv preprint arXiv:2004.05150 (2020).

1. Feichtenhofer, Christoph. "X3d: Expanding architectures for efficient video recognition." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 203-213. 2020.
  - progressive forward expansion and backward contraction approaches!!

1. Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
  - CNN matching performance: large amount of data pretrain and transfer on mid/small sized classification tasks.


## 2019

1. Moon, Gyeongsik, Ju Yong Chang, and Kyoung Mu Lee. "Camera distance-aware top-down approach for 3d multi-person pose estimation from a single rgb image." In Proceedings of the ieee/cvf international conference on computer vision, pp. 10133-10142. 2019.
  - PoseNet for 3D posture estimation for multi person simultaneously.
    - Pipeline: human detection, root detection, and 3D mapping (fully learning based) [figure 1]
      - DetectNet: Mark R-CNN to crop human images
      - RootNet: Returns camera centered location of the cropped images (assign depth values to them) [kinda noisy process]
      - PoseNet: Parallel to RootNet
  - Experiment: Dataset: Human3.6M, MuCo-3DHP

1. Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for mobilenetv3." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1314-1324. 2019.

1. Feichtenhofer, Christoph, Haoqi Fan, Jitendra Malik, and Kaiming He. "Slowfast networks for video recognition." In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6202-6211. 2019.
  - Motivation: No need to treat spatial (semantics) and temporal dimension equally (Retinal ganglion cells)
  - TP: Two pathway model (Slow {low fps} and fast {high fps})
    - slow is the heavyweight computation (higher number of channel) and fast is the lightweight computation (lower number of channel)
    - Two pathway fused by lateral connections (fast to slow)
    - Table 1: summarizes the model
  - Experiments: kinetics, charades, AVA dataset
  - Nice ablation study sections
  - Non-degenerate temporal filter (CNN filter with temporal kernel size >1)

1. Radosavovic, Ilija, Justin Johnson, Saining Xie, Wan-Yen Lo, and Piotr Dollár. "On network design spaces for visual recognition." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1882-1890. 2019.


## 2018

1. Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV) (pp. 801-818).
   - Deeplab-V3 (semantic segmentation network)
   - combine the advantages from both spatial pyramid pooling and encoder-decoder sturcture.
   - explore the Xception model and apply the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network

1. Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Deep image prior." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 9446-9454. 2018.
  - Struction of the network matters (it contains information)
    - the heck they are talking about (they do iterative training!!!)
    - Kinda decoder generator from random prior.
  - fitting a randomly-initialized ConvNet to corrupted images works as a “Swiss knife” for restoration problem
  - the slowness and the inability to match or exceed the results of problem specific methods: two main limitations
  - Use: Image restoration tasks

1. Iyer, Ganesh, R. Karnik Ram, J. Krishna Murthy, and K. Madhava Krishna. "CalibNet: Geometrically supervised extrinsic calibration using 3D spatial transformer networks." In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 1110-1117. IEEE, 2018.
  - Used for lidar.

1. Tran, Du, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. "A closer look at spatiotemporal convolutions for action recognition." In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 6450-6459. 2018.
  - factorizing the 3D convolutional filters into separate spatial and temporal components yields significantly gains in accuracy
  - Empirical studies (?? types of studies) leads to design choices **R(2+1)D**
  - 3D network with residual learning
  - Really close to P3D network (especially P3D-A block)
    - Difference: Instead of three P3D interleaving blocks TP proposes homogeneous block
  - Benefits: Provides SOTA results, homogeneous block, Matches R3D params.

1. Wang, Xiaolong, Ross Girshick, Abhinav Gupta, and Kaiming He. "Non-local neural networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7794-7803. 2018.
  - Computes the response at a position as a weighted sum of the features at all positions.
    - Plugged in with may static image recognition system (pose, object detections).
    - Aim to capture long-term dependencies
  - Figure 2 sums all (design choice of f and g)

1. Xie, Saining, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. "Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification." In Proceedings of the European conference on computer vision (ECCV), pp. 305-321. 2018.
  - Four types of network to combine 3D with 2D
    - I2D: 2D CNN on multiple frames
    - I3D: 3D over space and time
    - Bottom Heavy I3D: 3D in lower layer and 2D on higher layers
    - Top Heavy I3D: 2D on lower and 3D on higher layers
  - TP: Proposes separable 3D convolution [figure 2]
  - Read intro: says all
    - Performs network surgery (empirical analysis) to create new networks

1. Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018).
  - Architectural search in Differentiable manner (why? How? )
    - Why: differential search space for gradient descent in architecture search (using some relaxations), faster than non-differential counterparts (RL based)!
  - Experiments with both images and NLP tasks.

1. Lee, Hyungtae, Sungmin Eum, and Heesung Kwon. "Cross-domain CNN for hyperspectral image classification." In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium, pp. 3627-3630. IEEE, 2018.


## 2017
1. Chen, Liang-Chieh, George Papandreou, Florian Schroff, and Hartwig Adam. "Rethinking atrous convolution for semantic image segmentation." arXiv preprint arXiv:1706.05587 (2017).
  - Multi scale atrous CNN for Semantic image segmentation & modified ASPP
  - Enlarge Field-of-View
  - Removed DenseCRF
  - Dataset: PASCAL, VOC 2012
  - All 3x3 with different rate
  - Related works: Context Module!

1. Carreira, Joao, and Andrew Zisserman. "Quo vadis, action recognition? a new model and the kinetics dataset." In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6299-6308. 2017.
  - I3D/R3D model

1. Chollet, François. "Xception: Deep learning with depthwise separable convolutions." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1251-1258. 2017.

1. Zhang, Xiangyu, Xinyu Zhou, Mengxiao Lin, and Jian Sun. "Shufflenet: An extremely efficient convolutional neural network for mobile devices.(2017)." arXiv preprint arXiv:1707.01083 (2017).

1. Howard, Andrew G., Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).

1. Qiu, Zhaofan, Ting Yao, and Tao Mei. "Learning spatio-temporal representation with pseudo-3d residual networks." In proceedings of the IEEE International Conference on Computer Vision, pp. 5533-5541. 2017.
  - TP: proposes computational efficient 3D CNN (and their extensions) [p3d network]
    - Decomposes 3D CNN as DD spatial filter and 1D temporal filter
    - Combines of 3-inter-leaving P3D blocks [figure 3].

1. Xie, Saining, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. "Aggregated residual transformations for deep neural networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1492-1500. 2017.
  - ResNeXt: highly modulated network (figure 1) [another network engineering with a very simple idea]
  - Name origin: as it adds **Next dimension: Cardinality** so it refers to ResNeXt
    - Not Inception as no downsampling in between.
  - Why? how is it better than the ResNet? Introduces new parameters " Cardinality (the size of set of transformation)"! [with addition of deeper and wider in ResNet]
  - TP: extends *split-transform-merge* and ResNet
  - TP: Transformation to Low dimension and outputs are *aggregated* by summation.
  - Related works: Multi-branch CN, grouped CN, compressed CN, Ensembling.
  - Experiments with ImageNet, COCO object detection. [outperforms ResNet, Inception, VGG.]


## 2016  and Earlier

1. Su, Hang, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller. "Multi-view convolutional neural networks for 3d shape recognition." In Proceedings of the IEEE international conference on computer vision, pp. 945-953. 2015.
  - MVCNN

1. Mahendran, Aravindh, and Andrea Vedaldi. "Understanding deep image representations by inverting them." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5188-5196. 2015.
  - conduct a direct analysis of the visual information contained in representations (by image reconstruction)
    - inversion problem as regularlized inversion problem.
    - several layers in CNNs retain photographically accurate information about the image, with different degrees of geometric and photometric invariance
  - TP: Propose a general framework to invert representation (look into how much information is preserved)
  - Encoder-decoder setup!! (related work: DeConvNet) : Derivative in image domain!

1. Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." In European conference on computer vision, pp. 818-833. Springer, Cham, 2014.
  - Visualization for diagnostic role: study to discover the performance contribution from different model layers
  - TP: multi-layered Deconvolutional Network (deconvnet) to project the feature activations back to the input pixel space.
    - novel way to map these activities back to the input pixel space (Unpool, rectify, and filtering)

1. Wang, Limin, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. "Temporal segment networks: Towards good practices for deep action recognition." In European conference on computer vision, pp. 20-36. Springer, Cham, 2016.
  - Video activity recognition: segmentation and aggregation with multimodal (figure 1)
  - Experiments: i) Dataset: HMDB51 and UCF101

1. He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning for image recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

1. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012): 1097-1105.

1. Srivastava, Nitish, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. "Dropout: a simple way to prevent neural networks from overfitting." The journal of machine learning research 15, no. 1 (2014): 1929-1958.

# Metrics and Losses

## 2021

1. Leng, Z., Tan, M., Liu, C., Cubuk, E. D., Shi, J., Cheng, S., & Anguelov, D. (2021, September). PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions. In International Conference on Learning Representations.
  - New framework for loss function (taylor series expansion of log function)
  - PolyLoss allows the adjustment of polynomial bases depending on the tasks and datasets (subsumes cross-entropy loss and focal loss as special cases)
    - Experiment to support the requirement of adjustment.
    - Introduces an extra hyperparameters
  - Contributions: i) insight of common losses ii)  intuitive understanding of requirement to design different loss functions tailored to different imbalanced datasets!!
  - Proposes to manipulate the weight so the polynomial components
    - modifying the polynomial weights helps to go beyond the CE loss accuracy?
      - is it individual class dependent? For all classes or biased setup?
    - *the idea is to provide extra weights for the initial terms*

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
1. Boudiaf, Malik, Jérôme Rony, Imtiaz Masud Ziko, Eric Granger, Marco Pedersoli, Pablo Piantanida, and Ismail Ben Ayed. "A unifying mutual information view of metric learning: cross-entropy vs. pairwise losses." In European Conference on Computer Vision, pp. 548-564. Springer, Cham, 2020.
  - TP: a theoretical analysis to link the cross-entropy to several well-known and recent pairwise losses from two perspective
    - i) explicit optimization insight and ii) discriminative and generative views of the mutual information between the labels and the learned features
  - Experimented with four different DML benchmarks (CUB200, Cars-196, Stanford Online Products (SOP) and In-Shop Clothes Retrieval (In-Shop)) [also used by MS loss]
  - TP proves that minimizing cross-entropy can be viewed as an approximate bound optimization of a more complex pairwise loss. [interesting section 4.3]

1. Sainburg, Tim, Leland McInnes, and Timothy Q. Gentner. "Parametric UMAP embeddings for representation and semi-supervised learning." arXiv preprint arXiv:2009.12981 (2020).
  - Good starting note for the UMAP and tSNE. Parametric extension for the UMAP
  - [link](https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668) May contain Some bias towards UMAP

1. Sun, Yifan, Changmao Cheng, Yuhan Zhang, Chi Zhang, Liang Zheng, Zhongdao Wang, and Yichen Wei. "Circle loss: A unified perspective of pair similarity optimization." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6398-6407. 2020.

## 2019

1. Wang, Xun, Xintong Han, Weilin Huang, Dengke Dong, and Matthew R. Scott. "Multi-similarity loss with general pair weighting for deep metric learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5022-5030. 2019.
  - (TP) Establish a General Pair Weighting (GPW) framework: casts the sampling problem of deep metric learning into a unified view of pair weighting through gradient analysis, (tool for understanding recent pair-based loss functions)
  - (TP) Various existing pair-based methods are compared and discussed comprehensively, with clear differences and key limitations identified;
  - (TP) Proposes new loss called multi-similarity loss (MS loss) under the GPW,
    - implemented in two iterative steps (i.e., **mining**: involves P and **weighting**: Involves S,N): consider three similarities for pair weighting, providing a more principled approach for collecting and weighting informative pairs
      - 3 similarities (self-similarity: S, Negative relative similarity: N, Positive relative similarity: P) [figure 2]
      - state-of-the-art performance on four image retrieval benchmarks (CUB200, Cars-196, Stanford Online Products (SOP) and In-Shop Clothes Retrieval (In-Shop))
  - [github notes]
  - <embed src="https://mxahan.github.io/PDF_files/msgpw.pdf" width="100%" height="850px"/>


1. Cheung, Brian, Alex Terekhov, Yubei Chen, Pulkit Agrawal, and Bruno Olshausen. "Superposition of many models into one." arXiv preprint arXiv:1902.05522 (2019).
  - Multiply weights to project them in orthogonal space and sum them.

## 2018

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
