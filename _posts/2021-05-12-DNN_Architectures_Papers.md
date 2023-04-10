---
tags: Papers
---

# Architectural contributions

## 2022

1. Jia, M., Tang, L., Chen, B. C., Cardie, C., Belongie, S., Hariharan, B., & Lim, S. N. (2022, November). Visual prompt tuning. In *Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXIII* (pp. 709-727). Cham: Springer Nature Switzerland.

     - inspirated from efficiently tuning large language models, VPT introduces only a small amount of trainable parameters in the **input space** while keeping the model backbone frozen
       -  VPT instead adds extra parameters in the input space. 
       -  TP: modify the input to the Vision Transformer
          -   introduces a small amount of task-specific learnable parameters into the input space, freezing the entire pre-trained Transformer backbone during downstream training
          -  prepended into the input sequence of each Transformer layer and learned together with a linear head during fine-tuning
     - **RQ:** what is the best way to adapt large pre-trained Transformers to downstream tasks in terms of effectiveness and efficiency?
     - Gap: ViT fine-tuning: Classifier training only underperform!
     - Two approaches: VPT-shallow, VPT-Deep

1. Lee, H., Eum, S., & Kwon, H. (2022). Negative Samples are at Large: Leveraging Hard-distance Elastic Loss for Re-identification. In *European Conference on Computer Vision* (pp. 604-620). Springer, Cham.

     - Hard-distance elastic loss

     - $$
       \mathcal{L}_{q}(t)= \sum_{p \in P}\max(d_{pq}-t, 0) + \sum_{p \in N}\max(t -d_{nq}, 0) \\
       \text{Optimal Boundary } t^* = \underset{t}{\arg \min}\mathcal{L}_q \\
       \text{Gradient Properties } \frac{d\mathcal{L}}{dt} = \sum_{p \in P}-\mathbb{1}_{t<d_{pq}} + \sum_{p \in N}\mathbb{1}_{t>d_{pq}} = N_{hn}(t) - N_{hp}(t) \\
       \text{Mathe conditions for optimal conditions } N_{hp} = N_{hn}\\
       $$
       
     - Dataset:  Three re-ID datasets: VeRi-776, Market-1501, and VeRi-Wild.

1. Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A convnet for the 2020s. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 11976-11986).

     - Propose methods to improve convnet performance. (ConvNeXt)
     - ![](https://i0.wp.com/syncedreview.com/wp-content/uploads/2022/01/image-44.png?resize=680%2C813&ssl=1)

1. Abid, A., Yuksekgonul, M., & Zou, J. (2022, June). Meaningfully debugging model mistakes using conceptual counterfactual explanations. In International Conference on Machine Learning (pp. 66-88). PMLR.
     -  propose a systematic approach, conceptual counterfactual explanations (CCE) to explain why a classifier makes a mistake on a particular test sample(s) in terms of human-understandable .

1. Paul, S., & Chen, P. Y. (2022, June). Vision transformers are robust learners. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 2, pp. 2071-2081).

     - Compared with CNN and showed the better performance by ViT

     - Reason behind ViT robustness
       - Attention, Role of pretraining, robust to image masking
         - Fourier Spectrum Analysis Reveals Low Sensitivity for ViT
         - Adversarial Perturbations of ViT Has Wider Spread in Energy Spectrum

1. Chen, T., Zhang, Z., Cheng, Y., Awadallah, A., & Wang, Z. (2022). The Principle of Diversity: Training Stronger Vision Transformers Calls for Reducing All Levels of Redundancy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 12020-12030).

   - ViT *believed to* higher modeling capacity and representation flexibility

       - ViTs often suffer from over-smoothening (redundant models)
         - makes effective learning capacity of ViTs “collapsed”
       - studies the ubiquitous existence of *redundancy* at all three levels
         - Patch embedding, attention map, weight space.
       - **TP**: principle of diversity for training ViTs,
         - presenting corresponding regularizers (encourage representation diversity and coverage at each of those levels)
         - enabling capturing more discriminative information
         - **Diversity Regularization**
       - ViT training suffers from considerable instability, especially when going deeper
         - Over-smoothening: global information aggregation among all patches encourages their representations to become overly similar (causing substantially degraded discrimination ability)
         - **Solution**
           - contrastive-based regularization to diversity patch embedding
           - directly refines the self-attention maps via CNN aggregation to augment local patterns.
         - systematically demonstrate the ubiquitous existence of redundancy!!!
       - Related work: i) (re-)injecting locality via convolution-like structures and fusing global and local contexts, for self-attention
         - adopting patch-wise contrastive or mixing loss to boost diversity, for patch embeddings
   - **TP**: Measures redundancy by measuring distance based metrics (patch and attention and model weights).
   - Eliminating tri-level redundancy: section 3.2


1. Chan, S. C., Santoro, A., Lampinen, A. K., Wang, J. X., Singh, A., Richemond, P. H., ... & Hill, F. (2022). Data Distributional Properties Drive Emergent Few-Shot Learning in Transformers. arXiv preprint arXiv:2205.05055.

     - In-context learning emerges when the training data exhibits particular distributional properties such as *burstiness* (items appear in clusters rather than being uniformly distributed over time) and having *large numbers of rarely occurring classes*.
       - ‘in-context learning’: output is conditioned on a context. (kinda meta learning.)

     - **finding 1:** in-context learning traded off against more conventional weight-based learning, and models were unable to achieve both simultaneously

     - **finding 2:** the two modes of learning could co-exist in a single model when it was trained on data following a skewed Zipfian distribution

     - How future work might encourage both in-context and in-weights learning in domains beyond language.

     - architecture and data are both key to the emergence of in-context learning

## 2021

1. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 10012-10022).

   - address these differences of language and vision domain (scale variation of visual entities and high pixel resolutions)
   - propose a hierarchical Transformer whose representation is computed with Shifted windows (SWin)
     - brings greater efficiency by limiting self-attention computation to non-overlapping local windows, allow for cross-window connection
     - linear computational complexity with respect to image size
     - shift of the window partition between consecutive self-attention layers
   - compatible with a broad range of vision tasks: image classification (Imagenet-1K), dense prediction tasks (Obj. detection e.g. coco), semantic segmentation (ADE20kval)
   - hierarchical feature maps of the Swin can leverage techniques for dense prediction e.g. feature pyramid networks (FPN) or U-Net
   -  linear computational complexity is achieved by computing SA locally within non-overlapping windows that partition an image
   - Overall Architecture: 
     - Patching 4x4 provides H/4xW/4x48 to linear embedding of dimension C to H/4xW/4 tokens of C dimension. (stage 1)
     - Maintain the number of token H/4xW/4
     - Patch merging (2x2) and dimension reduction (4C to 2C)
       - So on so forth. 
       - Swin tx replace standard tx. 

1. Srinivas, A., Lin, T. Y., Parmar, N., Shlens, J., Abbeel, P., & Vaswani, A. (2021). Bottleneck transformers for visual recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16519-16529).

   - backbone architecture that incorporates self-attention for multiple computer vision tasks
     - By just replacing the spatial convolutions with global self-attention in the final three bottleneck blocks of a ResNet

   - ResNet bottleneck blocks with self-attention can be viewed as Transformer blocks

   - Nice taxonomy in figure 2

   1. Wang, W., Xie, E., Li, X., Fan, D. P., Song, K., Liang, D., ... & Shao, L. (2021). Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 568-578).
     - **TP**: Spatial reduction before the key and value encoding. (figure 4) [pvit]

1. Khan, S., Naseer, M., Hayat, M., Zamir, S. W., Khan, F. S., & Shah, M. (2021). Transformers in vision: A survey. ACM Computing Surveys (CSUR).

     - Transformers require minimal inductive biases for their design and are naturally suited as set-functions.

     - fundamental concepts behind the success of Transformers i.e., self-attention, large-scale pre-training, and bidirectional feature encoding.

     - taxonomy of the network design space and highlight the major strengths and shortcomings of the existing methods.

     - Foundation: Self-attention and pre-training
         - difference of self-attention with convolution:
           - CNN static filter, permutation invariant
           - SA theoretically a more flexible operation (can model the behavior of CNN towards encoding local features
           - self-attention provides the capability to learn the global as well as local features (provide expressivity to adaptively learn CNN kernel weights as well as the receptive field)
     - Multi-head vision tx
         - Uniform Scale: ViT
           - Multi-scale:  number of tokens is gradually reduced while the token feature dimension is progressively increased
           - P-ViT and co.
           - vision transformer with convolution

1. Khan, S., Naseer, M., Hayat, M., Zamir, S. W., Khan, F. S., & Shah, M. (2021). Transformers in vision: A survey. ACM Computing Surveys (CSUR).

1. Tuli, S., Dasgupta, I., Grant, E., & Griffiths, T. L. (2021). Are Convolutional Neural Networks or Transformers more like human vision?. arXiv preprint arXiv:2105.07197.

     - decision function in a ML system is determined by i) the data to which the system is exposed, ii) model inductive bias
         - study and compare CNNs) and the Vision Transformer (ViT)
           - ViT has weaker inductive bias (relax the translation invariance)
             - CNN: each patch in an image is processed by the same weights.
             -  have higher shape bias and are largely more consistent with human errors
           - using new metrics for examining error consistency with more granularity
           - CNN tend to classify images by texture rather than by shape
           - CNNs struggle to recognize sketches that preserve the shape rather than texture
     - Stylized ImageNet: can be used to check model bias.
      - **Metrics for error consistency:** two systems can differ in which stimuli they fail to classify correctly, which is not captured by accuracy metrics.
        - Shape bias testing: SIN dataset
          - ViT has a higher shape bias than traditional CNNs

1. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021, July). Training data-efficient image transformers & distillation through attention. In International Conference on Machine Learning (pp. 10347-10357). PMLR.

     - competitive convolution-free transformers by training on Imagenet only
         - a teacher-student distillation strategy specific to transformers (distillation token)
           - Extra token like the class token.
           -  using a convnet teacher gives better performance than using a transformer
     - interest of this token-based distillation (convnet as a teacher)
      - Transformer “do not generalize well when trained on insufficient amounts of data” (research gap)

      - This Paper:  Data-efficient image Transformers (Deit 4)

1. Ren, S., Gao, Z., Hua, T., Xue, Z., Tian, Y., He, S., & Zhao, H. (2021). Co-advise: Cross inductive bias distillation. arXiv preprint arXiv:2106.12378.

     - a novel distillation-based method to train vision transformers
       - TP: introduce lightweight teachers with different architectural inductive biases (e.g., convolution and involution) to co-advise the student ViT
         - cross inductive bias distillation method
         - Solve DeiT problems
           - the trained transformer is over-influenced by the inductive bias of the teacher CNN and mirrors CNN’s classification error

     - Three types of architecture: CNNs, transformers, INN

     - teacher model’s intrinsic inductive bias matters much more than its accuracy

     - CNNs and INNs are inclined to learn texture and structure respectively

     - convolution operator is spatial agnostic and channel-specific, while an involution kernel is shared across channels and distinct in the spatial extent.

     - theoretically proven that the self-attention mechanism used in transformers is at least as expressive as any convolution layer.

     - *Summary*: Figure 2

1. Li, Duo, Jie Hu, Changhu Wang, Xiangtai Li, Qi She, Lei Zhu, Tong Zhang, and Qifeng Chen. "Involution: Inverting the inherence of convolution for visual recognition." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12321-12330. 2021.

     - inherent principles of standard convolution for vision tasks, specifically spatialagnostic and channel-specific.
       - spatial-agnostic: guarantees the efficiency of convolution kernels by reusing them among different locations and pursues translation equivalence
       - Channel specific: responsible for collecting diverse information encoded in different channel
       
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

1. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020, August). End-to-end object detection with transformers. In European conference on computer vision (pp. 213-229). Springer, Cham.
     - DEtection TRansformer [DETR]

1. Goyal, A., & Bengio, Y. (2020). Inductive biases for deep learning of higher-level cognition. arXiv preprint arXiv:2011.15091.

     - *empirical* paper on model architecture and learning.

     - argue that having larger and more diverse datasets is important but insufficient without good architectural inductive biases.

     -  can compensate lack of sufficiently powerful priors by more data

1. Radosavovic, Ilija, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. "Designing network design spaces." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10428-10436. 2020.
    -

1. Beltagy, Iz, Matthew E. Peters, and Arman Cohan. "Longformer: The long-document transformer." arXiv preprint arXiv:2004.05150 (2020).

1. Feichtenhofer, Christoph. "X3d: Expanding architectures for efficient video recognition." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 203-213. 2020.
      - progressive forward expansion and backward contraction approaches!!

1. Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

     - CNN matching performance: large amount of data pretrain and transfer on mid/small sized classification tasks.

     - Seminal paper in vision architecture (tx in vision domain)

     - Small sample size overfit the model

     - Make fewer assumption about the data.


## 2019


1. Yin, D., Gontijo Lopes, R., Shlens, J., Cubuk, E. D., & Gilmer, J. (2019). A fourier perspective on model robustness in computer vision. Advances in Neural Information Processing Systems, 32.

     - investigate recently observed tradeoffs caused by Gaussian data augmentation and adversarial training
       - both methods improve robustness to corruptions that are concentrated in the high frequency domain while reducing robustness to corruptions in the low frequency domain
       - **personal thought: Architectural texture bias**

     - TP: AutoAugment is good

     - TP result:
       - Gaussian data augmentation and adversarial training bias models towards low frequency information
         - Does low frequency data augmentation improve robustness to low frequency corruptions? - **No, low freq are important to learn**

1. Cordonnier, J. B., Loukas, A., & Jaggi, M. (2019). On the relationship between self-attention and convolutional layers. arXiv preprint arXiv:1911.03584.

     - attention layers can perform convolution and, indeed, they often learn to do so in practice.

     - prove that a multi-head SA layer with sufficient number of heads is at least as powerful as any convolutional layer.

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

  - <embed src="https://mxahan.github.io/PDF_files/slowfast.pdf" width="100%" height="850px"/>

1. Radosavovic, Ilija, Justin Johnson, Saining Xie, Wan-Yen Lo, and Piotr Dollár. "On network design spaces for visual recognition." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1882-1890. 2019.

## 2018

1. Shang, C., Ai, H., Zhuang, Z., Chen, L., & Xing, J. (2018, November). Zoomnet: Deep aggregation learning for high-performance small pedestrian detection. In *Asian Conference on Machine Learning* (pp. 486-501). PMLR.

1. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).

     - focus on the channel relationship of CNN and propose a novel architectural unit “Squeeze-and-Excitation” (SE) block

     - adaptively recalibrates channel-wise feature responses by explicitly modeling inter dependencies between channels

     - improvements in performance over SoTA CNNs at slight additional computational cost

     - Takes each channel, encode and decode them to finally marge again. (fig 2,3)

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

  - <embed src="https://mxahan.github.io/PDF_files/non_local_neuron.pdf" width="100%" height="850px"/>

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
1. Arpit, D., Jastrzębski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., ... & Lacoste-Julien, S. (2017, July). A closer look at memorization in deep networks. In *International conference on machine learning* (pp. 233-242). PMLR.

     - Contributions
       - qualitative differences in DNN optimization behavior on real data vs. noise. In other words, DNNs do not just memorize real data
       - . DNNs learn simple patterns first, before memorizing (content-aware)
       - Obviously: Regularization helps. 
     - Critical Sample Ratio: Experimented with curated artificial dataset. 

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

     - <embed src="https://mxahan.github.io/PDF_files/pseudo_3D.pdf" width="100%" height="850px"/>

1. Xie, Saining, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. "Aggregated residual transformations for deep neural networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1492-1500. 2017.

     - ResNeXt: highly modulated network (figure 1) [another network engineering with a very simple idea]
         - Name origin: as it adds **Next dimension: Cardinality** so it refers to ResNeXt
           - Not Inception as no downsampling in between.
     - Why? how is it better than the ResNet? Introduces new parameters " Cardinality (the size of set of transformation)"! [with addition of deeper and wider in ResNet]
      - TP: extends *split-transform-merge* and ResNet
     
      - TP: Transformation to Low dimension and outputs are *aggregated* by summation.
     
      - Related works: Multi-branch CN, grouped CN, compressed CN, Ensembling.
     
      - Experiments with ImageNet, COCO object detection. [outperforms ResNet, Inception, VGG.]
      - ![ResNeXt summary](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_4.32.52_PM.png)

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

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
     - Resnet with bottleneck ideas.

1. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012): 1097-1105.

1. Srivastava, Nitish, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. "Dropout: a simple way to prevent neural networks from overfitting." The journal of machine learning research 15, no. 1 (2014): 1929-1958.
