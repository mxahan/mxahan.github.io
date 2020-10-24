# Introduction

This blog contains paperlist I want and plan to go through. For detail paper read please refer to the Paper summary section. Here we will store lots of papers title or maybe little summary considering I will be storing only interested papers. We can't worry about everything and nothing. We must stay focused and broad at the same time. Very true! By the way, I will appreciate any suggestion on the paperlist. I know the Feynman's thoughts on knowing name, which means nothing, it's just the entrance to the grand scheme of the world.

1. Fernandes, Patrick, Miltiadis Allamanis, and Marc Brockschmidt. "Structured neural summarization." arXiv preprint arXiv:1811.01824 (2018).

1. Du, Yilun, and Igor Mordatch. "Implicit Generation and Modeling with Energy Based Models." In Advances in Neural Information Processing Systems, pp. 3603-3613. 2019.

1. Grathwohl, Will, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, and Kevin Swersky. "Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One." arXiv preprint arXiv:1912.03263 (2019).

1. Kingma, Durk P., and Prafulla Dhariwal. "Glow: Generative flow with invertible 1x1 convolutions." In Advances in Neural Information Processing Systems, pp. 10215-10224. 2018.
  - Actnorm
  - invertible 1x1 convolution
  - affine coupling layers

1. Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016).

1. Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. "Distributed representations of words and phrases and their compositionality." In Advances in neural information processing systems, pp. 3111-3119. 2013.
  - Skip gram model and its extensions: skip gram tries to maximize the log probaiblites <img src="https://latex.codecogs.com/gif.latex?1/T\sum_1^T\sum_{-c<=j<=c, j\neq 0}\log p(w_{t+j}|w_t)">; the probabilities are defined by softmax of the vector represenation of the words. The issue is computation of derivatives of log.
  - Hierarchical Softmax: Tree structure from root to the word
  - Negative Sampling: Adds with target and penalize for noise words. Noise Contrastive Estimation.
  - Subsampling of Frequent words.

1. Wang, Yuxuan, R. J. Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J. Weiss, Navdeep Jaitly, Zongheng Yang et al. "Tacotron: Towards end-to-end speech synthesis." arXiv preprint arXiv:1703.10135 (2017).

1. Sinh Huynh, Rajesh Krishna Balan, JeongGil Ko, and Youngki Lee. 2019. VitaMon: measuring heart rate variability using smartphone front camera. In Proceedings of the 17th Conference on Embedded Networked Sensor Systems (SenSys ’19). Association for Computing Machinery, New York, NY, USA, 1–14. DOI:https://doi.org/10.1145/3356250.3360036
  - [link](https://docs.google.com/presentation/d/1w3h0b1xqa4SBKISvUv4Yu_9kmAxiyG8j15_vwC9TEHI/edit?usp=sharing)

1. Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).
  - Knowledge distillation
  - Distillation and the effect of temperature.
    - section 2 and 2.1 are important
  - Training small network to mimic the large network.
  - Train small network to learn the features and logits of the large network.
  - Softmax, temperature and the MSE with the prediction
  - Experimented with MNIST, speech and Specialist models.

1. Vinyals, Oriol, Charles Blundell, Timothy Lillicrap, and Daan Wierstra. "Matching networks for one shot learning." In Advances in neural information processing systems, pp. 3630-3638. 2016.

1. Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).

1. Graves, Alex, Greg Wayne, and Ivo Danihelka. "Neural turing machines." arXiv preprint arXiv:1410.5401 (2014).

1. Zhao, Junbo, Michael Mathieu, and Yann LeCun. "Energy-based generative adversarial network." arXiv preprint arXiv:1609.03126 (2016).
  - D: Assign low energy to data distribution (not normalized probabilities)
  - G: Sample data from the low energy by a parameterized function.
  - Convergence proof using hinge loss.

1. ani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." In Advances in neural information processing systems, pp. 5998-6008. 2017.
  - Under construction

1. Radford, Alec, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. "Improving language understanding by generative pre-training." (2018): 12.
  - GPT paper

1. Denton, Emily L. "Unsupervised learning of disentangled representations from video." In Advances in neural information processing systems, pp. 4414-4423. 2017.
  - Encoder-Decoder set up for the disentangled
  - Hypothesis: Content (time invariant) and Pose (time variant)
  - Two Encoders for the pose and content; Concatenate the output for single Decoder
  - Introduce adversarial loss
  - Video generation conditioned on context, and pose modeling via LSTM.

1. Lu, Jiasen, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. "12-in-1: Multi-task vision and language representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10437-10446. 2020.
  - MTL + Dynamic "stop and go" schedule.
  - ViLBERT base architecture.

1. Misra, Ishan, and Laurens van der Maaten. "Self-supervised learning of pretext-invariant representations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6707-6717. 2020.
  - Pretraining method
  - Pretext learning with transformation invariant + data augmentation invariant
  - Use contrastive learning (See NCE)
      - Maximize MI
  - Motivation from predicting video frames
  - Experiment of jigsaw pretext learning
  - Hypothesis: Representation of image and its transformation should be same
  - Use different head for image and jigsaw counterpart of that particular image.
      - Motivation for learning some extra things by different head network
  - Noise Contrastive learning (contrast with other images)
  - As two head so two component of contrastive loss. (One component to dampen memory update.)
  - Implemented on ResNet

1. Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Split-brain autoencoders: Unsupervised learning by cross-channel prediction." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1058-1067. 2017.
  - Extension of autoencoders to cross channel prediction
    - Predict one portion to other and vice versa + loss on full reconstruction.
    - Two disjoint auto-encoders.
  - Tried both the regression and classification loss

1. Srinivas, Aravind, Michael Laskin, and Pieter Abbeel. "Curl: Contrastive unsupervised representations for reinforcement learning." arXiv preprint arXiv:2004.04136 (2020).

1. Chen, Ting, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. "A simple framework for contrastive learning of visual representations." arXiv preprint arXiv:2002.05709 (2020).
  - Truely simple! (SimCLR)
  - Two transfers for each image and representation
  - Same origin image should be more similar than the others.
  - Contrastive (negative) examples are from image other than that.
  - A nonlinear projection head followed by the representation helps.

1. Donahue, Jeff, Philipp Krähenbühl, and Trevor Darrell. "Adversarial feature learning." arXiv preprint arXiv:1605.09782 (2016).
  - Bidirectional GAN
  - Image to feats (encoder), random feats to image (generator)
  - Encoder and Generator works together to fool the discriminator.
  - Discriminator takes (image, feats)
  - Discriminator needs to distinguish between joint distribution of (image, encoded), and (generate image, random latent space).
  - Making sure the both are the same distribution!!

1. Brock, Andrew, Jeff Donahue, and Karen Simonyan. "Large scale gan training for high fidelity natural image synthesis." arXiv preprint arXiv:1809.11096 (2018).
  - Training large GAN and report of what happened! and analysis!
  - Regularization for spectral norm and orthogonality constraint.

1. Miyato, Takeru, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. "Spectral normalization for generative adversarial networks." arXiv preprint arXiv:1802.05957 (2018).
  - Spectral normalization to tackle instability in gan training
  - Discriminator network in focus. (needs restriction)
  - Spectral normalization
    - Lipschitz constant of hyperparameters are bounded
  - matrix/l2norm(matrix) [consult paper for the derivation]
  - Power iteration for calculation of l2norm (Dominant singular value)

1. Chen, Xi, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. "Infogan: Interpretable representation learning by information maximizing generative adversarial nets." In Advances in neural information processing systems, pp. 2172-2180. 2016.
  - New component head (Q) from discriminator to find disentanglement representation.
  - Regularization upon the MI between (c, G(z,c)) [we want it to maximum]
  - Proof of ELBO to Mutual information (4 and 5 equation of paper)
    - QQ: ELBO always negative and MI positive so why we need this bound! ?
  - Notion of Q [potential posterior of x to content c] (takes fake_image to get c)
  - G takes content and RVs [G, Q works against G]
  - look back at code to better understand it [c, from the fake_image]

1. Furlanello, Tommaso, Zachary C. Lipton, Michael Tschannen, Laurent Itti, and Anima Anandkumar. "Born again neural networks." arXiv preprint arXiv:1805.04770 (2018).
  - Knowledge distillation and repeated Training
    - Not focus on the compression but improvement.
  - Teacher to student1 to student2 to student3 and so on.
    - Finally ensemble the learned weights.
  - Dark Knowledge! Knowledge about similar things [soft label of images.]
    - Nutshell math: Total gradient is sum of teacher + students cross entropy gradient [logits differences], if teacher is not so confident then reduces the contribution as in first term.
    - [section 3.2] two treatment to reason the effectiveness of the dark knowledge.
  - Sequence of teaching selves.

1. Fort, Stanislav, Huiyi Hu, and Balaji Lakshminarayanan. "Deep ensembles: A loss landscape perspective." arXiv preprint arXiv:1912.02757 (2019).

1. LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "Deep learning." nature 521, no. 7553 (2015): 436-444.

1. Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
  - Bridge the unidirection of the GPT pretraining.
  - Trained on two unsupervised task
    - Masked language model (masked word to predict)
    - Next sentence Prediction (sentence relationship)

1. Du, Xianzhi, Tsung-Yi Lin, Pengchong Jin, Golnaz Ghiasi, Mingxing Tan, Yin Cui, Quoc V. Le, and Xiaodan Song. "SpineNet: Learning scale-permuted backbone for recognition and localization." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11592-11601. 2020.
  - Meta learning, NAS
  - how does Scale decreased model work as backbone in object detection?
  - Propose scale permuted network

1. Khetan, Ashish, and Zohar Karnin. "PruneNet: Channel Pruning via Global Importance." arXiv preprint arXiv:2005.11282 (2020).
  - Importance score: Variance of input layer after filtering
  - New regularization scheme.

1. Chen, Mark, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, Prafulla Dhariwal, David Luan, and Ilya Sutskever. "Generative Pretraining from Pixels."
  - Image -> low resolution -> 1D transform -> Think like BERT or AR problem
  - Pretrain (BERT/ AR) -> FineTune (classification)

1. Airoldi, Edoardo M., David M. Blei, Stephen E. Fienberg, and Eric P. Xing. "Mixed membership stochastic blockmodels." Journal of machine learning research 9, no. Sep (2008): 1981-2014.

1. Tang, Danhang, Saurabh Singh, Philip A. Chou, Christian Hane, Mingsong Dou, Sean Fanello, Jonathan Taylor et al. "Deep Implicit Volume Compression." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1293-1303. 2020.
  - Well, little off chart for me!
  - Learnable compression via encoder Decoder
  - Addresses the need for compressing and streaming UV coordinates
    - Allows Image based compress
    - End-2-end entropy encoding

1. Shuster, Kurt, Samuel Humeau, Hexiang Hu, Antoine Bordes, and Jason Weston. "Engaging image captioning via personality." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 12516-12526. 2019.
  - Dataset for image captioning based on the mood
  - Used sentence and image representation models.
  - Network (Input1 : Image > ResNet152 (Freeze pretrained) > FFNN > 500 (o1), input2 : Personality trait > Linear Layer > 500 (o2), Input3 : Caption > Transformer > FFNN > 500 (o3), [{o1 add o1} dotProd {o3}] > score. )
  - Two models :
    - Retriever model (output any caption)
    - Generative model (conditioned caption output)
    - Image Encoder parts common (o1)

1. Shocher, Assaf, Yossi Gandelsman, Inbar Mosseri, Michal Yarom, Michal Irani, William T. Freeman, and Tali Dekel. "Semantic Pyramid for Image Generation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7457-7466. 2020.
  - architecture is crucial. [Figure 3]
  - Loss function {adversarial loss + reconstruction loss(masked area) + diversity loss}
  - Different application [especially image re-painting]
  - inverting deep features and deep features for image manipulation
  - Hierarchical generator setup.

1. Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." In International conference on machine learning, pp. 1180-1189. 2015.
  - Simple and great Idea [eq: 1,2,3].
    - Input to features (*f*), features to task (*y, loss<sub>y</sub>*), features to domain classifier (*d, loss<sub>d</sub>*).
    - *f* tries to minimize the *loss<sub>y</sub>* and maximize the *loss<sub>d</sub>*, and *d, y* tries to minimize their corresponding losses.
  - Final task need to be related (y) but the source may be different (f tries to find common ground).
  - Gradient Reversal layer to implement via SGD.
  - Reduces h delta h distance [eq 13]

1. Gururangan, Suchin, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." arXiv preprint arXiv:2004.10964 (2020).
  - Multiphase pretraining (PT) for NLP- (Domain and task specific)
  - Pretraining dataset selection strategy
  - Key motivation: RoBERTa
  - Experimented with DAPT, TAPT and DAPT+TAPT

1. Xie, Qizhe, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. "Self-training with noisy student improves imagenet classification." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10687-10698. 2020.
  - Interesting way to improve the Classifier
  - (labeled data) -> Build classifier (T) -> (predict unlabeled data) -> Train Student using both labeled + model predicted unlabeled data. Repeat.. [algo 1]
  - Introduce noise for both T and S.
    - Data noise, model noise (dropout)

1. Park, Taesung, Alexei A. Efros, Richard Zhang, and Jun-Yan Zhu. "Contrastive Learning for Unpaired Image-to-Image Translation." arXiv preprint arXiv:2007.15651 (2020).
  - Contrastive loss (Same patch of input - output are +ve and rest of the patches are -ve example)
  - Trains the encoder parts more! (Fig 1, 2) ; Decoders train only on adversarial losses.
  - Contribution in loss (SimCLR) kinda motivation

1. Berthelot, David, Peyman Milanfar, and Ian Goodfellow. "Creating High Resolution Images with a Latent Adversarial Generator." arXiv preprint arXiv:2003.02365 (2020).
  - Close to super-resolution problem [but different as loss is perception loss.]
    - When noise 0 we want the original image.
  - LAG From {R<sup>y</sup>(low resolution) x R<sup>z</sup>(noise)} to {R<sup>x</sup> (high resolution sample of natural image)}.
  - Gradient penalty loss (To ascertain 1-Lipschitz)

1. Grill, Jean-Bastien, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch et al. "Bootstrap your own latent: A new approach to self-supervised learning." arXiv preprint arXiv:2006.07733 (2020).
  - Unsupervised Representation learning in a discriminative method.
  - Alternative of contrastive learning methods (as CL depends on batch size, image augmentation method, memory bank, resilient). [No negative examples]
  - Online and Target network. [Augmented image output in online network should be close to main image in target network.] What about all zeros! (Empirically slow moving average helps to avoid that)
  - Motivation [section 3 method]
  - All about architecture. [encoder, projection, predictor and loss function]

1. Caron, Mathilde, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. "Deep clustering for unsupervised learning of visual features." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 132-149. 2018.
  - Cluster Deep features and make them pseudo labels. [fig 1]
  - Cluster (k-means) for training CNN [Avoid trivial solution of all zeros!]
  - Motivation from Unsupervised feature learning, self-supervised learning, generative model
  - [More](https://github.com/facebookresearch/deepcluster)

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

1. Chen, Liang-Chieh, George Papandreou, Florian Schroff, and Hartwig Adam. "Rethinking atrous convolution for semantic image segmentation." arXiv preprint arXiv:1706.05587 (2017).
  - Multi scale atrous CNN for Semantic image segmentation & modified ASPP
  - Enlarge Field-of-View
  - Removed DenseCRF
  - Dataset: PASCAL, VOC 2012
  - All 3x3 with different rate
  - Related works: Context Module!

1. Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).
  - Aim the undertrained issue of BERT + Some improvement both in dataset and hyperparameters [Pretraining]
  - BERT decision choice + novel dataset + better accuracy
  - Different training strategy [section 3]
    -  Dynamic masking while training
    - Full sequence NSP loss
    - Large mini batch
    - Bye level BPE
    - {Combination of all this is RoBERTa
    - [Full](https://docs.google.com/presentation/d/1xva3OiNm5qRG82stmkKpyGIIEt8KRZXWW04zIOkivUs/edit?usp=sharing)

1. Radford, Alec, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. "Language models are unsupervised multitask learners." OpenAI Blog 1, no. 8 (2019): 9.
  - Under construction

1. McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. "Communication-efficient learning of deep networks from decentralized data." In Artificial Intelligence and Statistics, pp. 1273-1282. PMLR, 2017.
  - Coined and proposed the Federated Learning methods
  - Decouple model training and data [privacy and security]
  - Unbalanced and IID data works best [experimented on 4 dataset]

1. Ravfogel, Shauli, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. "Null it out: Guarding protected attributes by iterative nullspace projection." arXiv preprint arXiv:2004.07667 (2020).
  - Under construction

1. Roy, Aurko, Mohammad Saffar, Ashish Vaswani, and David Grangier. "Efficient content-based sparse attention with routing transformers." arXiv preprint arXiv:2003.05997 (2020).
  - Under construction

1. Caron, Mathilde, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. "Unsupervised learning of visual features by contrasting cluster assignments." arXiv preprint arXiv:2006.09882 (2020).
  - SwAV (online algorithm) [swapped assignments between multiple vies of same image]
  - Contrastive learning, clustering
  - Predict cluster from different representation, memory efficiency!
  - 'code' consistency between image and its transformation {target}
  - online code computation
  - Features and codes are learnt online
  - multi-crop: Smaller image with multiple views
  - validation: ImageNet linear evaluation protocol
  - Interested related work section
  - Key motivation: Contrastive instance learning
  - Partition constraint to avoid trivial solution

1. Yang, Zhilin, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R. Salakhutdinov, and Quoc V. Le. "Xlnet: Generalized autoregressive pretraining for language understanding." In Advances in neural information processing systems, pp. 5753-5763. 2019.
  - under construction

1. Asano, Yuki M., Mandela Patrick, Christian Rupprecht, and Andrea Vedaldi. "Labelling unlabelled videos from scratch with multi-modal self-supervision." arXiv preprint arXiv:2006.13662 (2020).
  - Under consideration

1. Patrick, Mandela, Yuki M. Asano, Ruth Fong, João F. Henriques, Geoffrey Zweig, and Andrea Vedaldi. "Multi-modal self-supervision from generalized data transformations." arXiv preprint arXiv:2003.04298 (2020).

1. Khosla, Prannay, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. "Supervised contrastive learning." arXiv preprint arXiv:2004.11362 (2020).
  -

1. Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality reduction by learning an invariant mapping." In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06), vol. 2, pp. 1735-1742. IEEE, 2006.

1. Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." In ICML deep learning workshop, vol. 2. 2015.

1. Hu, Baotian, Zhengdong Lu, Hang Li, and Qingcai Chen. "Convolutional neural network architectures for matching natural language sentences." In Advances in neural information processing systems, pp. 2042-2050. 2014.
  - Matching network

1. Lee, Kwot Sin, Ngoc-Trung Tran, and Ngai-Man Cheung. "InfoMax-GAN: Improved Adversarial Image Generation via Information Maximization and Contrastive Learning." arXiv preprint arXiv:2007.04589 (2020).

1. Chuang, Ching-Yao, Joshua Robinson, Lin Yen-Chen, Antonio Torralba, and Stefanie Jegelka. "Debiased contrastive learning." arXiv preprint arXiv:2007.00224 (2020).

1. Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. "Representation learning with contrastive predictive coding." arXiv preprint arXiv:1807.03748 (2018).

1. He, Kaiming, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. "Momentum contrast for unsupervised visual representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729-9738. 2020.
  - Dynamic dictionary with MA encoder
  - (query) encoder and (key) momentum encoder.
  - The update of key encoder in a momentum fashion
      - Query updated by back propagation
  - Algorithm is the Core
  - key and query match but the queue would not match!
  - Momentum parametric dependencies
  - Start with the key and query encoder as the Same
    - key updates slowly, query updates with SGD.

1. Kolesnikov, Alexander, Xiaohua Zhai, and Lucas Beyer. "Revisiting self-supervised visual representation learning." In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 1920-1929. 2019.

1. Wang, Xiaolong, and Abhinav Gupta. "Unsupervised learning of visual representations using videos." In Proceedings of the IEEE international conference on computer vision, pp. 2794-2802. 2015.

1. Misra, Ishan, C. Lawrence Zitnick, and Martial Hebert. "Shuffle and learn: unsupervised learning using temporal order verification." In European Conference on Computer Vision, pp. 527-544. Springer, Cham, 2016.

1. Fernando, Basura, Hakan Bilen, Efstratios Gavves, and Stephen Gould. "Self-supervised video representation learning with odd-one-out networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3636-3645. 2017.

1. Wu, Zhirong, Yuanjun Xiong, Stella X. Yu, and Dahua Lin. "Unsupervised feature learning via non-parametric instance discrimination." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3733-3742. 2018.
  - non-parametric classifier via feature representation
  - Memory bank stores instance features (used for kNN classifier)
    - Dimention reduction
  - Experiments
    - obj detect and image classification
  - connect to
    - selfsupervised learning (related works) and Metric learning (unsupervised fashion)
    - NCE (to tackle class numbers) - [great idea, just contrast with everything else in E we get the classifier]
  - instance-level discrimination, non-parametric classifier.
    - compared with known example (non-param.)
  -  interesting setup section 3
    - representation -> class (image itself) (compare with instance) -> loss function (plays the key role to distinguish)
    - NCE from memory bank
    - Monte carlo sampling to get the all contrastive normalizing value for denominator
    - proximal parameter to ensure the smoothness for the representations 

1. Sermanet, Pierre, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. "Time-contrastive networks: Self-supervised learning from video." In 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 1134-1141. IEEE, 2018.
  - Multi view point [same times are same, different time frames are different]
    - Regardless of the viewpoint [same time same thing , same representation]
    - Considered images
    - Representation is the reward
    - TCN - a embedding
  - imitation learning
  - PILQR for RL parts
  - Huber-style loss

1. Liu, Sicong, Junzhao Du, Anshumali Shrivastava, and Lin Zhong. "Privacy Adversarial Network: Representation Learning for Mobile Data Privacy." Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 3, no. 4 (2019): 1-18.
  - presented in my course work instructed by my supervisor Dr. Nirmalya Roy
  - [link](https://docs.google.com/presentation/d/1OF7Y6yoIAuLVQ_OtV5kdCEXzCp3MRAzusg1o8k6kTvo/edit?usp=sharing)

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
