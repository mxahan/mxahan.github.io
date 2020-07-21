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

1. Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).
  - Knowledge distillation
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

1. Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." In Advances in neural information processing systems, pp. 5998-6008. 2017.

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

1. Wu, Zhirong, Yuanjun Xiong, Stella X. Yu, and Dahua Lin. "Unsupervised feature learning via non-parametric instance discrimination." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3733-3742. 2018.

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

1. Shocher, Assaf, Yossi Gandelsman, Inbar Mosseri, Michal Yarom, Michal Irani, William T. Freeman, and Tali Dekel. "Semantic Pyramid for Image Generation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7457-7466. 2020.

1. Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." In International conference on machine learning, pp. 1180-1189. 2015.
