# Introduction

This blog contains regularly updated paperlist of my personal interest with minimal summary.

Any suggestion regarding some new papers are highly appreciated. For some of the detail paper read please refer to the Paper summary section.

We can't worry about everything and nothing. We must stay focused and broad at the same time. Very true! I know the Feynman's thoughts on knowing name, which means nothing, it's just the entrance to the grand scheme of the world. however, the papers (many papers!) are as follows

1. Fernandes, Patrick, Miltiadis Allamanis, and Marc Brockschmidt. "Structured neural summarization." arXiv preprint arXiv:1811.01824 (2018).

1. Du, Yilun, and Igor Mordatch. "Implicit Generation and Modeling with Energy Based Models." In Advances in Neural Information Processing Systems, pp. 3603-3613. 2019.

1. Grathwohl, Will, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, and Kevin Swersky. "Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One." arXiv preprint arXiv:1912.03263 (2019).

1. Kingma, Durk P., and Prafulla Dhariwal. "Glow: Generative flow with invertible 1x1 convolutions." In Advances in Neural Information Processing Systems, pp. 10215-10224. 2018.
  - Actnorm
  - invertible 1x1 convolution
  - affine coupling layers

1. Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016).

1. Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013).
  - word2vec
  - Comparison with others [link](https://medium.com/@kashyapkathrani/all-about-embeddings-829c8ff0bf5b)

1. Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. "Distributed representations of words and phrases and their compositionality." In Advances in neural information processing systems, pp. 3111-3119. 2013.
  - non-contextual embedding
  - Word2vec extensions
  - Skip gram model and its extensions: skip gram tries to maximize the log probaiblites <img src="https://latex.codecogs.com/gif.latex?1/T\sum_1^T\sum_{-c<=j<=c, j\neq 0}\log p(w_{t+j}|w_t)">; the probabilities are defined by softmax of the vector represenation of the words. The issue is computation of derivatives of log.
  - Hierarchical Softmax: Tree structure from root to the word [link]((https://towardsdatascience.com/hierarchical-softmax-and-negative-sampling-short-notes-worth-telling-2672010dbe08) )
  - Negative Sampling: Adds with target and penalize for noise words. Noise Contrastive Estimation.
  - Subsampling of Frequent words.
  - input output word embedding [link](https://stats.stackexchange.com/questions/263284/what-exactly-are-input-and-output-word-representations)
   - Word representations [link](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795)
   - Trained on google dataset (billion!)
   - Phase considered by data driven approach – Bigram model
      - ind score (> th then phase!)
   - Reports for the phase skip gram model
   - Learning Explanation [link](https://arxiv.org/pdf/1411.2738.pdf)

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

1. Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).

1. Graves, Alex, Greg Wayne, and Ivo Danihelka. "Neural turing machines." arXiv preprint arXiv:1410.5401 (2014).
  - External memory source coupling for attention  (analogous to turing machine )
    - How to show it? what's the background? Experiments? How much they differ and align? Whats the perspective?
  - copying, storing and associative recall
  - Extension of RNN - NTM (differentiable unlike Turing Machine)
  - Two components: controller and memory with i/o for external interaction
    - Attending to memory (attention) [sharp or distributed attention]
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Zhao, Junbo, Michael Mathieu, and Yann LeCun. "Energy-based generative adversarial network." arXiv preprint arXiv:1609.03126 (2016).li
  - D: Assign low energy to data distribution (not normalized probabilities)
  - G: Sample data from the low energy by a parameterized function.
  - Convergence proof using hinge loss.

1. ani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." In Advances in neural information processing systems, pp. 5998-6008. 2017.
  - Under construction

1. Radford, Alec, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. "Improving language understanding by generative pre-training." (2018): 12.
  - GPT paper

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
  - Interesting [link](http://jalammar.github.io/illustrated-bert/)
  - Self-attention, query-key value - [link](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
  - [you tube](https://www.youtube.com/watch?v=rBCqOTEfxvg)
  - [another nice explanation](https://medium.com/swlh/what-exactly-is-happening-inside-the-transformer-b7f713d7aded)
  - All layers (q, k, v, decoder and encoders) are subjected to backpropagation
  - full paper breakdown [link](https://towardsml.com/2019/09/17/bert-explained-a-complete-guide-with-theory-and-tutorial/)

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

1. Berthelot, David, Peyman Milanfar, and Ian Goodfellow. "Creating High Resolution Images with a Latent Adversarial Generator." arXiv preprint arXiv:2003.02365 (2020).
  - Close to super-resolution problem [but different as loss is perception loss.]
    - When noise 0 we want the original image.
  - LAG From {R<sup>y</sup>(low resolution) x R<sup>z</sup>(noise)} to {R<sup>x</sup> (high resolution sample of natural image)}.
  - Gradient penalty loss (To ascertain 1-Lipschitz)

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

1. Yang, Zhilin, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R. Salakhutdinov, and Quoc V. Le. "Xlnet: Generalized autoregressive pretraining for language understanding." In Advances in neural information processing systems, pp. 5753-5763. 2019.
  - under construction

1. Hu, Baotian, Zhengdong Lu, Hang Li, and Qingcai Chen. "Convolutional neural network architectures for matching natural language sentences." In Advances in neural information processing systems, pp. 2042-2050. 2014.
  - Matching network

1. Lee, Kwot Sin, Ngoc-Trung Tran, and Ngai-Man Cheung. "InfoMax-GAN: Improved Adversarial Image Generation via Information Maximization and Contrastive Learning." arXiv preprint arXiv:2007.04589 (2020).

1. Liu, Sicong, Junzhao Du, Anshumali Shrivastava, and Lin Zhong. "Privacy Adversarial Network: Representation Learning for Mobile Data Privacy." Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 3, no. 4 (2019): 1-18.
  - presented in my course work instructed by my supervisor Dr. Nirmalya Roy
  - [link](https://docs.google.com/presentation/d/1OF7Y6yoIAuLVQ_OtV5kdCEXzCp3MRAzusg1o8k6kTvo/edit?usp=sharing)

1. Wang, Feng, Xiang Xiang, Jian Cheng, and Alan Loddon Yuille. "Normface: L2 hypersphere embedding for face verification." In Proceedings of the 25th ACM international conference on Multimedia, pp. 1041-1049. 2017.
  - Training using normalization features.
    - modification of softmax and optimize cosine losses
    - Metric learning
  - Research gap of necessity of normalization
  - Four contributions?
    - why cosine doesn't converge? buy normalized dot succeed.
    - different loss option explore? why!!

1. Gao, Chen, Ayush Saraf, Jia-Bin Huang, and Johannes Kopf. "Flow-edge Guided Video Completion." In European Conference on Computer Vision, pp. 713-729. Springer, Cham, 2020.
  - Gradient domain processing
  - Three steps
    - Flow-Completion
    - Temporal propagation
    - Fusion
  - Research Gap: memory for 3D, flow edges in flow based methods.
  - Contribution
    - Piecewise-smooth flow completion
    - non-local flow for obscure objects
    - Gradient domain operation (gradient of color through NN)
  - Architecture is everything
  - DAVIS dataset
  - Poisson reconstruction

1. Tan, Hao, and Mohit Bansal. "Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision." arXiv preprint arXiv:2010.06775 (2020).
  - Sentence to hidden features to probability distribution.  
  - Contextual visual representation
  - LM architecture with additional voken objective (BERT objective + voken classification)
  - Need revisit!

1. Peters, Matthew E., Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. "Deep contextualized word representations." arXiv preprint arXiv:1802.05365 (2018).

1. Shen, Yanyao, Hyokun Yun, Zachary C. Lipton, Yakov Kronrod, and Animashree Anandkumar. "Deep active learning for named entity recognition." arXiv preprint arXiv:1707.05928 (2017).

1. Gal, Yarin, Riashat Islam, and Zoubin Ghahramani. "Deep bayesian active learning with image data." arXiv preprint arXiv:1703.02910 (2017).

1. Ganin, Yaroslav, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. "Domain-adversarial training of neural networks." The Journal of Machine Learning Research 17, no. 1 (2016): 2096-2030.   

1. Karras, Tero, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. "Training generative adversarial networks with limited data." Advances in Neural Information Processing Systems 33 (2020).
  - GAN leaking in case of small dataset augmentation!!Learns augmented distribution
  - prevent the leaking!
  - nonleaking operation -invertible trasnformation (2.2: Point made)
  - augmentation Scheme
  - balanced consistency regularization! (old approach)
  - stochastic discriminator augmentation
    - Figure 2 (Whats the benefit!!)
    - fig2(b): Generator output always go through Augmentation (with p [most significant param]) before hitting D
  - Adaptive discriminator augmentation (section 3)
    - Point: What if the D learns nothing for anyone! then r = 0 [Eq1], right? No! Oh got it! [solved :)] 0 (real) < D < 1(generated)

1. Izacard, Gautier, and Edouard Grave. "Distilling Knowledge from Reader to Retriever for Question Answering." arXiv preprint arXiv:2012.04584 (2020).

1. Szegedy, Christian, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. "Rethinking the inception architecture for computer vision." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818-2826. 2016.

1. Gururangan, Suchin, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." arXiv preprint arXiv:2004.10964 (2020).

1. Goodfellow, Ian, Honglak Lee, Quoc Le, Andrew Saxe, and Andrew Ng. "Measuring invariances in deep networks." Advances in neural information processing systems 22 (2009): 646-654.
  - Evaluate invariant features
    - Proposed some empirical metrics to measure the degree of invariance
  - Experiment with stacked auto-encoder and CNN
  - Proposed invariant set and invariance measurement metrics
  - Empirical answer of why DL is useful - increasing depth increase invariance in representations
  - Network architecture
    - Stacked auto-encoder
    - CNN Deep belief network
  - Invariance measurement
    - Finding neurons firing under invariance
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Gao, Ruohan, and Kristen Grauman. "VisualVoice: Audio-Visual Speech Separation with Cross-Modal Consistency." arXiv preprint arXiv:2101.03149 (2021).
  -

1. Brock, Andrew, Soham De, Samuel L. Smith, and Karen Simonyan. "High-Performance Large-Scale Image Recognition Without Normalization." arXiv preprint arXiv:2102.06171 (2021).
  - Proposes alternative to BN [BN suffers instabilities] - Normalization freee models
    - Adaptive gradient clipping (AGC)
    - Experiment: Normalization free ResNet
  - Section 3: reasons and alternative to BN
  - Section 4: This papers key contribution (AGC)
    - Hypothesis: 1<sup>st</sup> paragraph: accelerating converge in poorly conditioned loss function for large Batch Size.
    - Key point in equation 3 (adaptive)

1. Smirnov, Evgeny, Aleksandr Melnikov, Andrei Oleinik, Elizaveta Ivanova, Ilya Kalinovskiy, and Eugene Luckyanets. "Hard example mining with auxiliary embeddings." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 37-46. 2018.
  - Deep embedding learning and hard example mining!!
    - Proposes auxiliary embedding [should solve the problem of mini-batch level, single class random selection] for hard mining [claims to be novel]: mini-batch with large number of hard exmples
    - Multiple ways to create auxiliary embedding!
  - Related works: mini-batch level hard exmaple mining, hard class mining
  - Figure 2: selection of hard example! auxiliary embedding are final layers

1. Gebru, Timnit, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. "Datasheets for datasets." arXiv preprint arXiv:1803.09010 (2018).
  - Nice guide for dataset documentation.
  - Propose to contain: Motivation, composition, collection process, recommended users, ... Cleaning, labeling, distribution, and maintenance.
  - Aim: Provide better Communication, standardization.
  - Clear intend of dataset creator and consumer.
  - Question and WorkFlow:
    - Motivation: why?
    - Composition: what modalities? Data Volume? Data variability? Instance elements? missing information, Labels? Any recommendation or comments on dataset? Self-content? confidential information? Data sample groups?
    - Collection Process: Description of collecting each instance, Sensors and settings, time frame, ethical review, any comments
    - Preprocessing: Cleaning and labeling, Meta-data, Any software?
    - Uses: tasks, order to use? application and not-applicable cases
    - Distribution: To whom, how, Any IP or copyright?
    - maintenance: supporter, owner, how to update, version, extension?

1. Foret, Pierre, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. "Sharpness-Aware Minimization for Efficiently Improving Generalization." arXiv preprint arXiv:2010.01412 (2020).
  - Aim to reduce the loss value and **loss sharpness** (motivated by geometry of loss space and generalization)
  - TP about increasing generalization (how to claim such results, what will be the evidence): Performance over the noisy dataset!
  - TP: Sharpness-Aware Minimization (SAM) : Aims to find params with uniform low-loss in their neighborhood. *still works with SGD*
  - Why only Cross-Entropy! Not sufficient everytime!
  - Efficient, scalable and effective method!! (what results it requires?)
    - Minimize loss value and loss sharpness!
    - Model generalization: Experiment with (CIFAR, ImageNet, SVHN, MNIST, ....)
    - coins m-sharpness terms
  - Reducing the gap between training loss and the population loss (generalization)
  - Algorithm 1 summarizes the ways to train the network!
  - Detailed experiment and discussion section
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Madry, Aleksander, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).
  - study a set of approaches [the adversarial robustness of neural networks through the lens of robust optimization]
    - (unifies earlier works).
    - provides security against first order adversarial attack.
  - Projected gradient descent.

1. Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).

# Self-Supervised Learning

1. Tosh, Christopher, Akshay Krishnamurthy, and Daniel Hsu. "Contrastive estimation reveals topic posterior information to linear models." Journal of Machine Learning Research 22, no. 281 (2021): 1-31.

1. Tosh, Christopher, Akshay Krishnamurthy, and Daniel Hsu. "Contrastive learning, multi-view redundancy, and linear models." In Algorithmic Learning Theory, pp. 1179-1206. PMLR, 2021.
  - TP: Theory for contrastive learning in the multi-view setting, where two views of each datum are available.
  - learned representations are nearly optimal on downstream prediction tasks whenever the two views provide **redundant information** about the label.
    - what is redundant information!! := whenever the best linear prediction of the label on each individual view is nearly as good as the best linear prediction of the label when both views are used together.

1. Teng, Jiaye, Weiran Huang, and Haowei He. "Can pretext-based self-supervised learning be boosted by downstream data? a theoretical analysis." arXiv preprint arXiv:2103.03568 (2021).
  - Highly theoretical paper.
  - whether we can make the CI (conditional independence) condition hold by using downstream data to refine the unlabeled data to boost self-supervised learning [*Not always true though it seems intuitive*]
    - Shows some result to prove the counter intuitive results (Hence focusing the importance of the conditional independence)
    - Can we make the CI condition hold with the help of downstream data to boost self-supervised learning? (as CI rearly helds)
    - Validate self-supervised approach (not to used fine-tune data during pretraining)

1. Lee, Jason D., Qi Lei, Nikunj Saunshi, and Jiacheng Zhuo. "Predicting what you already know helps: Provable self-supervised learning." arXiv preprint arXiv:2008.01064 (2020).
  - Highly theoretical paper.
  - TP:  is to investigate the statistical connections between the random variables of input features and downstream labels
    - Two important notion for the tasks: i) Expressivity (does the ssl good enough) ii) Sample complexity (reduce the complexity of sampling)
  - TP: analysis on the reconstruction based SSL
  - section 3 describes the paper summary. (connected to simsiam: Section 6
  - Discusses about conditional independence ([CI](!https://www.probabilitycourse.com/chapter1/1_4_4_conditional_independence.php)) condition of the samples w.r.t the labels

1. Ren, Xuanchi, Tao Yang, Yuwang Wang, and Wenjun Zeng. "Do Generative Models Know Disentanglement? Contrastive Learning is All You Need." arXiv preprint arXiv:2102.10543 (2021).

1. Jing, Li, Pascal Vincent, Yann LeCun, and Yuandong Tian. "Understanding Dimensional Collapse in Contrastive Self-supervised Learning." arXiv preprint arXiv:2110.09348 (2021).
  - Discusses about dimension collapse ( the embedding vectors end up spanning a lower-dimensional subspace instead of the entire available embedding space.)
  - key findins.
    - along the feature direction where data augmentation variance is larger than the data distribution variance, the weight collapses. (strong augmentation along feature dimensions)
    - even if the covariance of data augmentation has a smaller magnitude than the data variance along all dimensions, the weight will still collapse due to the interplay of weight matrices at different layers known as [implicit regularization](http://old.ins.sjtu.edu.cn/files/paper/20200722191948_slide.pdf). (implicit regularization driving models toward low-rank solutions)
  - looks into the covariance matrix (their eigen decomposition; some eigenvalues become zeros) to measure dimensional collapse
  - Theoretically justify the importance of the projection layers.
  - TP: proposed DirectCLR (uses direct Optimization on the representation layer by selecting a subsection of the layer for loss calculation. remove the requirement of trainable projection. )

1. Dwibedi, Debidatta, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. "Counting out time: Class agnostic video repetition counting in the wild." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10387-10396. 2020.
  - Countix Dataset for video repeatation count. (part of Kinetics dataset)
    - annotated with segments of repeated actions and corresponding counts.
  - Per-frame embedding and similarity!! (RepNet)
  - Compared with benchmark:  PERTUBE and QUVA
  - Not really a self-supervised set up, rather a multitask setup.
  - Propose to create synthetic dataset by frame repeatation and reversal! Camera motion augmentation (Augmentation)

1. Hendrycks, Dan, Mantas Mazeika, Saurav Kadavath, and Dawn Song. "Using self-supervised learning can improve model robustness and uncertainty." arXiv preprint arXiv:1906.12340 (2019).
  - TP: self-supervision can benefit robustness in a variety of ways, including robustness to adversarial examples, label corruption, and common input corruptions (how is this new!!)
    - Interesting problem and experiment setup for each of the problems
  - Besides a collection of techniques allowing models to catch up to full supervision, SSL is used two in conjunction of providing strong regularization that improves robustness and uncertainty estimation

1. Sordoni, Alessandro, Nouha Dziri, Hannes Schulz, Geoff Gordon, Philip Bachman, and Remi Tachet Des Combes. "Decomposed Mutual Information Estimation for Contrastive Representation Learning." In International Conference on Machine Learning, pp. 9859-9869. PMLR, 2021.

1. Ryali, Chaitanya K., David J. Schwab, and Ari S. Morcos. "Leveraging background augmentations to encourage semantic focus in self-supervised contrastive learning." arXiv preprint arXiv:2103.12719 (2021).
  - This Paper: Image augmentation regarding the subject and background relationship - "background Augmentation" [sampling method for CL]
    - How they separate the subject background in the first places!! What prior knowledge!!
    - May use different existing methods!!
  - Augmentation Scheme: Another data engineering
    - Used with methods like BYOL, SwAV, MoCo to push SOTA forward
    - Figure 1: Shows all

1. Huynh, Tri, Simon Kornblith, Matthew R. Walter, Michael Maire, and Maryam Khademi. "Boosting Contrastive Self-Supervised Learning with False Negative Cancellation." arXiv preprint arXiv:2011.11765 (2020).
  - False negative Problem!! (detail analysis) [sampling method for CL]
    - Aim: Boosting results
  - Methods to Mitigate false negative impacts (how? what? how much impact! significant means?? what are other methods?)
  - Hypothesis: Randomly taken negative samples (leaked negative)
  - Overview
    - identify false negative (how?): Finding potential False negative sample [3.2.3]
    - Then false **negative elimination** and **false negative attraction**
  - Contributions
    - applicable on top of existing cont. learning

1. Denton, Emily L. "Unsupervised learning of disentangled representations from video." In Advances in neural information processing systems, pp. 4414-4423. 2017.
  - Encoder-Decoder set up for the disentangled [disentanglement representation]
  - Hypothesis: Content (time invariant) and Pose (time variant)
  - Two Encoders for the pose and content; Concatenate the output for single Decoder
  - Introduce adversarial loss
  - Video generation conditioned on context, and pose modeling via LSTM.

1. Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. "Representation learning with contrastive predictive coding." arXiv preprint arXiv:1807.03748 (2018).
  - Predicting the future [self-supervised task design]
    - derive the concept of context vector (from earlier representation)
      - use the context vector for future representation prediction.
  - TP: Great works with some foundation of CL
  - probabilistic (AR) contrastive loss!!
    - in latent space
  - Experiments on the speech, image, text and RL
  - CPC (3 things)
    - compression, autoregressive and NCE
  - Energy based like setup
  - Figure 4: about what they did!
  - Aka- InfoNCE
  - [more notes](https://github.com/mxahan/PDFS_notes/blob/master/cpc_2017.pdf)

1. He, Kaiming, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. "Momentum contrast for unsupervised visual representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729-9738. 2020.
  - Dynamic dictionary with MA encoder [Algorithmic] Contribution
  - (query) encoder and (key) momentum encoder.
  - The update of key encoder in a momentum fashion
      - Query updated by back propagation
  - Algorithm [1] is the Core: A combination of **end-end & the Memory bank**
  - key and query match but the queue would not match!
  - Momentum parametric dependencies
  - Start with the key and query encoder as the Same
    - key updates slowly, query updates with SGD.

1. Kolesnikov, Alexander, Xiaohua Zhai, and Lucas Beyer. "Revisiting self-supervised visual representation learning." In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 1920-1929. 2019.
  - Insight about the network used for learning [experimentation]
  - Challenges the choice of different CNNs as network for vision tasks.
  - Experimentation with different architectures [ResNet, RevNet, VGG] and their widths and depths.
  - key findings : hat (1) lessons from architecture design in the fully supervised setting do not necessarily translate to the self-supervised setting; (2) contrary to previously popular architectures like AlexNet, in residual architectures, the final prelogits layer consistently results in the best performance; (3) the widening factor of CNNs has a drastic effect on performance of self-supervised techniques and (4) SGD training of linear logistic regression may require very long time to converge
  - pretext tasks for self-supervised learning should not be considered in isolation, but in conjunction with underlying architectures.

1. Wang, Xiaolong, and Abhinav Gupta. "Unsupervised learning of visual representations using videos." In Proceedings of the IEEE international conference on computer vision, pp. 2794-2802. 2015.
  - Visual tracking provides the supervision!!! [sampling method for CL]
  - Siamese-triplet networks: energy based max-margin loss
  - Experiments: VOC 2012 dataset (100k Unlabeled videos)
  - interesting loss functions (self note: please update the pdf files)

1. Misra, Ishan, C. Lawrence Zitnick, and Martial Hebert. "Shuffle and learn: unsupervised learning using temporal order verification." In European Conference on Computer Vision, pp. 527-544. Springer, Cham, 2016.
  - Pretext Task: a sequence of frames from a video is in the correct temporal order (figure 1) [sampling method for CL]
    - Capture temporary variations
    - Fusion and classification [not the CL directly]
  - experiment Net: CNN Based network, data: UCF101, HMDB51 & FLIC, MPII (pose Estimation)
  - self note: There's more.

1. Fernando, Basura, Hakan Bilen, Efstratios Gavves, and Stephen Gould. "Self-supervised video representation learning with odd-one-out networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3636-3645. 2017.
  - Pretext tasks: Finding the odd-one (O3N) video using fusion. [sampling method for CL]
    - Temporal odd one! Target: Regression task
  - Network: CNN with fusion methods.
  - Experiments: HMDB51, UCF101

1. Wu, Zhirong, Yuanjun Xiong, Stella X. Yu, and Dahua Lin. "Unsupervised feature learning via non-parametric instance discrimination." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3733-3742. 2018.
  - non-parametric classifier via feature representation **(Memory Bank)**
  - Memory bank stores instance features (used for kNN classifier)
    - Dimention reduction: one of the key [algorithmic] contribution
  - Experiments
    - obj detect and image classification
  - connect to
    - selfsupervised learning (related works) and Metric learning (unsupervised fashion)
    - NCE (to tackle class numbers) - [great idea, just contrast with everything else in E we get the classifier]
  - instance-level discrimination, non-parametric classifier.
    - compared with known example (non-param.)
  - interesting setup section 3
    - representation -> class (image itself) (compare with instance) -> loss function (plays the key role to distinguish)
    - NCE from memory bank
    - Monte carlo sampling to get the all contrastive normalizing value for denominator
    - proximal parameter to ensure the smoothness for the representations {proximal regularization:}

1. Sermanet, Pierre, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. "Time-contrastive networks: Self-supervised learning from video." In 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 1134-1141. IEEE, 2018.
  - Multiple view point [same times are same, different time frames are different], motion blur, viewpoint invariant
    - Regardless of the viewpoint [same time same thing , same representation]
    - Considered images [sampling method for CL]
    - Representation is the reward
    - TCN - a embedding {multitask embedding!}
  - imitation learning
  - PILQR for RL parts
  - Huber-style loss

1. Lu, Jiasen, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. "12-in-1: Multi-task vision and language representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10437-10446. 2020.
  - MTL + Dynamic "stop and go" schedule. [multi-modal representation learning]
  - ViLBERT base architecture.

1. Misra, Ishan, and Laurens van der Maaten. "Self-supervised learning of pretext-invariant representations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6707-6717. 2020.
  - Pre-training method [algorithmic]
  - Pretext learning with transformation invariant + data augmentation invariant
  - See the loss functions  
    - Tries to retain small amount of the transformation properties too !!
  - Use contrastive learning (See NCE)
      - Maximize MI
      - Utilizes extra head on the features.
  - Motivation from predicting video frames
  - Experiment of jigsaw pretext learning
  - Hypothesis: Representation of image and its transformation should be same
  - Use different head for image and jigsaw counterpart of that particular image.
      - Motivation for learning some extra things by different head network
  - Noise Contrastive learning (contrast with other images)
  - As two head so two component of contrastive loss. (One component to dampen memory update.)
  - Implemented on ResNet
  - PIRL

1. Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Split-brain autoencoders: Unsupervised learning by cross-channel prediction." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1058-1067. 2017.
  - Extension of autoencoders to cross channel prediction [algorithmic]
    - Predict one portion to other and vice versa + loss on full reconstruction.
    - Two disjoint auto-encoders.
  - Tried both the regression and classification loss
  - Section 3 sums it up
    - Cross-channel encoders
    - Split-brain autoencoders.
    -

1. Srinivas, Aravind, Michael Laskin, and Pieter Abbeel. "Curl: Contrastive unsupervised representations for reinforcement learning." arXiv preprint arXiv:2004.04136 (2020).

1. Chen, Ting, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. "A simple framework for contrastive learning of visual representations." arXiv preprint arXiv:2002.05709 (2020).
  - Truely simple! (SimCLR) [algorithmic]
  - Two transfers for each image and representation
  - Same origin image should be more similar than the others.
  - Contrastive (negative) examples are from image other than that.
  - A nonlinear projection head followed by the representation helps.

1. Asano, Yuki M., Mandela Patrick, Christian Rupprecht, and Andrea Vedaldi. "Labelling unlabelled videos from scratch with multi-modal self-supervision." arXiv preprint arXiv:2006.13662 (2020).
  - clustering method that allows pseudo-labelling of a video dataset without any human annotations, by leveraging the natural correspondence between the audio and visual modalities
  - [Multi-Modal representation learning]

1. Patrick, Mandela, Yuki M. Asano, Ruth Fong, João F. Henriques, Geoffrey Zweig, and Andrea Vedaldi. "Multi-modal self-supervision from generalized data transformations." arXiv preprint arXiv:2003.04298 (2020).
  - [Multi-Modal representation learning]

1. Khosla, Prannay, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. "Supervised contrastive learning." arXiv preprint arXiv:2004.11362 (2020).
  - [Algorithmic]

1. Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality reduction by learning an invariant mapping." In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06), vol. 2, pp. 1735-1742. IEEE, 2006.
  - Algorithm 1: distance based loss optimization [algorithmic]

1. Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." In ICML deep learning workshop, vol. 2. 2015.
  - Original contrastive approach with two (either similar or dissimilarity) images [algorithmic]

1. Park, Taesung, Alexei A. Efros, Richard Zhang, and Jun-Yan Zhu. "Contrastive Learning for Unpaired Image-to-Image Translation." arXiv preprint arXiv:2007.15651 (2020).
  - Contrastive loss (Same patch of input - output are +ve and rest of the patches are -ve example) [algorithmic]
  - Trains the encoder parts more! (Fig 1, 2) ; Decoders train only on adversarial losses.
  - Contribution in loss (SimCLR) kinda motivation

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

1. Tian, Yonglong, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. "What makes for good views for contrastive learning." arXiv preprint arXiv:2005.10243 (2020).
  - Multi-view in-variance
  - What is invariant?? (shared information between views)
  - balance to share the information we need in view!!
  - Questions
    - Knowing task what will be the view??!
    - generate views to control the MI
  - Maximize task related shared information, minimize nuisance variables. (InfoMin principle)
  - Contributions (4 - method, representation and task-dependencies, ImageNet experimentation)
  - Figure 1: summary.
  - Optimal view encoder.
    - Sufficient (careful notation overloaded! all info there), minimal sufficient (someinfo dropped), optimal representation (4.3)- only task specific information retrieved
  - InfoMin Principle: views should have different background noise else min encoder reduces the nuisance variable info. (proposition 4.1 with constraints.)
  - suggestion: Make contrastive learning hard
  - Figure 2: interesting. [experiment - 4.2]
  - Figure 3: U-shape MI curve.
  - section 6: different views and info sharing.

1. Tian, Yonglong, Dilip Krishnan, and Phillip Isola. "Contrastive representation distillation." arXiv preprint arXiv:1910.10699 (2019).
  - Missed structural knowledge of the teacher network!!
  - Cross modal distillation!!
  - KD -all dimension are independent (intro 1.2)
    - Capture the correlation/higher-order dependencies in the representation (how??).
    - Maximize MI between teacher and student.
  - Three scenario considered [fig 1]
  - KD and representation learning connection (!!)
  - large temperature increases entropy  [look into the equation! easy-pesy]
  - interesting proof and section 3.1 [great!]
  - Student takes query - matches with positive keys from teacher and contrast with negative keys from the teacher network.
  - Equation 20 is cross entropy (stupid notation)
  - Key contribution: New loss function: 3.4 eq21
  - [notes](https://github.com/mxahan/PDFS_notes/blob/master/contrastive_representation_distillation.pdf)

1. Liang, Weixin, James Zou, and Zhou Yu. "Alice: Active learning with contrastive natural language explanations." arXiv preprint arXiv:2009.10259 (2020).
  - Contrastive natural language!!
  - Experiments -  (bird classification and Social relationship classifier!!)
  - key steps
    - run basic Classifier
    - fit multivariate gaussian for all class (embedding!!), and find b pair of classes with lowest JS divergence.
    - contrastive query to machine understandable form (important and critical part!!). [crop the most informative parts and retrain.]
    - neural arch. morphing!! (heuristic and interesting parts) [local, super classifier and attention mechanism!]

1. Ma, Shuang, Zhaoyang Zeng, Daniel McDuff, and Yale Song. "Learning Audio-Visual Representations with Active Contrastive Coding." arXiv preprint arXiv:2009.09805 (2020).

1. Saunshi, Nikunj, Orestis Plevrakis, Sanjeev Arora, Mikhail Khodak, and Hrishikesh Khandeparkar. "A theoretical analysis of contrastive unsupervised representation learning." In International Conference on Machine Learning, pp. 5628-5637. 2019.
  - present a framework for analyzing CL (is there any previous?)
    - introduce latent class!! shows generalization bound.
  - Unsupervised representation learning
  - TP: notion of latent classes (downstream tasks are subset of latent classes), rademacher complexity of the architecture! (limitation of negative sampling), extension!
  - CL gives representation learning with plentiful labeled data! TP asks this question. \
  - Theoretical results to include in the works.

1. Anand, Ankesh, Evan Racah, Sherjil Ozair, Yoshua Bengio, Marc-Alexandre Côté, and R. Devon Hjelm. "Unsupervised state representation learning in atari." arXiv preprint arXiv:1906.08226 (2019).

1. Devon, R. "Representation Learning with Video Deep InfoMax." arXiv preprint arXiv:2007.13278 (2020).
  - DIM: prediction tasks between local and global features.
    - For video (playing with sampling rate of the views)

1. Bromley, Jane, J. W. Bentz, L. Bottou, I. Guyon, Y. LeCun, C. Moore, E. Sackinger, and R. Shah. "Signature Veriﬁcation using a “Siamese” Time Delay Neural Network." Int.]. Pattern Recognit. Artzf Intell 7 (1993).
  - Gold in old
  - Siamese network early application for hand writing

1. Ebbers, Janek, Michael Kuhlmann, and Reinhold Haeb-Umbach. "Adversarial Contrastive Predictive Coding for Unsupervised Learning of Disentangled Representations." arXiv preprint arXiv:2005.12963 (2020).
  - video deep infomax: UCF101 dataset
  - Local and global features:
  - self note: go over this

1. Becker, Suzanna, and Geoffrey E. Hinton. "Self-organizing neural network that discovers surfaces in random-dot stereograms." Nature 355, no. 6356 (1992): 161-163.

1. Oh Song, Hyun, Yu Xiang, Stefanie Jegelka, and Silvio Savarese. "Deep metric learning via lifted structured feature embedding." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4004-4012. 2016.
  - Proposes a different loss function (Equation-3)
    - Non-smooth and requires special data mining
    - Solution: This paper: optimize upper bound of eq3, instead of mining use stochastic approach!!
    - This paper: all pairwise combination in a batch!! O(m<sup>2</sup>)
      - uses mini batch
      - Not-random batch formation: Importance Sampling
      - Hard-negative mining
      - Gradient finding in the algorithm-1, mathematical analysis
  - Discusses one of the fundamental issues with contrastive loss and triplet loss!
    - different batches puts same class in different position
  - Experiment: Amazon Object dataset- multiview .
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. G. W. Taylor, I. Spiro, C. Bregler, and R. Fergus, ‘‘Learning invariance through imitation,’’ in Proc. CVPR, Jun. 2011, pp. 2729–2736, doi:10.1109/CVPR.2011.5995538

1. Zhuang, Chengxu, Alex Lin Zhai, and Daniel Yamins. "Local aggregation for unsupervised learning of visual embeddings." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6002-6012. 2019.

1. Purushwalkam, Senthil, and Abhinav Gupta. "Demystifying contrastive self-supervised learning: Invariances, augmentations and dataset biases." arXiv preprint arXiv:2007.13916 (2020).
  - object detection and classification
  - quantitative experiment to Demystify CL gains! (reason behind success)
    - Observation1: MOCO and PIRL (occlusion invariant)
      - but Fails to capture viewpoint
    - gain from object-centric dataset - imagenet!
    - Propose methods to leverage learn from unstructured video (viewpoint invariant)
  - Utility of systems: How much invarinces the system encodes
  - most contrastive setup - occlusion invariant! what about viewpoint invariant?
  - Related works
    - Pretext tasks
    - Video SSL
    - Understanding SSRL
      - Mutual information
    - This work - Why CL is useful
      - study two aspects: (invariances encoding & role of dataset)
  - Demystifying Contrastive SSL
    - what is good Representation? Utilitarian analysis: how good the downstream task is?
      - What about the insights? and qualitative analysis?
    - Measuring Invariances
      - What invariance do we need? - invariant to all transformation!!
        - Viewpoint change, deformation, illumination, occlusion, category instance  
      - Metrics: Firing representation, global firing rate, local firing rate, target conditioned invariance, representation invariant score.
      - Experimental dataset
        - occlusion (GOR-10K), viewpoint+instance invariance (Pascal3D+)
      - image and video careful augmentation
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers2.pdf)

1. Ermolov, Aleksandr, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. "Whitening for self-supervised representation learning." arXiv preprint arXiv:2007.06346 (2020).
  - New loss function (why? and where it works?)
    - Generalization of the BYOL approach?
    - No negative examples (the scatters are preserved)
  - Whitening operation (scattering effect)

1. Tschannen, Michael, Josip Djolonga, Paul K. Rubenstein, Sylvain Gelly, and Mario Lucic. "On mutual information maximization for representation learning." arXiv preprint arXiv:1907.13625 (2019).

1. Kalantidis, Yannis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, and Diane Larlus. "Hard negative mixing for contrastive learning." arXiv preprint arXiv:2010.01028 (2020).
  - TP (MoCHi): The effect of Hard negatives (how, definition?)
  - TP: feature level mixing for hard negatives (minimal computational overhead, momentum encoder) by synthesizing hard negatives (!!)
  - Related works: Mixup workshop

1. Löwe, Sindy, Peter O'Connor, and Bastiaan S. Veeling. "Putting an end to end-to-end: Gradient-isolated learning of representations." arXiv preprint arXiv:1905.11786 (2019).

1. Tsai, Yao-Hung Hubert, Martin Q. Ma, Muqiao Yang, Han Zhao, Louis-Philippe Morency, and Ruslan Salakhutdinov. "Self-supervised representation learning with relative predictive coding." arXiv preprint arXiv:2103.11275 (2021).
  -

1. Xiong, Yuwen, Mengye Ren, and Raquel Urtasun. "LoCo: Local contrastive representation learning." arXiv preprint arXiv:2008.01342 (2020).

1. Grill, Jean-Bastien, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch et al. "Bootstrap your own latent: A new approach to self-supervised learning." arXiv preprint arXiv:2006.07733 (2020).
  - Unsupervised Representation learning in a discriminative method. (BOYL)
  - Alternative of contrastive learning methods (as CL depends on batch size, image augmentation method, memory bank, resilient). [No negative examples]
  - Online and Target network. [Augmented image output in online network should be close to main image in target network.] What about all zeros! (Empirically slow moving average helps to avoid that)
  - Motivation [section 3 method]
  - similarity constraint between positive keys are also enforced through a prediction problem from an online network to an offline momentum-updated network
  - BYOL tries to match the prediction from an online network to a randomly initialised offline
  network. This iterations lead to better representation than those of the random offline network.
  -By continually improving the offline network through the momentum update, the quality of the representation is bootstrapped from just the random initialised network
  - All about architecture! [encoder, projection, predictor and loss function]
  - Works only with batch normalization - else mode collapse
  - [More criticism](https://generallyintelligent.ai/understanding-self-supervised-contrastive-learning.html)

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

1. Hoffer, Elad, and Nir Ailon. "Deep metric learning using triplet network." In International Workshop on Similarity-Based Pattern Recognition, pp. 84-92. Springer, Cham, 2015.
  - Triplet networks
  - Experimented on the MNIST dataset.

1. Tian, Yonglong, Dilip Krishnan, and Phillip Isola. "Contrastive multiview coding." arXiv preprint arXiv:1906.05849 (2019).
  - Find the invariant representation
  - Multiple view of objects (image) (CMC) - multisensor view or same object!!]
    - Same object but different sensors (positive keys)
    - Different object same sensors (negative keys)
    - Experiment: ImageNEt, STL-10, two views, DIV2K cropped images
  - Positive pairs (augmentation)
  - Follow-up of Contrastive Predictive coding (no RNN but more generalized)
  - Compared with baseline: cross-view prediction!!
  - Interesting math: Section 3 and experiment 4
    - Mutual information lower bound Log(k = negative samples)- L<sub>contrastive</sub>
    -  Memory bank implementation

1. Hjelm, R. Devon, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. "Learning deep representations by mutual information estimation and maximization." arXiv preprint arXiv:1808.06670 (2018).
  - locality of input knowledge and match prior distribution adversarially (DeepInfoMax)
    - Maximize input and output MI
  - Experimented on Images
    - Compared with VAE, BiGAN, CPC
  - Evaluate represenation by Neural Dependency Measures (NDM)
  - Global features (Anchor, Query) and Local features of the query (+), local feature map of random images (-)
  - [personal note](https://github.com/mxahan/PDFS_notes/blob/master/deepinfomax_paper.pdf)

1. Wang, T. and Isola, P., 2020. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. arXiv preprint arXiv:2005.10242.
  - How to contraint on these and they perform better? weighted loss
    - Alignment (how close the positive features) [E<sub>pos</sub>[f(x)-f(y)]<sup>2</sup>]
    - Uniformly [take all spaces in the hyperplane] [little complex but tangible 4.1.2]
      - l_uniform loss definition [!!]
      - Interpretation of 4.2 see our future paper !!
  - cluster need to form spherical cap
  - Theoretical metric for above two constraints??
    - Congruous with CL
    - gaussing RBF kernel e^{[f(x) -f(y)]^2} helps on uniform distribution achieving.
  - Result figure-7 [interesting]
  - Alignment and uniform loss

1. Tishby, Naftali, and Noga Zaslavsky. "Deep learning and the information bottleneck principle." In 2015 IEEE Information Theory Workshop (ITW), pp. 1-5. IEEE, 2015.

1. Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

1. Belghazi, Mohamed Ishmael, Aristide Baratin, Sai Rajeshwar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and Devon Hjelm. "Mutual information neural estimation." In International Conference on Machine Learning, pp. 531-540. 2018.

1. Gutmann, Michael, and Aapo Hyvärinen. "Noise-contrastive estimation: A new estimation principle for unnormalized statistical models." In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pp. 297-304. 2010.

1. Caron, Mathilde, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. "Unsupervised learning of visual features by contrasting cluster assignments." arXiv preprint arXiv:2006.09882 (2020).
  - SwAV (online algorithm) [swapped assignments between multiple vies of same image]
  - Contrastive learning, clustering
  - Predict cluster from different representation, memory efficiency!
  - 'code' consistency between image and its transformation {target}
    - similarity is formulated as a swapped prediction problem between positive pairs
    - no negative examples
    - the minibatch clustering methods implicitly prevent collapse of the representation space by encouraging samples in a batch to be distributed evenly to different clusters.
  - online code computation
  - Features and codes are learnt online
  - multi-crop: Smaller image with multiple views
  - validation: ImageNet linear evaluation protocol
  - Interested related work section
  - Key motivation: Contrastive instance learning
  - Partition constraint (batch wise normalization) to avoid trivial solution

1. Chen, Xinlei, and Kaiming He. "Exploring Simple Siamese Representation Learning." arXiv preprint arXiv:2011.10566 (2020).
  - Not using Negative samples, large batch or Momentum encoder!!
  - Care about Prevention of collapsing to constant *(one way is contrastive learning, another way - Clustering, or online clustering, BYOL)
  - Concepts in figure 1
  - Stop collapsing by introducing the stop gradient operation.
  - interesting section in 3
    - Loss SimCLR (but differently ) [eq 1 and eq 2]
    - Detailed Eq 3 and eq 4
    - Empirically shown to avoid the trivial solution

1. Caron, Mathilde, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. "Unsupervised learning of visual features by contrasting cluster assignments." Advances in Neural Information Processing Systems 33 (2020).

1. Morgado, Pedro, Nuno Vasconcelos, and Ishan Misra. "Audio-visual instance discrimination with cross-modal agreement." arXiv preprint arXiv:2004.12943 (2020).
  - learning audio and video representation [audio to video and video to audio!]
    - How they showed its better??
    - Exploit cross-modal agreement [what setup!] how it make sense!
  - consider in-sync audio video, proposed AVID (au-visu-instan-discrim)
  - Experiments with UCF-101, HMDB-51
  - Discussed limitation AVID & proposed improvement
    - Optimization method - [dimentional reduction by LeCunn and NCE paper]
      - Training procedure in section 3.2 AVID
    - Cross modal agreement!! it groups (how?) similar videos [both audio and visual]
  - Prior arts - binary task of audio-video alignment instance-based
    - This paper: matches in the representation embedding domain.
  - AVID calibrated by formulating CMA??
  - Figure 2: Variant of avids [summary of the papers]
    - This people are first to do it!!
    - Joint and Self AVID are bad in result! Cross AVID is the best for generalization in results!
  - CMA - Extension of the AVID [used for fine tuning]
    - section 4: Loss function extension with cross- AVID and why we need this?

1. Robinson, Joshua, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. "Contrastive Learning with Hard Negative Samples." arXiv preprint arXiv:2010.04592 (2020).
  - Sample good negative (difficult to distinguish) leads better represenation
    - challenges: No label! unsupervised method! Control the hardness!
    - enables learning with fewest instances and distance maximization.
  - Problem (1): what is true label? sol: Positive unlabeled learning !!!
  - Problem (2): Efficient sampling? sol:  efficient importance sampling!!! (consider lack of dissimilarity information)!!!
  - Section 3 most important!
  - section 4 interesting
  - [Notes](https://github.com/mxahan/PDFS_notes/blob/master/contrastive_sampling_debias_hardMining.pdf)

1. Chuang, Ching-Yao, Joshua Robinson, Yen-Chen Lin, Antonio Torralba, and Stefanie Jegelka. "Debiased contrastive learning." Advances in Neural Information Processing Systems 33 (2020).
  - Sample bias [negatives are actually positive! since randomly sampled]
  - need unbiased - improves vision, NLP and reinforcement tasks.
  - Related to Positive unlabeled learning
  - interesting results
  - [Notes](https://github.com/mxahan/PDFS_notes/blob/master/contrastive_sampling_debias_hardMining.pdf)

1. Zhao, Nanxuan, Zhirong Wu, Rynson WH Lau, and Stephen Lin. "What makes instance discrimination good for transfer learning?." arXiv preprint arXiv:2006.06606 (2020).

1. Metzger, Sean, Aravind Srinivas, Trevor Darrell, and Kurt Keutzer. "Evaluating Self-Supervised Pretraining Without Using Labels." arXiv preprint arXiv:2009.07724 (2020).

1. Bhardwaj, Sangnie, Ian Fischer, Johannes Ballé, and Troy Chinen. "An Unsupervised Information-Theoretic Perceptual Quality Metric." Advances in Neural Information Processing Systems 33 (2020).

1. Qian, Rui, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, and Yin Cui. "Spatiotemporal contrastive video representation learning." arXiv preprint arXiv:2008.03800 (2020).
  - unlabeled videos! video representation, pretext task! (CVRL)
  - simple idea: same video together and different videos differ in embedding space.  (TCN works on same video)
  - Video SimCLR?
  - Inter video clips (positive negatives)
  - method
    - Simclr setup for video (InfoNCE loss)
    - Video encoder (3D resnets)
    - Temporal augmentation for same video (good for them but not Ubiquitous)
    - Image based spatial augmentation (positive)
    - Different video (negative)
  - Downstream Tasks
    - Action Classification
    - Action Detection

1. Ye, Mang, Xu Zhang, Pong C. Yuen, and Shih-Fu Chang. "Unsupervised embedding learning via invariant and spreading instance feature." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6210-6219. 2019.
  - Contrastive idea but uses siamese network.

1. Li, Junnan, Pan Zhou, Caiming Xiong, Richard Socher, and Steven CH Hoi. "Prototypical contrastive learning of unsupervised representations." arXiv preprint arXiv:2005.04966 (2020).
  - Addresses the issues of instance wise learning (?)
    - issue 1: semantic structure missing
  - claims to do two things
    - Learn low-level features for instance discrimination
    - encode semantic structure of the data
  - prototypes as latent variables to help find the maximum-likelihood estimation of the network parameters in an Expectation-Maximization framework.
    - E-step as finding the distribution of prototypes via clustering
    - M-step as optimizing the network via contrastive learning
  - Offers new loss function ProtoNCE (Generalized InfoNCE)
  - Show performance for the unsupervised representation learning benchmarking (?) and low-resolution transfer tasks
  - Prototype: a representative embedding for a group of semantically similar instances
    - prototype finding by standard clustering methods
  - Goal described in figure 1
  - EM problem?
    - goal is to find the parameters of a Deep Neural Network (DNN) that best describes the data distribution, by iteratively approximating and maximizing the log-likelihood function.
    - assumption that the data distribution around each prototype is isotropic Gaussian
  - Related works: MoCo
    - Deep unsupervised Clustering: not transferable?
  - Prototypical Contrastive Learning (PCL)
    - See the math notes from section 3
  - Figure 2 - Overview of methods
  -[github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers2.pdf)

1. Bachman, Philip, R. Devon Hjelm, and William Buchwalter. "Learning representations by maximizing mutual information across views." arXiv preprint arXiv:1906.00910 (2019).
  - Multiple view of shared context
  - Why MI (analogous to human representation, How??) - same understanding regardless of view
    - Whats the problem with others !!
  - Experimented with Imagenet
  - Extension of local DIM in 3 ways (this paper calls it - augmented multi-scale DIM (AMDIM))
    - Predicts features for independently-augmented views
    - predicts features across multiple views
    - Uses more powerful encoder
  - Methods relevance: Local DIM, NCE, Efficient NCE computation, Data Augmentation, Multi-scale MI, Encoder, mixture based representation

1. Chen, Xinlei, Haoqi Fan, Ross Girshick, and Kaiming He. "Improved baselines with momentum contrastive learning." arXiv preprint arXiv:2003.04297 (2020).

1. Goyal, Priya, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. "Scaling and benchmarking self-supervised visual representation learning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6391-6400. 2019.

1. Tao, Li, Xueting Wang, and Toshihiko Yamasaki. "Self-Supervised Video Representation Using Pretext-Contrastive Learning." arXiv preprint arXiv:2010.15464 (2020).
  - Tasks and Contrastive setup connection  (PCL)
  - hypersphere features spaces
  - Combine Pretext(some tasks, intra information)+Contrastive (similar/dissimilarity, inter-information) losses
  - Assumption: pretext and contrastive learning doing the same representation.
  - Loss function (Eq-6): Contrast (same body +head1)+pretext task (same body + head2)! - Joint optimization
  - Figure 1- tells the story
  - Three pretext tasks (3DRotNet, VCOP, VCP) - Experiment section
  - Both RGB and Frame difference
  - Downstrem tasks (UCF and HMDB51)
  - Some drawbacks at the discussion section.

1. Doersch, Carl, Abhinav Gupta, and Alexei A. Efros. "Unsupervised visual representation learning by context prediction." In Proceedings of the IEEE international conference on computer vision, pp. 1422-1430. 2015.
  - Crop position learning pretext!
  - Figure 2 (problem formulation) and 3 (architectures) shows the key contribution

1. Noroozi, Mehdi, and Paolo Favaro. "Unsupervised learning of visual representations by solving jigsaw puzzles." In European conference on computer vision, pp. 69-84. Springer, Cham, 2016.
  - Pretext tasks (solving jigsaw puzzle) - self-supervised

1. Han, Tengda, Weidi Xie, and Andrew Zisserman. "Video representation learning by dense predictive coding." In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pp. 0-0. 2019.
  - Self-supervised AR (DPC)
    - Learn dense coding of spatio-temporal blocks by predicting future frames (decrease with time!)
    - Training scheme for future prediction  (using less temporal data)
  - Care for both temporal and spatial negatives.
  - Look at their case for  - the easy negatives (patches encoded from different videos), the spatial negatives (same video but at different spatial locations), and the hard negatives (TCN)
  - performance evaluated by Downstream tasks (Kinetics-400 dataset (pretrain), UCF101, HMDB51- AR tasks)
  - Section 3.1 and 3.2 are core (contrastive equation - 5)

1. Sohn, Kihyuk. "Improved deep metric learning with multi-class n-pair loss objective." In Proceedings of the 30th International Conference on Neural Information Processing Systems, pp. 1857-1865. 2016.
  - Deep metric learning (solves the slow convergence for the contrastive and triple loss)
    - what is the penalty??
    - How they compared the convergences
  - This paper: Multi-class N-pair loss
    - developed in two steps (i) Generalization of triplet loss (ii) reduces computational complexity by efficient batch construction (??) taking (N+1)xN examples!!
  - Experiments on visual recognition, object recognition, and verification, image clustering and retrieval, face verification and identification tasks.
  - identify multiple negatives [section 3], efficient batch construction
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Dosovitskiy, Alexey, Philipp Fischer, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. "Discriminative unsupervised feature learning with exemplar convolutional neural networks." IEEE transactions on pattern analysis and machine intelligence 38, no. 9 (2015): 1734-1747.

1. Tschannen, Michael, Josip Djolonga, Marvin Ritter, Aravindh Mahendran, Neil Houlsby, Sylvain Gelly, and Mario Lucic. "Self-supervised learning of video-induced visual invariances." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13806-13815. 2020.
  - Framework to learn image representations from non-curated videos in the wild by learning **frame-**, **shot-**, and **video-level invariances**
    -  benefit greatly from exploiting video-induced invariances,
  - Experiment with Youtube8M (3.7 M
  - Scale-able to image and video data. (however requires frame label encoding!! what about video AR networks)

1. Ma, Shuang, Zhaoyang Zeng, Daniel McDuff, and Yale Song. "Learning Audio-Visual Representations with Active Contrastive Coding." arXiv preprint arXiv:2009.09805 (2020).

1. Sun, Chen, Fabien Baradel, Kevin Murphy, and Cordelia Schmid. "Learning video representations using contrastive bidirectional transformer." arXiv preprint arXiv:1906.05743 (2019).

1. Xiao, Tete, Xiaolong Wang, Alexei A. Efros, and Trevor Darrell. "What should not be contrastive in contrastive learning." arXiv preprint arXiv:2008.05659 (2020).
  - What if downstream tasks violates data augmentation (invariance) assumption!
    - Requires prior knowledge of the final tasks
  - This paper: Task-independent invariance
    - Requires separate embedding spaces!  (how much computation increases, redundant!)
      - Surely, multihead networks and shared backbones
      - new idea: Invariance to all but one augmentation !!
  - Pretext tasks: Tries to recover transformation between views
  - Contrastive learning: learn the invariant of the transformations
  - Is augmentation helpful: Not always!
    - rotation invariance removes the orientation senses! Why not keep both! disentanglement and the multitask!
  - This paper: Multi embedding space (transfer the shared backbones and task specific heads)
    - Each head is sensitive to all but one transformations
    - LooC: (Leave-one-out Contrastive Learning)
      - Multi augmentation contrastive learning
      - view generation and embedded space
      - Figure 2 (crack of jack)
    - Good setup to apply the ranking loss function
    - Careful with the notation (bad notation)
    -  instances and their augmentations!
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Weinberger, Kilian Q., John Blitzer, and Lawrence K. Saul. "Distance metric learning for large margin nearest neighbor classification." In Advances in neural information processing systems, pp. 1473-1480. 2006.
  - Triplet loss proposal

1. Liu, Yang, Keze Wang, Haoyuan Lan, and Liang Lin. "Temporal Contrastive Graph for Self-supervised Video Representation Learning." arXiv preprint arXiv:2101.00820 (2021).
  - Graph Neural Network And Contrastive Learning
  - Video frame shuffling

1. Tao, Li, Xueting Wang, and Toshihiko Yamasaki. "Self-supervised video representation learning using inter-intra contrastive framework." In Proceedings of the 28th ACM International Conference on Multimedia, pp. 2193-2201. 2020.
  - Notion of intra-positive (augmentation, optical flow, frame differences, color of same frames)
  - Notion of intra-negative (frame shuffling, repeating)
  - Inter negative (irrelevant video clips - Different time or videos)
  - Two (Multi) Views (same time - positive keys) - and intra-negative
  - Figure 2 (memory bank approach)
  - Consider contrastive from both views.

1. Chen, Ting, and Lala Li. "Intriguing Properties of Contrastive Losses." arXiv preprint arXiv:2011.02803 (2020).
  - Generalize the CL loss to broader family of losses
    - weighted sum of alignment and distribution loss
      - Alignment: align under some transformation
      - Distribution: Match a prior distribution
  - Experiment with weights, temperature, and multihead projection
  - Study feature suppression!! (competing features)
    - Impacts of final loss!
    - Impacts of data augmentation
  - Suppression feature phenomena: Reduce unimportant features
  - Experiments
    - two ways to construct data
  - Expand beyond the uniform hyperspace prior
    - Can't rely on logSumExp setting in NT-Xent loss
    - Requires new optimization (Sliced Wasserstein distance loss)
      - Algorithm 1
    - Impacts of temperature and loss weights
  - Feature suppression
    - Target: Remove easy-to-learn but less transferable features for CL (e.g. Color distribution)
    - Experiments by creating the dataset
      - Digit on imagenet dataset
      - RandBit dataset
  -[github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Anand, Ankesh, Evan Racah, Sherjil Ozair, Yoshua Bengio, Marc-Alexandre Côté, and R. Devon Hjelm. "Unsupervised state representation learning in atari." arXiv preprint arXiv:1906.08226 (2019).

1. Gordon, Daniel, Kiana Ehsani, Dieter Fox, and Ali Farhadi. "Watching the world go by: Representation learning from unlabeled videos." arXiv preprint arXiv:2003.07990 (2020).
  - Multi-frame multi-pairs positive negative (single imgae)- instance discrimination

1. Alwassel, Humam, Dhruv Mahajan, Bruno Korbar, Lorenzo Torresani, Bernard Ghanem, and Du Tran. "Self-supervised learning by cross-modal audio-video clustering." arXiv preprint arXiv:1911.12667 (2019).

1. Chen, Ting, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. "Big self-supervised models are strong semi-supervised learners." arXiv preprint arXiv:2006.10029 (2020).
  - Empirical paper
  - unsupervised pretrain (task agnostic), semi-supervised learning (task dependent), fine-tuning
  - Impact of Big (wide and deep) network
    - More benefited from the unsupervised pretraining
    - Big Network: Good for learning all the Representations  
      - Not necessarily require in one particular task - So we can leave the unnecessary information
    - More improve by distillation after fine tuneing
  - Wow: Maximum use of works: Big network to learn representations, and fine-tune then distill the task (supervised and task specific fine-tune) to a smaller network
  - Proposed Methods (3 steps) [this paper]
    - Unsupervised pretraining (SimCLR v2)
    - Supervised fine-tune using smaller labeled data
    - Distilling with unlabeled examples from big network for transferring task-specific knowledge into smaller network
  - This paper:
    - Investigate unsupervised pretraining and selfsupervised fine-tune
      - Finds: network size is important
      - Propose to use big unsupervised_pretraining-fine-tuning network to distill the task-specific knowledge to small network.
    - Figure 3 says all
  - Key contribution
    - Big networks works best (in unsupervised pretraining - very few data fine-tune) althought they may overfit!!!
    - Big model learns general representations, may not necessary when task-specific requirement
    - Importance of multilayer transform head (intuitive because of the transfer properties!! great: since we are going away from metric dimension projection, so more general features)
  - Section 2: Methods (details from loss to implementation)
  - Empirical study findings
    - Bigger model are label efficient
    - Bigger/Deeper Projection Heads Improve Representation Learning  
    - Distillation using unlabeled data improves semi-supervised learning

1. Zbontar, Jure, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." arXiv preprint arXiv:2103.03230 (2021).
  - Tries to avoid trivial solution (by new objective function, cross-correlation matrix)!!
    - A new loss function
  - Same images ~ augmentation representations are distorted version of each other.
  - Motivation: Barlows Redundancy-reduction principle
  - Pair of identical neurons
  - Just WOW: The hell the idea is!!
    - Intriguingly avoids trivial solutions
    - Should require large batch size
  - Figure 1 & algorithm 1: Crack of all jack
  - BOYL follow up works
  - Contrastive learning
    - Either Negative examples
    - Or architectural constraint/ Asymmetric update  
  - Motivated from Barlow-twin (redundancy-reduction principle) 1961!
    - H. Barlow hypothesized that the goal of sensory processing is to recode highly redundant sensory inputs into a factorial code (a code with statistically independent components).  
    - propose new loss function:  tries to make the cross-correlation matrix computed from twin representations as close to the identity matrix as possible
      - Advantages: Not required Asymmetric update or large batch size
  - Methods description
    - What is Barlow twins
      - connection to Information bottleneck (IB)
    - Implementation details
  - Result  
    - Linear and Semi-supervised Evalution of imagenet
    - Transfer learning
    - Object detection and segmentation
  - Ablation study
    - Variation of Loss function
    - Impacts of Batch Size
      - Outperformed by Large BS with BYOL and SimCLR
    - network selection impacts
      - projection head importance
    - Importance of the data augmentation
  - Discussion (interesting)
    - Comparison with Prior Art
      - InfoNCE
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers2.pdf)

1. Xie, Zhenda, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, and Han Hu. "Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning." arXiv preprint arXiv:2011.10043 (2020).
  - Alternative to instance-level pretext learning - Pixel-level pretext learning!
    - Pixel level pretext learning! pixel to propagation consistency!!
      - Avail both backbone and head network! to reuse
      - complementary to instance level CL
      - How to define pixel level pretext tasks!
    - Why instance-label is suboptimal? How? Benchmarking!
    - Dense feature learning
  - Application
    - Object detection (Pascal VOC object detection)
  - Pixel level pretext tasks
    - Each pixel is a class!! what!!
      - Features from same pixels are same !
    - PixContrast: Training data collected in self-supervised manner
    - requires pixel feature vector !
    - Feature map is warped into the original image space
    - Now closer pictures together and .... contrastive setup
    - Learns spatially sensitive information
  - Pixel-to-propagation consistency !! (pixpro)
    - positive pair obtaining methods
    - asymmetric pipeline
    - Learns spatial smoothness information
      - Pixel propagation module
      - pixel to propagation  consisency loss
    - Figure 3

1. Piergiovanni, A. J., Anelia Angelova, and Michael S. Ryoo. "Evolving losses for unsupervised video representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 133-142. 2020.
  - video representation learning! (generic and transfer) (ELo)
    - Video object detection
  - Zeroshot and fewshot AcRecog
  - introduce concept of loss function evolving (automatically find optimal combination of loss functions capturing many (self-supervised) tasks and modalities)
      - using an evolutionary search algorithm (!!)
  - Evaluation using distribution matching and Zipf's law!
  - only outperformed by fully labeled dataset
  - This paper: new unsupervised learning of video representations from unlabeled video data.
    - multimodal, multitask, unsupervised learning
      - Youtube dataset
    - combination of single modal tasks and multi-modal tasks
    - too much task! how they combined it!! Engineering problem
      - evolutionary algorithm to solve these puzzle
      - Power law constraints and KL divergence  
    - Evolutionary search for a loss function that automatically combines self-supervised and distillation task
    - unsupervised representation evaluation metric based on power law distribution matching
  - Multimodal task, multimodal distillation
    - show their efficacy by the power distribution of video classes (zipf's law)
  - Figure 2:
    - overwhelming of tasks! what if no commonalities.
  - Hypo: Synchronized multi-modal data source should benefit representation learning
    - Distillation losses between the multiple streams of networks + self supervised loss
  - Methods
    - Multimodal learning
    - Evolving an unsupervised loss function
      - [0-1] constraints
      - zipfs distribution matching
        - Fitness measurement - k-means clustering
        - use smaller subset of data for representation learning
        - Cluster the learned representations
        - the activity classes of videos follow a Zipf distribution
        - HMDB, AVA, Kinetics dataset, UCF101
      - ELo methods with baseline weakly-supervised methods
      - Self-supervised learning
        - reconstruction (encoder-decoder) and prediction tasks (L2 distance minimization)
        - Temporal ordering
        - Multi-modal contrastive loss {maxmargin loss}
      - ELo and ELo+Distillation
  - [github notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Hénaff, Olivier J., Skanda Koppula, Jean-Baptiste Alayrac, Aaron van den Oord, Oriol Vinyals, and João Carreira. "Efficient Visual Pretraining with Contrastive Detection." arXiv preprint arXiv:2103.10957 (2021).
  - Tackles the computational complexity of the self-supervised learning
    - By providing new objective (extract rich information from each image!! ) named *Contrastive Detection* (Figure 2)
      - Two variants: SimCLR and the BYOL
      - Knowledge tx across the dataset
      - Heuristic mask on image and train!
        - Of the shelf unsupervised/human annotators [external methods]
      - Pull the features spaces close!!
    -  maximizes the similarity of object-level features across augmentations.
    - Result 5x less pretraining
    - Compared with SEER!!
  - Experiment with imagenet to COCO dataset
  - This paper: Special Data augmentation scheme

1. Appalaraju, Srikar, Yi Zhu, Yusheng Xie, and István Fehérvári. "Towards Good Practices in Self-supervised Representation Learning." arXiv preprint arXiv:2012.00868 (2020).
  - Tries to unravel the mysteries behind CL (!!)
  - Empirical analysis: provide practice/insight tips
  - Why data augmentation and adding projection head works in CL??
    - not true in the supervised setting!!
  - Design choice and good practice boost CL representation!
  - This paper: Focuses (empirical analysis) on three of the key points
  - (i) Importance of MLP heads (key factor)
  - (ii) semantic label shift problems by data augmentation:  breakdown of class boundaries in strong augmentations
  - (iii) Investigate on Negative Samples: Quality and Quantity

1. Bulat, Adrian, Enrique Sánchez-Lozano, and Georgios Tzimiropoulos. "Improving memory banks for unsupervised learning with large mini-batch, consistency and hard negative mining." arXiv preprint arXiv:2102.04442 (2021).
  - Improvement for the memory bank based formulation (whats the problem??)
    - TP: (I) Large mini-batch: Multiple augmentation! (II) Consistency: Not negative enforce! The heck? how to prevent collapse? (III) Hard Negative Mining
    - Results: Improve the vanilla memory bank! Evidence!! Dataset experimentation!
  - Exploration:  With Batch Size and visually similar instances (is the argument 2 is valid?)
  - Contribution 2 seems important!
  - Each image is augmented k times: More data augmentation!
  - Interesting way to put the negative contrastive parts to avoid collapse (eq 3)
  - Experiments: Seen testing categories (CIFAR, STL), & unseen testing categories (Stanford Online Product). ResNet-18 as baseline model   

1. Dwibedi, Debidatta, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. "With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations." arXiv preprint arXiv:2104.14548 (2021).
  - Positive from other instance (instead of augmented version of same image!)
  - positive sampling: Nearest neighbor in the latent space (NNCLR) [this covers both same samples and Nearest neighbor in the latent spaces]
  - Benefit: Less reliant on complex data augmentation (empirical results)
  - Experiments: semi-sup benchmark, tx-learning benchmark
  - Training: figure 1 (support set similar to memory bank but only provides positive samples)
  - Similarity across previously seen objects! (tricky implementation)! Initialization!!
  - Figure 2: details and key difference with others! (requires support set!)
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Han, Tengda, Weidi Xie, and Andrew Zisserman. "Self-supervised co-training for video representation learning." arXiv preprint arXiv:2010.09709 (2020).
  - visual only selfsupervised Representation learning!! [sampling method for CL]
  - new form of supervised CL (adding semantic positive to instance based Infor NCE loss)!! HOW? {explanation in figure 1}
  - co-training scheme with infoNCE loss!! (Explore complementary information from different views
  - evaluating the quality of learned representations (two downstream tasks: action recognition and video retrieval)
  - Question of "is instance discrimination is best??" - NO
    - hard positves! oracle experiment (UberNCE!)
    - CoCLR!! mining positive samples of data. (TCN extension by adding another modality, different network for each modality! how it is trained? Dynamic objective!!)
  - Focused more on sampling procedure
  - Experimented with RGB and FLOW network (aim to improve their representation)
  - Related works: Visual-only supervised learning, Multi-modal self-supervised learning, Co-training Paired networks and Video action recognition
  - InfoNCE and UberNCE *differs in sampling positives*
  - CoCLR algorithm (Initialization: flow and RGB net trained individually, Alternation: mine hard positive based on others Eq 3,4)
  - Dataset: UCF101, Kinetics-400, HMDB51

1. Goyal, Priya, Mathilde Caron, Benjamin Lefaudeux, Min Xu, Pengchao Wang, Vivek Pai, Mannat Singh et al. "Self-supervised pretraining of visual features in the wild." arXiv preprint arXiv:2103.01988 (2021).
  - SEER and connection to few shot learning
  - RQ: pretraining extremely large collection of uncurated, unlabeled images for good achievement?
    - Solution: Continuous learning in a self-supervised manner! (online fashion training)
  - TP: pretrain high capacity model (RegNet Architecture!! 700M params) on billions images! using SwAV approaches with large BS.
  - Results: One of the best model!! both on curated and uncarated data.
  - Related works: Scale on large uncurated images. (also scaling the network - Good Reads)
  - Results on finetuning large models, low shot learning and transfer learning.
  - Ablation studies: Model architecture, scaling the training data, scaling the self-supervised model head

1. Miech, Antoine, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. "End-to-end learning of visual representations from uncurated instructional videos." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9879-9889. 2020.
  - Experiments with Dataset: HowTo100M (task progression) [video with narration]
  - This paper: (multi-instance learning) MIL-NCE (to address misalignment in video narration)!! HOW? Requires instructional videos! (video with text)
  - Video representation learning: shows effectiveness in four downstream tasks: AR (HMDB, UCF, Kinetics), action localization (Youtube8M, crosstask), action segmentation (COIN), Text-to-video retrieval (YouCook2, MSR-VTT).
  - TP: Learn video representation from narration only (instructional video)!
  - Related works: Learning visual representation from unlabeled videos. (ii) Multiple instance learning for video understanding (MIL) [TP: connect MCE with MIL]
  - Two networks in the NCE Calculation : [unstable targets!!]
  - Experiments: Network: 3D CNN

1. Joulin, Armand, Laurens Van Der Maaten, Allan Jabri, and Nicolas Vasilache. "Learning visual features from large weakly supervised data." In European Conference on Computer Vision, pp. 67-84. Springer, Cham, 2016.

1. Xie, Qizhe, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le. "Unsupervised data augmentation for consistency training." arXiv preprint arXiv:1904.12848 (2019).
  - TP: how to effectively noise unlabeled examples (1) and importance of advanced data augmentation (2)
    - Investigate the role of noise injection and advanced data augmentation
    - Proposes better data augmentation in consistency training: Unsupervised Data Augmentation (UDA)
    - Experiments with vision and language tasks
  - Bunch of experiment with six language tasks and three vision tasks.
  - Consistency training as regularization.
  - UDA: Augment unlabeled data!! and quality of the noise for augmentations.
  - Noise types: Valid noise, Diverse noise, and Targeted Inductive biases
  - Augmentation types: RandAugment for image, backtranslating the language
  - Training techniques: confidence based masking, Sharpening Predictions, Domain relevance data filtering.
  - Interesting graph comparison under three assumption.
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Dave, Ishan, Rohit Gupta, Mamshad Nayeem Rizve, and Mubarak Shah. "TCLR: Temporal Contrastive Learning for Video Representation." arXiv preprint arXiv:2101.07974 (2021).
  - temporal CL framework!!- (why?) - two novel loss functions
    - (local-local): non-overlapping of same videos!!
    - (global-local): increase temporal diversity!!
    - NCE based loss function formulation.
  - Interesting way to sample local and global (why is it necessary???) [figure 2, 3, and 4]
  - Architecture: 3D ResNet-18

1. Tian, Yuandong, Xinlei Chen, and Surya Ganguli. "Understanding self-supervised learning dynamics without contrastive pairs." arXiv preprint arXiv:2102.06810 (2021).

1. Caron, Mathilde, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. "Emerging properties in self-supervised vision transformers." arXiv preprint arXiv:2104.14294 (2021).
  - DINO: knowledge DIstillation with NO labels (Figure 2, Algorithm 1)
  - Self-supervised learning with vision transformer
  - Observations: (i) T features contain explicit information (!!!) about the semantic segmentation of an image
    - explicit information: scene layout, object boundaries (directly accessible)
    - Performs basic k-NN without any supervision.
  - Cross entropy loss (sharpening and centering requires to avoid collapse)
  - Experiments: Architecture (i) ViT and (ii) ResNet

1. Aljalbout, Elie, Vladimir Golkov, Yawar Siddiqui, Maximilian Strobel, and Daniel Cremers. "Clustering with deep learning: Taxonomy and new methods." arXiv preprint arXiv:1801.07648 (2018).
  - Three components: Main encoder networks (concerns with architecture, Losses, and cluster assignments)
  - Non-cluster loss: Autoencoder reconstruction losses
  - Various types of clustering loss (note)
  - Combine losses: Pretraining / jointly training / variable scheduleing
  - Cluster update: Jointly update with the network model / Alternating update with network models
  - Relevant methods: Deep Embedded Clustering (Xie et al, 2016), Deep Clustering Network (yang et al, 2016), Discriminatively Boosted Clustering (Li et al, 2017), ..  

1. Tan, Hao, Jie Lei, Thomas Wolf, and Mohit Bansal. "VIMPAC: Video Pre-Training via Masked Token Prediction and Contrastive Learning." arXiv preprint arXiv:2106.11250 (2021).

1. Tian, Yonglong, Olivier J. Henaff, and Aaron van den Oord. "Divide and Contrast: Self-supervised Learning from Uncurated Data." arXiv preprint arXiv:2105.08054 (2021).
  - Effects of contrastive learning from larger, less-curated image datasets such as YFCC
    - Finds a large difference in the resulting representation quality
    - because (hypothesis) distribution shift in image class (less relevant negative to learn)
  - TP: new approach DnC (divide and contrast) - alternate between CL and cluster based hard negative mining
    - Methods: Train individual models on subset and distill them into single model
    - Application Scope: less curated data to train! Aim: attempts to recover local consistency.
    - The distillation parts requires k+1 networks!
  - what if: the networks reaches different embedding in each running (from scratch!)
    - How come predicting both simultaneously makes sense?
  - Experiment dataset: JFT-300, YFCC100M, 95M Flickr

1. Zheltonozhskii, Evgenii, Chaim Baskin, Avi Mendelson, Alex M. Bronstein, and Or Litany. "Contrast to Divide: Self-Supervised Pre-Training for Learning with Noisy Labels." arXiv preprint arXiv:2103.13646 (2021).
  - **warm-up obstacle**: the inability of standard warm-up stages to train high quality feature extractors and avert memorization of noisy labels.!!
    - SoTA depends on warm-up stage where standard supervised training is performed using the full (noisy) training set
  - TP: contrast to divide (C2D)
    - benefit: drastically reducing the warm-up stage’s susceptibility to noise level, shortening its duration, and increasing extracted feature quality
  - warp-up stages!! Current works focus on warm up length only! requires optimal warm-up length! or relying on external dataset! TP: Self-supervised pretraining!
  - TP: firstly perform simclr, then proceed with standard LNL algorithm (: ELR+ and DivideMix!!)

1. Huang, Lianghua, Yu Liu, Bin Wang, Pan Pan, Yinghui Xu, and Rong Jin. "Self-supervised Video Representation Learning by Context and Motion Decoupling." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13886-13895. 2021.
  - a method that explicitly decouples motion supervision from context bias through a *carefully designed* pretext task
    - (i) Context matching (CL between key frame (how to find it?? - Random frame selection) and video clips) & Motion Prediction (estimate motion features in the future & also a regularizer)
    - Architecture: Shared backbone and separate head for the tasks
  - Figure 2: says all: Two target for the V-network (context {top, extracted from image} & motion {bottom, derived using the motion vectors?})
  - Related works: representation learning, AR in compressed videos, Motion prediction
  - Experiments:
    - Networks: Video backbone: C3D, R(2+1)D-26 and R3D-26 (V, video), shallow R2D-10 (I, context), and R3D-10 (video).
    - Data: UCF, Kinetics, HMDB51, with augmentation (same for one example), hard negatives,  

1. Feichtenhofer, Christoph, Haoqi Fan, Bo Xiong, Ross Girshick, and Kaiming He. "A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3299-3309. 2021.
  - A large-scale study on unsupervised spatiotemporal representation learning from videos
    - Generalize image based method into space-time. (e.g. crop [image] to clip [video])
    - *Provides many empirical results*

1. Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis." In International conference on machine learning, pp. 478-487. PMLR, 2016.
  - Deep Embedded Clustering (DEC) Learns (i) Feature representation (ii) cluster assignments
  - Experiment: Image and text corpora
  - Contribution: (a) joint optimization of deep embedding and clustering; (b) a novel iterative refinement via soft assignment (??); (c) state-of-the-art clustering results in terms of clustering accuracy and speed
  - target distribution properties: (1) strengthen predictions (i.e., improve cluster purity), (2) put more emphasis on data points assigned with high confidence, and (3) normalize loss contribution of each centroid to prevent large clusters from distorting the hidden feature space.
  - Computational complexity as iteration over large data samples

1. Dwibedi, Debidatta, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. "Temporal cycle-consistency learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1801-1810. 2019.
  - Temporal video alignment problem: the task of finding correspondences across multiple videos despite many factors of variation
  - TP: Temporal cycle consistency losses (complementary to other methods [TCN, shuffle and learn])
  - Dataset: Penn AR, Pouring dataset
  - TP: Focus on temporal reasoning!! (metrics)
    - learns representations by aligning video sequences of the same action
    - Requires differentiable cycle-consistency losses
  - Figure 2: interesting but little hard to understand

1. Liu, Xiao, Fanjin Zhang, Zhenyu Hou, Li Mian, Zhaoyu Wang, Jing Zhang, and Jie Tang. "Self-supervised learning: Generative or contrastive." IEEE Transactions on Knowledge and Data Engineering (2021).
  - Another Survey Paper

1. Akbari, Hassan, Linagzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. "Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text." arXiv preprint arXiv:2104.11178 (2021).

1. Zhu, Benjin, Junqiang Huang, Zeming Li, Xiangyu Zhang, and Jian Sun. "EqCo: Equivalent Rules for Self-supervised Contrastive Learning." arXiv preprint arXiv:2010.01929 (2020).
  - Theoretical paper: Challenges the large number of negative samples [algorithmic]
    - Though more negative pairs are usually reported to derive better results, Interpretation: it may be because the hyper-parameters in the loss are not set to the optimum according to different numbers of keys respectively
    - Rule to set the hyperparameters
    - SiMo: Alternate for the InfoNCE
  - EqCo: Concept of Batch Size (N) and Negative Sample number (K)
    - CPC work follow-up
    - the learning rate should be adjusted proportional to the number of queries N per batch
    - linear scaling rule needs to be applied corresponding to N rather than K
  - SiMo: cancel the memory bank as rely on a few negative samples per batch. Instead, use the momentum encoder to extract both positive and negative key embeddings from the current batch.

1. Sordoni, Alessandro, Nouha Dziri, Hannes Schulz, Geoff Gordon, Philip Bachman, and Remi Tachet Des Combes. "Decomposed Mutual Information Estimation for Contrastive Representation Learning." In International Conference on Machine Learning, pp. 9859-9869. PMLR, 2021.

1. Yang, Mouxing, Yunfan Li, Zhenyu Huang, Zitao Liu, Peng Hu, and Xi Peng. "Partially view-aligned representation learning with noise-robust contrastive loss." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1134-1143. 2021.
  - Partially view alignment Problem (PVP)???
  - Objective: aligning data and learning representation (why it is important??)
  - TP: Propose noise robust Contrastive loss to eliminate False negative {related to learning with noisy labels, any evidence on this?}
  - New definition of noisy labels (why, is it valid?) [false negative pair as noisy label! ]
  - Experiment: 10 SOTA classification and clustering tasks!
  - Two assumption on Data: Completeness (contains all views [partially data-missing problem]) and Consistency (no false negative/positive [partial view alignment problem]) of views.  
  - Proposed modified distance loss for the negative pairs.

1. Huang, Zhenyu, Peng Hu, Joey Tianyi Zhou, Jiancheng Lv, and Xi Peng. "Partially view-aligned clustering." Advances in Neural Information Processing Systems 33 (2020).

1. Chen, Kai, Lanqing Hong, Hang Xu, Zhenguo Li, and Dit-Yan Yeung. "MultiSiam: Self-supervised Multi-instance Siamese Representation Learning for Autonomous Driving." arXiv preprint arXiv:2108.12178 (2021).
  - TP: Two issues: (1) define positive samples for cross-view consistency ? (2) measure similarity in multi-instance circumstances ?
  - Experiments data: WayMo, SODA10M
  - Global consistency to local consistency?
  - basic assumption of instance discrimination: different views of the same image should be consistent in the feature space
    - what about multi-instance in a single image (realistic case)!!!
    - definition of positive samples is definitely needed to extend cross-view consistency framework to multi-instance circumstances. (multiple things in single image)
  - Methods: Uses IoU as proxy for data and Noise. (les IoU - More Noise)
    - Remove global pooling (counter the multiple instance collapse)
    - Well, very complex loss function!

1. Yang, Ceyuan, Zhirong Wu, Bolei Zhou, and Stephen Lin. "Instance localization for self-supervised detection pretraining." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3987-3996. 2021.
  - Propose a new self-supervised tasks : Instance Localization. [self-supervised task design]
  - Put image crop into another and try to predict using RPN!!

1. Wang, Feng, and Huaping Liu. "Understanding the behaviour of contrastive loss." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2495-2504. 2021.
  - Studies the impact of the temperature in loss function (uniformity and temperature).
    - How to ensure tolerant for semantic similar examples [uniformity tolerance dilemma]
  - This paper: studies hardness aware properties (parameter in loss function).

1. Pan, Tian, Yibing Song, Tianyu Yang, Wenhao Jiang, and Wei Liu. "Videomoco: Contrastive video representation learning with temporally adversarial examples." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11205-11214. 2021.

1. Xu, Jiarui, and Xiaolong Wang. "Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective." arXiv preprint arXiv:2103.17263 (2021).

1. Li, Xueting, Sifei Liu, Shalini De Mello, Xiaolong Wang, Jan Kautz, and Ming-Hsuan Yang. "Joint-task self-supervised learning for temporal correspondence." arXiv preprint arXiv:1909.11895 (2019).


# Semi-Supervised



1. Blum, Avrim, and Tom Mitchell. "Combining labeled and unlabeled data with co-training." In Proceedings of the eleventh annual conference on Computational learning theory, pp. 92-100. 1998.
  - Early semi-supervised learning (mathematical framework with graph setting)
    - Web page (and its augmentation)

1. Zhai, Xiaohua, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. "S4l: Self-supervised semi-supervised learning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1476-1485. 2019.
  - Pretext task of rotation angle prediction!!
    - Rotation, invariant across augmentation
  - Baseline: vitrural adversarial training [inject noise with the original images], EntMin
    - EntMin is bad: because the model can easily become extremely confident by increasing the weights of the last layer

1. Gupta, Divam, Ramachandran Ramjee, Nipun Kwatra, and Muthian Sivathanu. "Unsupervised Clustering using Pseudo-semi-supervised Learning." In International Conference on Learning Representations. 2019.

1. Grandvalet, Yves, and Yoshua Bengio. "Semi-supervised learning by entropy minimization." In CAP, pp. 281-296. 2005.
  - Semi-supervised learning by minimun entropy regularization!
    - result compared with mixture models ! (entropy methods are better)
    - Connected to cluster assumption and manifold learning
  - Motivation behind supervised training for unlabeled data
    - Exhaustive generative search
    - More parameters to be estimation that leads to more uncertainty
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." arXiv preprint arXiv:1703.01780 (2017).
  - Improves Temporal Ensemble by average model weights (usual practice now!) instead of label prediction (WOW!)
    - Temporal and Pi model suffers from confirmation bias (requires better target) as self-teacher!
  - Two ways to improve: chose careful perturbation or chose careful teacher model  
  - Result: Mean teacher is better! faster converge and higher accuracy
  - Importance of good architecture (ThisPaper: Residual networks):
  - TP: how to form better teacher model from students.
  - TP: Large batch, large dataset, on-line learning.

1. Laine, Samuli, and Timo Aila. "Temporal ensembling for semi-supervised learning." arXiv preprint arXiv:1610.02242 (2016).
  - A form of consistency regularization.
  - Self-ensemble (Exponential moving average), consensus prediction of unknown labels!!
    - Ensemble Utilization of outputs of different network-in-training (same network: different epochs, different regularization!!, and input augmentations) to predict unlabeled data.
    - The predicted unlabeled data can be used to train another network
  - One point to put a distinction between semi-supervised learning and representation learning-fine tuning. In Semi-sup the methods uses the label from the beginning.
  - Importance on the Data Augmentation and Regularization.
  - TP: Two self-ensemble methods: pi-model and temporal ensemble (Figure 1) based on consistency losses.
  - Pi Model: Forces the embedding to be together (Contrastive parts is in the softmax portion; prevents collapse)
  - Pi vs Temporal model:
    - (Benefit of temporal) Temporal model is faster.In case of temporal, training target is less noisy.
    - (Downside of temporal) Store auxiliary data! Memory mapped file.  

1. Assran, Mahmoud, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Armand Joulin, Nicolas Ballas, and Michael Rabbat. "Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples." arXiv preprint arXiv:2104.13963 (2021).
  - PAWS (Predicting view assignments with support samples)
  - Minimize a consistency loss!! different view to get same pseudo labels
  - RQ: can we leverage the labeled data throughout training while also building on advances in self-supervised learning?
  - How it is different than augmentation (may be using some unlabeled counterparts)
  - How the heck the distance between view representation and labeled representation is used to provide weights over class labels (why is makes sense, and what benefits it offers??)
  - Related works: Semi-supervised learning, few shot learning, and self-supervised learning
  - Interesting ways to stop the collapse [sharpening functions] (section 3.2)
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Sohn, Kihyuk, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. "Fixmatch: Simplifying semi-supervised learning with consistency and confidence." arXiv preprint arXiv:2001.07685 (2020).

1. Pham, Hieu, Zihang Dai, Qizhe Xie, Minh-Thang Luong, and Quoc V. Le. "Meta pseudo labels." arXiv preprint arXiv:2003.10580 (2020).
  - semi supervised learning (should cover the labeled and unlabeled at the beginning)
  - teacher network - to generate - pseudo labels [extension of pseudo label work]
    - TP: Contrary to the Pseudo label work: *The teacher network is not fixed* and constantly adapting [claim: teacher learn better pseudo labels]
    - Connected to Pseudo Labels/self-training (semi-supervised)
  - Issues with the confirmation bias of pseudo label? does TP solve this??
    - correct the confirmation bias using a systematic mechanism!!! (how pseudo-label affect the student network) [figure 1]
    - Parallel training [student's learning feedback goes to teacher]!! (dynamic target!)
  - assumption: Pseudo label of teacher can be adjusted
    - However, extremely complication optimization as it requires to unroll everything!
  - Sampling hard pseudo labels (modified version of REINFORCE algorithm!)
  - Nice experiment section: Dataset, network, and baseline
  - [Github Notes](https://github.com/mxahan/PDFS_notes/blob/master/papers/Papers.pdf)

1. Miyato, Takeru, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. "Virtual adversarial training: a regularization method for supervised and semi-supervised learning." IEEE transactions on pattern analysis and machine intelligence 41, no. 8 (2018): 1979-1993.
  - Find the distortion in allowed range for a given input to maximize the loss. (gradient ascent for the input spaces)
    - Network trains to minimize the loss on the distorted input. (Gradient descent in the network parameter spaces)

1. Li, Junnan, Richard Socher, and Steven CH Hoi. "Dividemix: Learning with noisy labels as semi-supervised learning." arXiv preprint arXiv:2002.07394 (2020).
  - TP:  divide the training data into a labeled set with clean samples and an unlabeled set with noisy samples (co-training two networks), and trains the model on both data (?). Improved MixMatch
  - TP: Two diverged network (avoid confirmation bias of self-training) use each others data! GMM to find labeled and unlabeled (too much noisy) data.  Ensemble for the unlabeled.
  - Related strategy: MixUp (noisy sample contribute less to loss!), co-teaching?  Loss correction approach? Semi-supervised learning, MixMatch (unifies SSL and LNL [consistency regularization, entropy minimization, and MixUp])
  - Application: Data with noisy label (social media image with tag!) may result poor generalization (as may overfit)!
  - Hypothesis: DNNs tend to learn simple patterns first before fitting label noise Therefore, many methods treat samples with small loss as clean ones (discards the sample labels that are highly likely to be noisy! and leverages them as unlabeled data)
  - Algorithm is nice to work with

1. Berthelot, David, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin Raffel. "Mixmatch: A holistic approach to semi-supervised learning." arXiv preprint arXiv:1905.02249 (2019).
  - TP: guesses low-entropy labels for data-augmented unlabeled examples and mixes labeled and unlabeled data using MixUp (Algorithm 1)
  - Related works: Consistency Regularization, Entropy Minimization

1. Xie, Qizhe, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. "Self-training with noisy student improves imagenet classification." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10687-10698. 2020.
  - Interesting way to improve the Classifier
  - (labeled data) -> Build classifier (T) -> (predict unlabeled data) -> Train Student using both labeled + model predicted unlabeled data. Repeat.. [algo 1]
  - Introduce noise for both T and S.
    - Data noise, model noise (dropout)

1
