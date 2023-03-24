# Introduction

This blog contains regularly updated paperlist (generlized but top-notch) of my personal interest with minimal summary.

Any suggestion regarding some new papers are highly appreciated. For some of the detail paper read please refer to the Paper summary section.

We can't worry about everything and nothing. We must stay focused and broad at the same time. Very true! I know the Feynman's thoughts on knowing name, which means nothing, it's just the entrance to the grand scheme of the world. however, the papers (many papers!) are as follows

## 2022
1. Soviany, P., Ionescu, R. T., Rota, P., & Sebe, N. (2022). Curriculum learning: A survey. *International Journal of Computer Vision*, *130*(6), 1526-1565.
   - learning models in a meaningful order, from the easy samples to the hard ones instead of random shuffling.
   - Aim: finding a way to rank the samples from easy to hard,  the right pacing function for introducing more difficult data
   - Opposite: anti-curriculum learning: hard negative mining
   - Four main component of ML: Data, model, task, and the performance measurement. 
   - Division based on: Data type, task, curriculum strategy, ranking criterion, and curriculum schedule
   - Taxonomy of CuL: 
     - Vanilla CuL: Rule base criterion for sample selection
     - self-paced learning: The sample depends on the model itself. Model decides whats easy and whats hard. 
     - Balanced CuL: Select data from all action groups: Focus on intra group hard and easy instances. 
     - Self-paced CuL: predefined criteria and learning-based metrics are jointly used. 
     - Progressive CuL: progressive mutation of the model capacity or task settings
     - Teacher-student CuL: splits the training into two tasks, a model that learns the principal task (student) and an auxiliary model (teacher) that determines the optimal learning parameters (policy) for the student. 
     - Implicit CuL:  the easy-to-hard schedule can be regarded as a side effect of a specific training methodology!
1. Lee, Yoonho, Huaxiu Yao, and Chelsea Finn. "Diversify and Disambiguate: Learning From Underspecified Data." arXiv preprint arXiv:2202.03418 (2022).
1. Zhao, B., Cui, Q., Song, R., Qiu, Y., & Liang, J. (2022). Decoupled knowledge distillation. In *Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition* (pp. 11953-11962).
   - TP: Study logit distillation (provide novel insight) instead of the deep embedding features. 
   - The class overlaps so not the NCD/continual setting
   - Figure 1 Summarizes all
   - TCKD transfers the knowledge concerning the “difficulty” of training samples: transfers “dark knowledge” via the binary classification task, related to the sample “difficulty”
   - NCKD is the prominent reason why logit distillation works but is greatly suppressed:  only NCKD is applied, the performances are comparable or even better than the classical KD
     - Well **Connected to Negative Learning**
     - All about complementary labels. 
     - the more confident the teacher in a sample, the more reliable and valuable knowledge it could provide. However, the loss weights are highly suppressed by this confident predictions. *Hypothesis:* this fact would limit the effectiveness of knowledge transfer.
   - Experimentation: Image and video datasets
   - However, really interesting formulation: See notes

## 2021

1. Gao, Ruohan, and Kristen Grauman. "VisualVoice: Audio-Visual Speech Separation with Cross-Modal Consistency." arXiv preprint arXiv:2101.03149 (2021).
   
1. Brock, Andrew, Soham De, Samuel L. Smith, and Karen Simonyan. "High-Performance Large-Scale Image Recognition Without Normalization." arXiv preprint arXiv:2102.06171 (2021).
  - Proposes alternative to BN [BN suffers instabilities] - Normalization freee models
    - Adaptive gradient clipping (AGC)
    - Experiment: Normalization free ResNet
  - Section 3: reasons and alternative to BN
  - Section 4: This papers key contribution (AGC)
    - Hypothesis: 1<sup>st</sup> paragraph: accelerating converge in poorly conditioned loss function for large Batch Size.
    - Key point in equation 3 (adaptive)

## 2020

1. Yao, A., & Sun, D. (2020). Knowledge transfer via dense cross-layer mutual-distillation. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16* (pp. 294-311). Springer International Publishing.
     - restrict our focus to advance two-way KT research in the perspective of promoting knowledge representation learning and transfer design.
     - Bidirectional KD between intermediate layers of the identical networks [overview figure 1]
       - Utilizes auxiliary classifier to enable the distillation [similar to google network auxiliary prediction]
       - Also bidirectional KT between different stages
     - Joint optimization of multiple losses: Eq - 6
1. Kavalerov, I., Czaja, W., & Chellappa, R. (2020). A study of quality and diversity in K+ 1 GANs.
1. Lee, Kwot Sin, Ngoc-Trung Tran, and Ngai-Man Cheung. "InfoMax-GAN: Improved Adversarial Image Generation via Information Maximization and Contrastive Learning." arXiv preprint arXiv:2007.04589 (2020).
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
1. Gururangan, Suchin, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." arXiv preprint arXiv:2004.10964 (2020).
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

  - <embed src="https://mxahan.github.io/PDF_files/SAM_p.pdf" width="100%" height="850px"/>

1. Tan, Hao, and Mohit Bansal. "Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision." arXiv preprint arXiv:2010.06775 (2020).

     - Sentence to hidden features to probability distribution.  

     - Contextual visual representation

     - LM architecture with additional voken objective (BERT objective + voken classification)

     - Need revisit!

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

2. Ravfogel, Shauli, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. "Null it out: Guarding protected attributes by iterative nullspace projection." arXiv preprint arXiv:2004.07667 (2020).
  - Under construction

1. Roy, Aurko, Mohammad Saffar, Ashish Vaswani, and David Grangier. "Efficient content-based sparse attention with routing transformers." arXiv preprint arXiv:2003.05997 (2020).
  - Under construction

## 2019

1. Du, Yilun, and Igor Mordatch. "Implicit Generation and Modeling with Energy Based Models." In Advances in Neural Information Processing Systems, pp. 3603-3613. 2019.

1. Grathwohl, Will, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, and Kevin Swersky. "Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One." arXiv preprint arXiv:1912.03263 (2019).


## 2018

1. Tanaka, D., Ikami, D., Yamasaki, T., & Aizawa, K. (2018). Joint optimization framework for learning with noisy labels. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 5552-5560).
     - To avoid overfiting to noisy labels TP proposes a joint optimization framework of learning DNN parameters and estimating true labels.
       -  can correct labels during training by **alternating** update of network parameters and labels
     - experiments on the noisy CIFAR-10 datasets and the Clothing1M dataset.
     - observe (empirically find) that a DNN trained on noisy labeled datasets does not memorize noisy labels and maintains high performance for clean data under a high learning rate
     - Related works: Generalized abilities of DNN, learning on noisy labeled dataset (regularization, noise transition matrix, robust loss function), self-training and pseudo-labeling
     - **very close to our neural distribution works** 
1. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training gans. *Advances in neural information processing systems*, *29*.
     - TP: present a variety of new architectural features and training procedures for GANs framework. 
       - Proposed Feature matching, minibatch features techniques (applicable to SGAN)
       - Proposes virtual batch normalization (extension for batch normalization)
     - achieve state-of-the-art results in semi-supervised classification on MNIST, CIFAR-10 and SVHN
     - Feature Matching: Essential parts to the stability of the SGAN (semi-supervised GAN by Odena et. al) setting 
       - Simple setting and key idea: Match discriminator features section **3.1** 
1. Dai, Z., Yang, Z., Yang, F., Cohen, W. W., & Salakhutdinov, R. R. (2017). Good semi-supervised learning that requires a bad gan. *Advances in neural information processing systems*, *30*.
     - Why Semi-supervised gan works
       - how the discriminator benefits from joint training with a generator
       - why good semi-supervised classification performance and a good generator cannot be obtained at the same time
     - TP: show that given the discriminator objective, good semi-supervised learning requires a bad generator, and propose the definition of a preferred generator.
     - Improved upon feature matching GAN
1. Malach, E., & Shalev-Shwartz, S. (2017). Decoupling" when to update" from" how to update". *Advances in neural information processing systems*, *30*.
     - TP: meta learning algo against noisy label when combined from different sources
     - Experiment with gender classification from facial images
       - experimented with FF networks only
     - Fairly easy idea: Train two networks and update only when there is a mismatch between them at the end of optimization process
       - Assumption: Only few are noisy labeled.
     - multi-agent learning (multi-network training)
     - closely related to approaches for active learning and selective sampling
     - tackle two questions: 1. does this algorithm converge? and if so, how quickly? and 2. does it converge to an optimum?
       - Depends on the initialization of the learners
     - Explicit regularization (drop out, weight decay, data augmentation) may improve generalization performance, but is neither necessary nor by itself sufficient for controlling generalization error.
     - SGD acts as an implicit regularizer. For linear models, SGD always converges to a solution with small norm. 
1. Arpit, D., Jastrzębski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., ... & Lacoste-Julien, S. (2017, July). A closer look at memorization in deep networks. In *International conference on machine learning* (pp. 233-242). PMLR.
     - Examine the role of memorization in deep learning, drawing connections to *capacity*, *generalization*, and *adversarial robustness*
       -  DL tend to prioritize learning simple patterns first (content-aware)
       -  expose qualitative differences in gradient-based optimization behavior of DNNs on noise vs. real data.
         - “memorization” as the behavior exhibited by DNNs trained on noise, and conduct a series of experiments that contrast the learning dynamics of DNNs on real vs. noise data
       - demonstrate that appropriately tuned explicit regularization (e.g., dropout) can degrade DNN training performance on noise datasets without compromising generalization on real data. 
       - training data itself plays an important role in determining the degree of memorization.
     - TP: only examine the representational capacity, that is, the set of hypotheses a model is capable of expressing via some value of its parameters.
1. Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2021). Understanding deep learning (still) requires rethinking generalization. *Communications of the ACM*, *64*(3), 107-115.
     - Deep neural networks easily fit random labels.
     - Experimented with two layer FF neural network
       - model effective capacity: The effective capacity of neural networks is sufficient for memorizing the entire data set
       - 
1. Jiang, L., Zhou, Z., Leung, T., Li, L. J., & Fei-Fei, L. (2018, July). Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In *International conference on machine learning* (pp. 2304-2313). PMLR.
     - KD setting to avoid noisy (corrupted) label.
       - MentorNet provides a curriculum (sample weighting scheme) for StudentNet to focus on the samples with probably correct label
       - Unlike the existing curriculum predefined by human, MentorNet learns a data-driven curriculum dynamically with StudentNet.
     - Experimentation: WebVision Dataset.
     - Related to Curriculum Learning: Gradually learning process. 
1. Blum, A., & Mitchell, T. (1998, July). Combining labeled and unlabeled data with co-training. In *Proceedings of the eleventh annual conference on Computational learning theory* (pp. 92-100).
     - Co-training algorithm: In presence of two distinct views of each example suggests strategies
       - two learning algorithms are trained separately on each view,
       - then each algorithm's predictions on new unlabeled examples are used to enlarge the training set of the other. 
     - assume that either view of the example would be sufficient for learning if we had enough labeled data, but TP use both views together to allow inexpensive unlabeled data to augment a much smaller set of labeled examples.
     - provide a PAC-style analysis for this setting, and, more broadly, a PAC-style framework for the general problem of learning from both labeled and unlabeled data.
1. Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., ... & Sugiyama, M. (2018). Co-teaching: Robust training of deep neural networks with extremely noisy labels. *Advances in neural information processing systems*, *31*.
     - Early for of swapped prediction: train be opposite networks' clean performance. 
       - TP uses two networks to train together. (**Algorithm 1** Tells all)
         - Motivated from co-training algorithm (Blum et. al 1998)
       - Each network feeds forward all data and selects some data of possibly clean labels 
       - two networks communicate with each other what data in this mini-batch should be used for training (cross update)
         - Boosting
       - each network back propagates the data selected by its peer network and updates itself.
     - Related works: decoupled KD, 
     - Experimented with noisy MNIST, CIFAR
     - Hypothesis: NN learns the easy labels first and then go for the complex data (in between we can distinguish the noisy data)
1. Zhang, Y., Xiang, T., Hospedales, T. M., & Lu, H. (2018). Deep mutual learning. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4320-4328).
     - mutual learning starts with a pool of untrained students who simultaneously learn to solve the task together
       -  each student is trained with two losses: a supervised loss, and a mimicry loss (aligns each student’s class posterior with each other) [fig 1]
       - provide better results in this way! (than the distillation or single model learning)
     - Assumption: accuracy is the only concern
     - Related work: Collaborative learning: a different from mutual learning where all models address the same task and domain.
     - Kind of extra regularization for the classification layers [shallow resembles the momentum update.]
       - swapped prediction
       - Can we do it in the embedding layers
1. Yang, H. M., Zhang, X. Y., Yin, F., & Liu, C. L. (2018). Robust classification with convolutional prototype learning. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 3474-3482).
1. Fernandes, Patrick, Miltiadis Allamanis, and Marc Brockschmidt. "Structured neural summarization." arXiv preprint arXiv:1811.01824 (2018).
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

1. Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).

1. Graves, Alex, Greg Wayne, and Ivo Danihelka. "Neural turing machines." arXiv preprint arXiv:1410.5401 (2014).
  - External memory source coupling for attention  (analogous to turing machine )
    - How to show it? what's the background? Experiments? How much they differ and align? Whats the perspective?
  - copying, storing and associative recall
  - Extension of RNN - NTM (differentiable unlike Turing Machine)
  - Two components: controller and memory with i/o for external interaction
    - Attending to memory (attention) [sharp or distributed attention]
  - <embed src="https://mxahan.github.io/PDF_files/Neural_turing_machine.pdf" width="100%" height="850px"/>

1. Zhao, Junbo, Michael Mathieu, and Yann LeCun. "Energy-based generative adversarial network." arXiv preprint arXiv:1609.03126 (2016).li

     - D: Assign low energy to data distribution (not normalized probabilities)

     - G: Sample data from the low energy by a parameterized function.

     - Convergence proof using hinge loss.

  - <embed src="https://mxahan.github.io/PDF_files/Gan_math.pdf" width="100%" height="850px"/>

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
  - Hierarchical generator setup

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

1. Yang, Zhilin, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R. Salakhutdinov, and Quoc V. Le. "Xlnet: Generalized autoregressive pretraining for language understanding." In Advances in neural information processing systems, pp. 5753-5763. 2019.
  - under construction

1. Hu, Baotian, Zhengdong Lu, Hang Li, and Qingcai Chen. "Convolutional neural network architectures for matching natural language sentences." In Advances in neural information processing systems, pp. 2042-2050. 2014.
  - Matching network

1. Liu, Sicong, Junzhao Du, Anshumali Shrivastava, and Lin Zhong. "Privacy Adversarial Network: Representation Learning for Mobile Data Privacy." Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 3, no. 4 (2019): 1-18.
  - presented in my course work instructed by my supervisor Dr. Nirmalya Roy
  - [link](https://docs.google.com/presentation/d/1OF7Y6yoIAuLVQ_OtV5kdCEXzCp3MRAzusg1o8k6kTvo/edit?usp=sharing)

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



1. Peters, Matthew E., Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. "Deep contextualized word representations." arXiv preprint arXiv:1802.05365 (2018).

1. Shen, Yanyao, Hyokun Yun, Zachary C. Lipton, Yakov Kronrod, and Animashree Anandkumar. "Deep active learning for named entity recognition." arXiv preprint arXiv:1707.05928 (2017).

1. Gal, Yarin, Riashat Islam, and Zoubin Ghahramani. "Deep bayesian active learning with image data." arXiv preprint arXiv:1703.02910 (2017).

1. Ganin, Yaroslav, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. "Domain-adversarial training of neural networks." The Journal of Machine Learning Research 17, no. 1 (2016): 2096-2030.   


1. Szegedy, Christian, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. "Rethinking the inception architecture for computer vision." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818-2826. 2016.

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
  - <embed src="https://mxahan.github.io/PDF_files/measuring_invariance_dl.pdf" width="100%" height="850px"/>

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


1. Madry, Aleksander, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).
  - study a set of approaches [the adversarial robustness of neural networks through the lens of robust optimization]
    - (unifies earlier works).
    - provides security against first order adversarial attack.
  - Projected gradient descent.

1. Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).
