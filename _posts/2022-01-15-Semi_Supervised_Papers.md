---
tags: Papers
---

# Semi-Supervised

## 2022

1. Yang, F., Wu, K., Zhang, S., Jiang, G., Liu, Y., Zheng, F., ... & Zeng, L. (2022). Class-aware contrastive semi-supervised learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 14421-14430).
   -  Confirmation Bias of Pseudo-labeling based approaches and worsening by out-of-distribution data
   - TP: Joint optimization for three losses (sup, modified pseudo-label and modified CL)
   - TP:  Class-aware Contrastive Semi-Supervised Learning (CCSSL): a drop-in helper to improve the pseudo-label quality and robustness 
     - separately handles reliable ID data with class-wise clustering (downstream tasks) and noisy OOD data with instance contrastive
     -  applying target re-weighting to emphasize clean label learning and simultaneously reduce noisy label learning
       - The confidence values between two examples is the weight.
     - Threshold based data selection for the re-training
   - Framework: Data Augmentation, Encoder, Semi-Supervised Module (Pseudo-label based approach), Class-Aware CL
     - Class-aware CL: SCL and reweight. 
       - Key modification  in equation 8: takes multiple positives from the same-class set and high confidency unlabeled data. 
2. Verma, V., Kawaguchi, K., Lamb, A., Kannala, J., Solin, A., Bengio, Y., & Lopez-Paz, D. (2022). Interpolation consistency training for semi-supervised learning. *Neural Networks*, *145*, 90-106.
   - Interpolation Consistency Training (ICT), a simple and computation efficient algorithm for semi-supervised learning
     - encourages the prediction at an interpolation of unlabeled points to be consistent with the interpolation of the predictions at those points. 
     - MixUp for the unlabeled data!!
     - Figure 2 summarizes the approaches. 
3. Xia, J., Tan, C., Wu, L., Xu, Y., & Li, S. Z. (2022, May). OT Cleaner: Label Correction as Optimal Transport. In *ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 3953-3957). IEEE.
   - methods to fix the noisy label problem
     - Finds the clean labels and re-configure the labels with low confidences. 
     - matches the distribution via SK algorithm
4. Xu, Y., Wei, F., Sun, X., Yang, C., Shen, Y., Dai, B., ... & Lin, S. (2022). Cross-model pseudo-labeling for semi-supervised action recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2959-2968).
   - Swapped prediciton approaches utilizing two network primary (F) and auxiliary (A)
   - Very easy setting and loss [equation 6]
     - Two different augmentation and swapped prediction using two different network
     - Supervised loss for the labeled components 
     - Unsupervised loss as shown in the figure below for the unlabeled counter parts
       - Pseudo-labeling using two different network and swapped prediction. 
   - ![name](https://i.ytimg.com/vi/f4dVNFbtEx4/maxresdefault.jpg)

## 2021

1.  Englesson, E., & Azizpour, H. (2021). Consistency Regularization Can Improve Robustness to Label Noise. *arXiv preprint arXiv:2110.01242*.
   - consistency loss propose in Equation 1
     - Weighted sum of
       - JS divergence between two prediction of two augmentation of same image
       - JS divergence of true prediction and average prediction for the augmented images. 
1.  Tai, K. S., Bailis, P. D., & Valiant, G. (2021, July). Sinkhorn label allocation: Semi-supervised classification via annealed self-training. In *International Conference on Machine Learning* (pp. 10065-10075). PMLR.
   - Self-training: learner’s own predictions on unlabeled data are used as supervision during training (iterative bootstrapping)
   - Provides a formulation to facilitate a practical annealing strategy for label assignment and allows for the inclusion of prior knowledge on class
   - Related work: FixMatch (th for selecting pseudo-label)
   - **TP**: Sinkhorn label allocation (SLA). 
     - Minimal computational overhead in the inner loop for label assignment. 
   - ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAb4AAABxCAMAAACdmjYOAAABaFBMVEX///8YKlfx8fH3+v4AAAD5+fnobk7nYUnKNTYmTInSPTrpeVH+/tL998bjU0TyunTvrGjo8PceOW/AMTT78Lfun2L7667Y5O/1zIJqGSftmWC9vb0aGhqJiYnJyckcJ0UAB0OoqKh6enqWlpYAGE0AADI/c6c3Z58DHks9RmEAEUuusbcqN1cwXJfJ2eqyzN7yY0ttn8NTibXZ2doAGk7m5uajpq48PDyTk5OqVT7S0tJrVTW7VSygAADOzqfxtJSHkKRiZFBPDhNXGxhbOiJlIhlsbG2Dl7iZprmJIyW0rIMbMmSoiobchnP//9jWYVWcLCvyuIzdbUyuraC0eUv44KI4NCKvkl1MERyfgoOFRkT10IRgAyNxcXFXY4EnSn9whqGZudI7WoFbW1tMTEwyMjIuOFSwYlB2AAAAABN6gZAAACgAAEgAADqsnZNoc40/IwhtWFhgMjEzAAWsa0UxVoOqRjOGWjzXdvuzAAAL2UlEQVR4nO2di4PbtB2AdaoFDBhbuw4YOmn3k0q5NowNdNoJ7dW9TLd2sA3YYJsL1Dk2WNkT2L8/KbZzkXNxlFyc+Fp9vSaOfb7Y+qy3JSOUSDwMKI20f8eTn1nCj3prR5SIh1FJuV/IEZKBMKLMzK8hvt3jSkRhpEZMl8IyBSByJCTQ3IhcIwxjawBoSZlQx0pSI8iY0V0fcGIWog0wTQmIExf7KAUkAFNtudNn0DGUVLttnCEj0IhLJHZ9wIlZSqWM0IClcB4xtQUuQIPmTp9kOCcAWlrJBTECC5X0DQ2gSBFNtELWFWKIBv+BYOziJbYYsCYElNauhAO+kKN2fbyJRCKxZSztC7vrUxsCmqBp7XiyvAjctXEx+6wv9tc6Hj19aTUKXMwWAlcDI8j4qjPxyw0EUXW6jKU7PXzW7ktZL5D7+8vMFY8Qq5dPr0iCWpG52jT4Ii9oAQDMVZ3dmyvliVzljDMA45SdsEJwCbktITc2p2tUn/fXi7TLIevpM1jIEZO+qYAagrjJcW5UAVTgnMlClnosWK5dAIxcUIytEOtdtlvC1bdAAFJQujftm7BctBO8sD72MSQJWBj7ihhVdrzG3x+ePpDuyhxB7t4I0jJXhmEQTphvKkBM+VOlYxAjzS1DwvR1AhvBVY5VLinTuZZOHwegzEqpS+6imqs3YyjxCRb+nApYo7TQqQ8vo2PndfVRw5mwhRYanBlmBTdcMi2Yj3GF16eo5ZPQ4LZQLjzW+p4toZFVrqqsEabEF12sQpr7Dz4z8FVmrLn7HVendpnfGtXnTn0fP7g5y8ENrCsODiYrPusIujX1uSvUnS/HSFNfWiFU48n5urydc597EDz5RyevWttHu4Dbqe/GXsDhPXSp4qhacXPz+jrwuX0i5ALpS8yT9PUCWVps6CT6ewakD5NO5opN4dZVv61f3v9hyI9C/vrjkC/DzX+L/p7h6OOfvNgJD7Cf/j3Y+smwOv7vPhXyRMhv3nx6ljffejLgt9HfMyB9B3udXGq+fAK6fS3YenCx9T3Rh76PHxxMOJrqq9iNvktJ3xyd+uhzFbdqXXv3b1TsJX3dDEKfRWcG1ZSkbxHD0Hd2UCV9S0n6kr4Ikr5eSPqSvgiSvl4YhL6mA2pO31HFQdK3gEHo+/RWxb0PaprAunV7whsdYZb07VzfjWtVLPvgOzVNYL1S09FXmvTtXl8dNo2+aexL+pZw97El+kLeCrZeTH2fHQUctvWFvz5sff/4XciTob63fhLy+5DXo79nQPrqDLXhXstfuPX2P8PNM/rw9KX9tj3e/mrIky2eCXnhmwHvXFr+DRUD0tcK42k7ec21kJbcGX25f6lv4C1mVm2TR1CfDjO3tr5uvD6GGBUlEVzm5FiCZoIwawQqzJp3Y5N1hx4MW9/hYPUB5lZgKJVBwhhGWSELoKvcTK/gdJnydcd9D0rf/dcqrjf1vyYD2mi979z6clyCplYILdQYC7BUC6ldLNSxiScWhR2BZNhHXIGlACCMkYKteC6D0nf04XcrPrpec+1aD60u59anqfZDgBUGhS0QRClSaMSRorH3LwN2sU/BGOeU5IAE1QwMg2LFUxmqvpdqfXWYbbbN89z6zg9YOxqV6FhrQy0Fp0+MrFKrZ51J3y70IZdYai45kuAW3SFhjkCuMX1N0rcZfXg3wzqTvnX0YTUZtqO0X1KIKGJpe2DnVkj61tCHBQcJknKhj+mYQsGFdT+rlzzOzTD1XR+2Pl64PCtDVpaKIYOkQUD5MSwp9lNYiaiTGZK+ww8/qnmpZqD6SOFKi1owWlqB/JRdo8JSRpcUHDOxCllUZjokfXuHdTtjO0CHps+P7iTunx/X6ScvqMa9LwvvrHtoTIirVMSczNtPh3Tbm9MXHWirjHGYY3j6QuJq69kqo7lIlD51/NOQn7X4ecgvQv4V3VQ3IH2Xw89L9B2GHRCfrV/v60EfZL8M+UE3LdlZdA/JgPTdDP/4la+EtOx9+FzAKwPT972Qr7X4esi7V5+d4eofLqK+1s0S3fqObg1c3+OzPLNM37NJ37o8ZPrIdLrqNfRNC+kXWR9tFvjcmkHqG4+MVMxo5nuntaR15bQennl0sIDJvC4lSGndvpJxJjRr9t2pPgoqfrqetj4CqlmEZpOlQ9bHnARKmQWEgXEo64YKVcFUG1u/43pfoEwbRMBwU9T77lIfIJ77Ke+XhTRMahZzsY9NV4B/8X2IRJAIfby5ZHrX10ygVekTSABnkgspmaASRmb2d7MF513B3L6U+R8Qxr9W+0bqI9MHDWxQH0ZyEpDLGrjybN99faOPllx4WQqINZT7mOf/c54LTthUX8dda5BlTFULPesTWWb8dTTN+0j9Q+oPM8zrOwmbEc/aN1ZflpXVNbvRvG+MXdSjVUThWSflfm1GF6jUEpQFol1iourEU2uBMDGNvnH3X8uyfbsJfUu/xuESguUzCnKa0Va5QGfzRue87KMy4hAq5Gb1neAM+ecW1GnIwgYwd4BCTWMfB1xyI62LfSRHZZP3uVxU4ZnYpxc3qEFzLZ5f35KmOxf7Cp9+nupbdE+cO6aslQrJLFvaieb0xbQh4iw7gUl6tEF9lOuRi3lsWd5XJbG1PiIlYKkKn/cRoDmp8z6XHfDZvK8j8aSAmzDrOfGUdS470Yd9Ky+3LvUjvrfTvczMy+SSt3Zcc2uCPjTinwXg955pZYxNPEWzy+YrDuUo6hCa2FdiQiilxGXHdAS8iX3+GnNRMKboMmWrJU8FjI6E5VLTEoTQpbQFnMbFst0Ep45Pjo9nVxSUcWm4BHM6oe8A6n2R80PX+jRVdTHT1fKoIkF1bzTgeh/44qNLIkBTiYFa905n5him2VzQsHCN71wT1EiJzTSZHYC+SJrYF9X4MkB9mDFljKvfGZd7U66o5ozNXLnz5RQR1okNAkmxv7dOXmB9UfTTZP3s1YANtXnWzBdTIqZd7l3f5dDfwPSdj83oG32/xaih/tzRH3pefYf33gi5E3IvHC92a1D6yL9/FfLHFr8O+c97Af+NvruxU9/ezaBL9OadZpB0veJBR9HuvPqu3b50eYZLl6+E3Blydy16/4lvBDz/5+dneTdMLK/+6VsB70Ufe6e+l8MEa+9O+1alXvVdDmnrCw/tHHdZ93Gvy5y+55O+vvTJVYi70yzp25o+udKjmeIe+5H0bU1fHyR9SV8E3fpa4/+TvlgGoa+e+vhGEwuTvlgGoa+plS+6YzbpW8Qw9C25Xz3pW0TSl/RFkPT1QtKX9EWQ9PXCfJN1wJy+vwQ8FPqC4X7zPQ6D1nc37G1/vDU761OPhbwQ8nn0TGpLKg4L9B1WHPSoz9UyQ86+gi6Ivse6aenb1ODolys+qDu8rzSB9WrFa9ucijzpO4OoRrMrC/S9mvQtYBj69pK+9Uj6kr4Ikr5eSPqSvgiSvl5I+pK+CKL0fVHPRPdSU4W+X/FFr9X2pG85Ufqaqchfa1qsmjBL+hYwSH3XW/r6bPNM+mJI+noh6Uv6IhiQvgcPkb73vwx5vZvP3/l8hnf+t5kOo63qsy9++xy8OCx922I4+hJrsFK9736r6zTp2zVxYxzq3vV2hpP07ZqVhqgkfUMj6ds8MOKLClL6rFnyrHYbnAc9M+EDjSt8Jn2bhuQYKblo61lPdPFKaUsfiXvcR9K3aXLV9eArgRT3TxTkFmGuJr6422GyNNGn67Vm0V8ISPo2TTVdmHZR0M94RD3IC3WQagIqUKWPhVYQJ5Mi0EhrHyu9PqygempW0rcTyNi9cD+Rr4+BU32ac06x06cNotRM5t0VqCRAsZ+GEHtbVbTzs4Hrzeg77GajHUYPC4ZjjrVA1k9ZW00Zc7qRlEhgRiT4Z7UK5dTmyD+adVLW8bGPQ661NRvRt3yWk459H1V9LhK5cGEYnfXgMuISUEU0uFiJwJU4AYMvpZDJFHJV3kf0ZE7shYWfgJg5zdbi0dU3gWO1OGki4BNSP5Mp9rGO2ur54qopeTqPNG4yy6SvJ9QKT3xvpv8mc6uWMT7n3HeLGccffiLxfwnGsI8DOQTkAAAAAElFTkSuQmCC)
   - Label annealing strategies where the labeled set is slowly grown over time
   - Provide fast approximation: the typical solution we see. (algorithm 1 and 2)
   - <embed src="https://mxahan.github.io/PDF_files/sinkhorn_knopp_label_allocation.pdf" width="100%" height="850px"/>
1.  Tai, K. S., Bailis, P. D., & Valiant, G. (2021, July). Sinkhorn label allocation: Semi-supervised classification via annealed self-training. In International Conference on Machine Learning (pp. 10065-10075). PMLR.
1. Assran, Mahmoud, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Armand Joulin, Nicolas Ballas, and Michael Rabbat. "Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples." arXiv preprint arXiv:2104.13963 (2021).

     - PAWS (Predicting view assignments with support samples)

     - Minimize a consistency loss!! different view to get same pseudo labels

     - RQ: can we leverage the labeled data throughout training while also building on advances in self-supervised learning?

     - How it is different than augmentation (may be using some unlabeled counterparts)

     - How the heck the distance between view representation and labeled representation is used to provide weights over class labels (why is makes sense, and what benefits it offers??)

     - Related works: Semi-supervised learning, few shot learning, and self-supervised learning

     - Interesting ways to stop the collapse [sharpening functions] (section 3.2)

  -  <embed src="https://mxahan.github.io/PDF_files/PAWs.pdf" width="100%" height="850px"/>

## 2020

1. Wang, Y., Guo, J., Song, S., & Huang, G. (2020). Meta-semi: A meta-learning approach for semi-supervised learning. *arXiv preprint arXiv:2007.02394*.

     - section 2: methods contain the gist. 

2. Yu, Q., Ikami, D., Irie, G., & Aizawa, K. (2020). Multi-task curriculum framework for open-set semi-supervised learning. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16* (pp. 438-454). Springer International Publishing.

     - Assumption: Labeled data and unlabeled data, some common classes in unlabeled data and rest is from novel classes (GCD setting)
     - Instead of training an OOD detector and SSL separately, TP propose a multitask curriculum learning framework
       - OOD detection:  estimate the probability of the sample belonging to OOD
       - a joint optimization framework, which updates the network parameters and the OOD score alternately
       - to improve performance on the classification (ID) data, TP  select ID samples in unlabeled data having small OOD scores, and retrain training the deep neural networks to classify ID samples in a semi-supervised manner.
     - **Key assumption:**  that a network trained with a high learning rate is less likely to overfit to noisy labels
       - train OOD detection and hope that the noisy samples will be filtered automatically.
     - As simple as heck: Select only top confidence sample from the unlabeled data for retraining the labeled classifier. (curriculum)
       - Algorithm 1 and 2

3. Guo, L. Z., Zhang, Z. Y., Jiang, Y., Li, Y. F., & Zhou, Z. H. (2020, November). Safe deep semi-supervised learning for unseen-class unlabeled data. In *International Conference on Machine Learning* (pp. 3897-3906). PMLR.

     -  unlabeled data contains some classes not seen in the labeled data.
       - TP: proposes a simple and effective safe deep SSL method to alleviate the harm caused by it
     - Kinda NCD formulation
     - Proposes instance to weight weighting funciton!! multiplied with consistency regularization.  
       - What and how and why, is that scalable?? how to do it for image?
       - Vague one, safely ignore the paper!
     - Safe (!)
     - Bi-level optimization problem
       - supervised and consistency loss regularization

4. Arazo, E., Ortego, D., Albert, P., O’Connor, N. E., & McGuinness, K. (2020, July). Pseudo-labeling and confirmation bias in deep semi-supervised learning. In *2020 International Joint Conference on Neural Networks (IJCNN)* (pp. 1-8). IEEE.

     - Soft pseudo-label (with correct setting can outperform consistency regularization: Noise Accumulation!)
       - TP: Tries to eliminate the CB without Consistency regularization
         - Drop-out and data augmentation can eliminate CB
       - Naive pseudo-label overfits to incorrect one due to confirmation bias (how to measure CB). 
       - Mixup and setting a minimum number of labeled samples per mini-batch are effective regularization techniques reduces confirmation bias
       - They utilize external KL matching. 
       - label smoothing alleviate the overconfidence problem. 
       - Mixup data augmentation alone is insufficient to deal with CB when few labeled examples are provided
     - Sum of three losses: Entropy loss (enforce single decision), distribution (removing collapse) matching and mixup (counter CB) loss 
       - Relative weight between them matters

5. Van Gansbeke, W., Vandenhende, S., Georgoulis, S., Proesmans, M., & Van Gool, L. (2020, August). Scan: Learning to classify images without labels. In European conference on computer vision (pp. 268-285). Springer, Cham.

     - advocate a two-step approach where feature learning and clustering are decoupled
       - SSL and SSL prior for learnable clustering.
       - remove the ability for cluster learning to depend on low-level features (current end-to-end learning approaches.)

     - Representation **learning  leads to imbalanced clusters and there is no guarantee that the learned clusters aligns with the semantic classes**

         - end-to-end learning pipelines combine feature learning with clustering.
           - leverage the architecture of CNNs as a prior to cluster images (DEC)
           - learn a clustering function by maximizing the mutual information between an image and its augmentations
             - sensitive to initialization or prone to degenerate solutions
             - since the cluster learning depends on the network initialization, they are likely to latch onto low-level features

         - [TP]: SCAN (Semantic Clustering by Adopting Nearest neighbors) - 2-steps
           - *leaverage advantage of both representation learning and end-end learning*
           - encourage invariance w.r.t. the nearest neighbors, and not solely w.r.t. augmentations

      - Methods summary: 1. Representation learning for semantic clustering,
        - Contrastive learning (loss function 2) with entropy regularizer.
          - Fine-tuning through self-labeling

6. Sohn, Kihyuk, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. "Fixmatch: Simplifying semi-supervised learning with consistency and confidence." arXiv preprint arXiv:2001.07685 (2020).
   - FixMatch: a significant simplification of existing SSL methods
     - Generates pseudo-labels using the model’s predictions on weakly augmented unlabeled images
       - Retained if confidence is high. 
       - Utilize the retained label to train strongly-augmented (via Cutout, CTAugment, RandAugment) version of same image. 
     - *Consistency Regularization* and *Pseudo-labeling*
   - ![](https://miro.medium.com/max/1077/1*5SCSOqvXcrxL-IwZmZaH_g.png)
   - Simple setup- retraining with the good predictions with consistency losses. 

7. Pham, Hieu, Zihang Dai, Qizhe Xie, Minh-Thang Luong, and Quoc V. Le. "Meta pseudo labels." arXiv preprint arXiv:2003.10580 (2020).

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

   - <embed src="https://mxahan.github.io/PDF_files/Meta_pseudo_label.pdf" width="100%" height="850px"/>

8. Li, Junnan, Richard Socher, and Steven CH Hoi. "Dividemix: Learning with noisy labels as semi-supervised learning." arXiv preprint arXiv:2002.07394 (2020).

   - TP:  divide the training data into a labeled set with clean samples and an unlabeled set with noisy samples (co-training two networks), and trains the model on both data (?). Improved MixMatch

   - TP: Two diverged network (avoid confirmation bias of self-training) use each others data! GMM to find labeled and unlabeled (too much noisy) data.  Ensemble for the unlabeled.

   - Related strategy: MixUp (noisy sample contribute less to loss!), co-teaching?  Loss correction approach? Semi-supervised learning, MixMatch (unifies SSL and LNL [consistency regularization, entropy minimization, and MixUp])

   - Application: Data with noisy label (social media image with tag!) may result poor generalization (as may overfit)!

   - Hypothesis: DNNs tend to learn simple patterns first before fitting label noise Therefore, many methods treat samples with small loss as clean ones (discards the sample labels that are highly likely to be noisy! and leverages them as unlabeled data)

   - Algorithm is nice to work with

   - <embed src="https://mxahan.github.io/PDF_files/DivideMix.pdf" width="100%" height="850px"/>

9. Xie, Qizhe, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. "Self-training with noisy student improves imagenet classification." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10687-10698. 2020.

     - Interesting way to improve the Classifier

     - (labeled data) -> Build classifier (T) -> (predict unlabeled data) -> Train Student using both labeled + model predicted unlabeled data. Repeat.. [algo 1]

     - Introduce noise for both T and S.
       - Data noise, model noise (dropout)


## 2019

1. Zhai, Xiaohua, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. "S4l: Self-supervised semi-supervised learning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1476-1485. 2019.

     - Pretext task of rotation angle prediction!!
       - Rotation, invariant across augmentation

     - Baseline: vitrural adversarial training [inject noise with the original images], EntMin
       - EntMin is bad: because the model can easily become extremely confident by increasing the weights of the last layer

1. Gupta, Divam, Ramachandran Ramjee, Nipun Kwatra, and Muthian Sivathanu. "Unsupervised Clustering using Pseudo-semi-supervised Learning." In International Conference on Learning Representations. 2019.

1. Berthelot, David, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin Raffel. "Mixmatch: A holistic approach to semi-supervised learning." arXiv preprint arXiv:1905.02249 (2019).

     - TP: guesses low-entropy labels for data-augmented unlabeled examples and mixes labeled and unlabeled data using MixUp (Algorithm 1)

     - Related works: Consistency Regularization, Entropy Minimization

  - <embed src="https://mxahan.github.io/PDF_files/mixmatch.pdf" width="100%" height="850px"/>

## 2018 and earlier

1. Qiao, S., Shen, W., Zhang, Z., Wang, B., & Yuille, A. (2018). Deep co-training for semi-supervised image recognition. In *Proceedings of the european conference on computer vision (eccv)* (pp. 135-152).

     - Joint optimization of three losses in semi-supervised setting
     - Utilize two different network for co-training (DCT)
     - Cross entropy loss for labeled data
     - JS divergence minimization for two views prediction from of two network
     - Swapped prediction using adversarial examples utilizing two different network.  

1. Tanaka, D., Ikami, D., Yamasaki, T., & Aizawa, K. (2018). Joint optimization framework for learning with noisy labels. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 5552-5560)

     - Alternative optimization: generate pseudo-label and update network, repeat... 
       - Suggests high learning rate
       - Issues with CB
     -  reinforces the findings of Arpit et al. [1] that suggest that DNNs first learns simple patterns and subsequently memorize noisy data

1. Chang, J., Wang, L., Meng, G., Xiang, S., & Pan, C. (2017). Deep adaptive image clustering. In Proceedings of the IEEE international conference on computer vision (pp. 5879-5887).

     - adopts a binary pairwise classification framework for image clustering [DAC]
         - learned label features tend to be one-hot vectors (constraint into DAC)
             - single-stage named Adaptive Learning algorithm to  streamline the learning procedure for image clustering
               - dynamically learn the threshold for pairwise label selection [section 3.3]
     - *summary:* chose two adaptive th for pairwise labels. iterative fix the network and th to reduce number of unlabeled images.  
       - *metrics:* Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) and clustering Accuracy (ACC).

1. Rasmus, A., Berglund, M., Honkala, M., Valpola, H., & Raiko, T. (2015). Semi-supervised learning with ladder networks. Advances in neural information processing systems, 28.

     - trained to simultaneously minimize the sum of sup and unsup cost functions by backpropagation, avoiding the need for layer-wise pre-training
       - builds on top of the Ladder network [bit out of my interest for now]

1. Blum, Avrim, and Tom Mitchell. "Combining labeled and unlabeled data with co-training." In Proceedings of the eleventh annual conference on Computational learning theory, pp. 92-100. 1998.
     - Early semi-supervised learning (mathematical framework with graph setting)
       - Web page (and its augmentation)

1. Grandvalet, Yves, and Yoshua Bengio. "Semi-supervised learning by entropy minimization." In CAP, pp. 281-296. 2005.

     - Semi-supervised learning by minimum entropy regularization!
       - result compared with mixture models ! (entropy methods are better)
       - Connected to cluster assumption and manifold learning

     - Motivation behind supervised training for unlabeled data
       - Exhaustive generative search
         - More parameters to be estimation that leads to more uncertainty

     - <embed src="https://mxahan.github.io/PDF_files/Minimum_entropy_reg.pdf" width="100%" height="850px"/>

1. Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." arXiv preprint arXiv:1703.01780 (2017).

     - Improves Temporal Ensemble by average model weights (usual practice now!) instead of label prediction (WOW!)
          - Temporal and Pi model suffers from confirmation bias (requires better target) as self-teacher!

     - Two ways to improve: chose careful perturbation or chose careful teacher model

     - Result: Mean teacher is better! faster converge and higher accuracy

     - Importance of good architecture (TP: Residual networks):

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

1. Miyato, Takeru, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. "Virtual adversarial training: a regularization method for supervised and semi-supervised learning." IEEE transactions on pattern analysis and machine intelligence 41, no. 8 (2018): 1979-1993.

     - Find the distortion in allowed range for a given input to maximize the loss. (gradient ascent for the input spaces)
        - Network trains to minimize the loss on the distorted input. (Gradient descent in the network parameter spaces)
