# Introduction

This blog contains state of the art application and research on different applications. The applications will be stated in the subheadings. Although one paper can fall into multiple categories we will be mostly going with one tag per papers. However, the tag is subjected to change time-to-times.

# Multi View Application

# Few-Shot learning

1. Wang, Yaqing, Quanming Yao, James T. Kwok, and Lionel M. Ni. "Generalizing from a few examples: A survey on few-shot learning." ACM Computing Surveys (CSUR) 53, no. 3 (2020): 1-34.
  - Three approaches (i) Data (ii) Model (iii) Algorithm, all of them uses prior knowledge!
    - What is prior knowledge
  - Interesting section that distinguish between FSL with other learning problems (2.2)
    - FSL avails various options as prior knowledge. For example: Weak-supervised (semi-supervised, active learning) learning is a kind of FSL (when prior knowledge is the unlabeled data); alternatively, the prior knowledge may be the learned model (aka transfer learning.)
    - Unreliable empirical risk minimization
  - (i) data prior knowledge
    - Data augmentation - Transforming samples from the labeled training data
    - Transforming data from the unlabeled data (semi-supervised)
    - Transforming data from the similar dataset (GAN)
  - (ii) Model
    - Multitask learning
      - Parameter sharing
      - Parameter tying ? Pairwise difference is penalized
    - Embedding Learning
      - Task specific
      - Task invariant: Matching net, prototypical networks
      - hybrid
    - External memory: Key-value memory
      - Refining representation
      - Refining parameters
    - Generative modeling
      - Decomposable components
      - Groupwise Shared Prior
      - Parameters of inference networks
  - (iii) Algorithm
    - Refining existing parameters
      - Regularization
      - aggregation
      - Fine-tuning existing parameter
    - Refining Meta-learned parameters
    - Learning the optimizer
  - Future works: four possible directions :
    - Problem setup
    - Techniques
    - applications
    - Theories
