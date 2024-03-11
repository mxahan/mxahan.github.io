---
tags: Papers
---

# Uncertainty Papers



# 2021



1. Paul, M., Ganguli, S., & Dziugaite, G. K. (2021). Deep learning on a data diet: Finding important examples early in training. *Advances in Neural Information Processing Systems*, *34*, 20596-20607.
   - simple scores (Gradient Normed and Error L2-Norm) averaged over several weight initializations can be used to identify important examples very early in training
   - These scores to detect noisy examples and study training dynamics through the lens of important examples
   - investigate how the data distribution shapes the loss surface and identify subspaces of the model’s data representation that are relatively stable over training
   - highest scoring examples tend to be either unrepresentative outliers of a class, have non standard backgrounds or odd angles, are subject to label noise, or are otherwise difficult
