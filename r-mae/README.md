# What is the problem?
TL;DR: What if we could use region augmented reconstructive Pre training for vision models starting with Masked Auto Encoders.

“Human perception will group similar scenes and objects together to parse complex scenes and objects”

Faster RCNN has already validated this idea through multiple literature and architecture.

Reconstructive pretraining such as Masked Auto Encoder has proven to be an effective visual model while providing competitive performance.

So the question remains how well does regions improve the performance of reconstructive pretraining models like MAE?

# What is the solution?
The authors of the paper propose to begin with MAE as a representative baseline and explore the use/advantage of pre computed regions in an MAE style.

This is done in two ways:
- First the authors create RAE (Region Auto Encoder) which is similar to MAE but with the exception of focusing on regions or region maps as opposed to pixels.
- The authors mention that the trained RAE model can be used to optimize the MAE model parallely by simply restoring the pixel decoder.

## What other solutions have been considered in the past?

# Wait what is MAE again?

## Why MAE?

# How R-MAE works?
￼
