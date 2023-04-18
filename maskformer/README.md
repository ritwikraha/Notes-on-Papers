## Pre-Requisites
- What is segmentation?
    - process of dividing an image into multiple smaller segments, or regions, each of which represents a distinct object or part of the image. The goal of image segmentation is to simplify the image data, making it easier to analyze and understand.
    - simple thresholding, cnns, transformers

- Difference between instance, panoptic and semantic segmentation?
    - Semantic Segmentation: Semantic segmentation is a type of image segmentation that classifies each pixel in an image into a specific category or class. The goal is to identify the different regions in the image and assign them semantic labels based on their contents. For example, in an image of a street scene, semantic segmentation might label each pixel as road, building, tree, car, or sky.
    - Instance Segmentation: Instance segmentation is a more advanced version of semantic segmentation that not only identifies the different regions in an image, but also distinguishes between different instances of the same object. This means that each individual object in the image is assigned a unique label, making it possible to differentiate between two cars or two people that might otherwise appear similar.
    - Panoptic Segmentation: Panoptic segmentation is a newer type of segmentation that combines both semantic and instance segmentation. In panoptic segmentation, all the pixels in the image are classified into either object or stuff classes. The object classes represent instances of objects, while the stuff classes represent the background or environment. This approach provides a more comprehensive understanding of the image by combining both object-level and scene-level information.


## Abstract 
- Semantic Segmentation
    - Per-pixel classification: Here we apply a classification loss to each output pixel. This formulation naturally partitions the image into semantic regions.
    - Mask classification: This is an alternative paradigm that disentangles the image partitioning and classification aspects of segmentation. Here a set of binary masks are predicted associated with a single class prediction. Upon each iteration, the binary masks and the predicted labels are tuned.
- **Question**: 
    - Can a **single** mask classification model simplify the landscape of effective approaches to semantic- and instance-level segmentation tasks?
    - And can such a mask classification module **outperform** existing per-pixel classification methods for semantic segmentation?

## Introduction

