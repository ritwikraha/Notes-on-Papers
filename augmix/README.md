# AUGMIX:A SIMPLE DATA PROCESSING METHOD TO IMPROVE ROBUSTNESS AND UNCERTAINTY

## Abstract 

In machine learning, we use a set of data (known as the "training" data) to teach an algorithm how to solve a task. In this context, we're talking about deep neural networks which are a kind of machine learning algorithm designed to classify images. An image classifier's job is to look at an image and decide which category it belongs to, like identifying whether a photo is of a cat or a dog.

When the algorithm is learning, it adjusts itself to perform well on the training data. It's then tested on separate "test" data to see how well it has learned. Ideally, the training and test data should be very similar (identically distributed) in nature. For example, if the task is to distinguish between cats and dogs, and if all the images in the training set are taken in broad daylight, we expect the test set also to have images taken in similar conditions.

However, in real-world scenarios, there can be a mismatch between the training and test data. This could be due to a variety of factors like the lighting conditions, the angle of the camera, or the breed of the dogs and cats in the images. When this mismatch happens, the accuracy of the image classifier can drop significantly because it has not encountered these conditions during training.

Most current techniques for training these algorithms struggle when the test data is different from the training data in ways they didn't anticipate. This is where the technique called "AUGMIX" comes in.

AUGMIX is a method that helps improve the robustness of the model. Robustness in this context refers to the model's ability to maintain accuracy even when the test data differs from the training data in unexpected ways.

Here's how AUGMIX works: it applies a mix of augmentations (slight modifications) to the images in the training data. These might include things like adjusting the brightness or contrast, rotating the image, or zooming in slightly. By doing this, AUGMIX creates a wide range of scenarios the model might encounter. It's like showing the model not only pictures of cats and dogs taken in the day but also at twilight, from different angles, of different breeds, etc.

This helps in two ways:

1. It makes the model more robust because it has seen a wider range of image conditions during training.
2. It helps the model to provide better uncertainty estimates. Uncertainty estimates tell us how confident the model is about its predictions. A well-calibrated model knows when it's likely to be wrong, which is very useful when decisions based on these predictions have significant consequences.

## Introduction

### The Problem

1. **Machine Learning Models and Training Data:** Machine learning models are like students. They learn from a book (training data) and they're expected to use that knowledge to solve problems (make predictions) in the real world (deployment). It's very important that the book is representative of the real world, otherwise, they'll struggle to solve real-world problems correctly.

2. **Mismatch between Training and Test Data:** Imagine if a student studied from a book about mammals but was then tested on birds. The student would likely fail because the test (test data) doesn't match what they learned (training data). The same thing can happen with machine learning models. But even though this problem is common, it's not studied enough. As a result, machine learning models often struggle when they encounter data that's different from what they were trained on.

3. **Data Corruption:** Just a little bit of change or "corruption" to the data (like a bird wearing a hat in the test data when the book only had birds without hats) can confuse the machine learning model (classifier). There aren't a lot of techniques available yet to make these models more resistant to such changes.

4. **Uncertainty Quantification:** Some machine learning models (like probabilistic and Bayesian neural networks) try to also measure how sure they are about their predictions (uncertainty). But these models can also struggle when the data shifts because they weren't trained on similar examples.

5. **Corruption-Specific Training:** Training a model with a focus on specific corruptions (like birds wearing specific types of hats) encourages the model to only remember these specific corruptions and not learn a general ability to handle any kind of hat.

6. **Data Augmentation:** Some people propose aggressively changing the training data (aggressive data augmentation) to prepare the model for all kinds of changes it might encounter. But this approach can require a lot of computational resources.

7. **Trade-off between Accuracy, Robustness, and Uncertainty:** Chun et al. (2019) found that many techniques that improve the model's test score (clean accuracy) make it less able to handle data corruption (robustness). Similarly, techniques that make a model more robust often make it worse at estimating how sure it is about its predictions (uncertainty). So there's often a trade-off between accuracy, robustness, and uncertainty.

### The Solution

1. **AUGMIX:** This is a new technique that manages to hit a sweet spot. It improves the model's ability to handle data corruption (robustness) and its ability to estimate how sure it is about its predictions (uncertainty), all while maintaining or even improving its test score (accuracy). And it does this on standard benchmark datasets, which are datasets commonly used to test machine learning models.

2. **How AUGMIX Works:** AUGMIX uses randomness (stochasticity) and a wide variety of changes (diverse augmentations) to the training data. It also uses a special type of loss function called **Jensen-Shannon Divergence consistency loss**, which is a mathematical way to measure how different two probability distributions are. Furthermore, it mixes multiple changed versions of an image to achieve great performance. It's like showing the model different versions of the same image (like a cat in different lighting conditions, from different angles, etc.) and teaching it that they are all still the same category (a cat).

3. **AUGMIX Results on ImageNet:** ImageNet is a popular benchmark dataset for image classifiers. AUGMIX achieves the best performance so far on this dataset in terms of handling data corruption. It also reduces perturbation instability from 57.2% to 37.4%. Perturbation instability is a measure of how much small changes to the input (like slightly changing the color of an image) affect the model's predictions. A high perturbation instability means the model's predictions change a lot even for small changes to the input, which is not desirable. By reducing this from 57.2% to 37.4%, AUGMIX makes the model more stable and reliable.

4. Offocial code released at : https://github.com/google-research/augmix


