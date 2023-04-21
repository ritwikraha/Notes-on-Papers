## Pre-Requisites
- What is segmentation?
    - process of dividing an image into multiple smaller segments, or regions, each of which represents a distinct object or part of the image. The goal of image segmentation is to simplify the image data, making it easier to analyze and understand.
    - simple thresholding, cnns, transformers

- Difference between instance, panoptic and semantic segmentation?
    - Semantic Segmentation: Semantic segmentation is a type of image segmentation that classifies each pixel in an image into a specific category or class. The goal is to identify the different regions in the image and assign them semantic labels based on their contents. For example, in an image of a street scene, semantic segmentation might label each pixel as road, building, tree, car, or sky.
    - Instance Segmentation: Instance segmentation is a more advanced version of semantic segmentation that not only identifies the different regions in an image, but also distinguishes between different instances of the same object. This means that each individual object in the image is assigned a unique label, making it possible to differentiate between two cars or two people that might otherwise appear similar.
    - Panoptic Segmentation: Panoptic segmentation is a newer type of segmentation that combines both semantic and instance segmentation. In panoptic segmentation, all the pixels in the image are classified into either object or stuff classes. The object classes represent instances of objects, while the stuff classes represent the background or environment. This approach provides a more comprehensive understanding of the image by combining both object-level and scene-level information.


Example:
![image](https://i.imgur.com/tGnBSOq.png)

Links:
[Semantic vs. Instance vs. Panoptic Segmentation](https://pyimagesearch.com/2022/06/29/semantic-vs-instance-vs-panoptic-segmentation/)

# Per-Pixel Classification is Not All You Need for Semantic Segmentation

## Abstract 
- Semantic Segmentation
    - Per-pixel classification: Here we apply a classification loss to each output pixel. This formulation naturally partitions the image into semantic regions.
    - Mask classification: This is an alternative paradigm that disentangles the image partitioning and classification aspects of segmentation. Here a set of binary masks are predicted associated with a single class prediction. Upon each iteration, the binary masks and the predicted labels are tuned.
- **Question**: 
    - Can a **single** mask classification model simplify the landscape of effective approaches to semantic- and instance-level segmentation tasks?
    - And can such a mask classification module **outperform** existing per-pixel classification methods for semantic segmentation?

## Introduction
- To address the above questions the authors propose Maskformer approach that seamelessly connects per-pixel classification model into a mask classification model.
- **How?**
    - Uses the set prediction mechanism shown in DETR
    - Transformer Decoder to compute set of pairs (class prediction, mask embedding vector)
    - Use the mask embedding vector to get the binary mask precition. How?
        - Dot product between mask embedding vector and per-pixel embedding from underlying CNN

## Related Works

## From Per-Pixel to Mask Classification
- Formulating semantic segmentation as either a per-pixel classification or mask classification
    - ### Per Pixel Classification
        - predict the probability distribution over all possible K categories for every pixel of an H×W image
        - Prediction values are: <img width="175" alt="Screenshot 2023-04-19 at 11 14 11 AM" src="https://user-images.githubusercontent.com/44690292/232977761-b490a9e4-07e1-4e52-9b42-33bea524c6ee.png">
        - Ground truth labels are: <img width="254" alt="Screenshot 2023-04-19 at 11 16 48 AM" src="https://user-images.githubusercontent.com/44690292/232978118-689b2d43-0cf7-433e-93a5-080b357c96ff.png">
        - for every pixel a per pixel cross entropy or negative log likelihood is applied <img width="284" alt="Screenshot 2023-04-19 at 11 17 52 AM" src="https://user-images.githubusercontent.com/44690292/232978323-cfd05218-91b7-442c-b3c9-b696d6e360f2.png">
    - ### Mask Classification 
    -  requires two steps
        - Partitioning:
            - Group into N regions with the help of binary masks.
            - <img width="201" alt="Screenshot 2023-04-19 at 12 19 15 PM" src="https://user-images.githubusercontent.com/44690292/232989832-307fe005-964e-4747-8a2e-7213fc2f9881.png">
        - Associating:
             - Each region is associated with some distribution over K categories
             - Desired output z is defined as <img width="139" alt="Screenshot 2023-04-19 at 12 25 06 PM" src="https://user-images.githubusercontent.com/44690292/232991086-355583c6-3762-44c8-a37b-0f9e656758ae.png">
             - In addition the probability distribiution also contains a $\phi$ token for no-object classes.
        - Training Mask Classification:
             - To train a mask classification model
             - We need to find a matching $\sigma$ between <img width="136" alt="Screenshot 2023-04-19 at 12 35 03 PM" src="https://user-images.githubusercontent.com/44690292/232993746-4045fa1a-c76f-4c16-849c-a96cb44f809a.png"> and <img width="420" alt="Screenshot 2023-04-19 at 12 35 38 PM" src="https://user-images.githubusercontent.com/44690292/232993869-831bc7b1-f8c7-49b3-8a40-20b63e60c61b.png">
             - Since $N$ is far larger than $N_gt$, we pad the set of ground truth tokens with $\phi$
       ## Object detection set prediction loss

        DETR infers a *fixed-size* set of $N$ predictions. $N$ is significantly larger than the typical number of objects in an image.

        * $y \to$ ground truth set of objects. (Note: The ground truth set of objects is padded with $\phi$ to be of the same length as that of the predicted set).
        * $\hat{y} \to$ predicted set of objects. $\hat{y} = \\{\hat{y}_{i}\\}^{N}\_{i}$

        > Given a bipartite graph, a matching is a subset of the edges for which every vertex belongs to exactly one of the edge.

        The objective is to find an *optimal* bipartite matching of the two sets (ground truth and the predictions). The authors propose a loss to fulfil this objective.

        To understand the proposed loss let's lay some foundation first. 

        $\mathfrak{S}_{N}\to$ is a set of all the possible permutations of $N$.

        If $N = 2$, $\mathfrak{S}_{N} = \\{ \\{1, 2\\}, \\{2, 1\\}\\}$

        To find the optimal bipartite matching between thes two sets we search for a permutaiton of $N$ elements $\sigma \in \mathfrak{S}_{N}$ with the lowest cost:

        $$\hat{\sigma} = \text{arg min}_{\sigma \in \mathfrak{S}_{N}} \sum_{i}^{N} \mathcal{L}_{\text{match}}(y_{i}, \hat{y}_{\sigma(i)})$$

        With $N=2$ example the cost function looks something like this:

        For $\sigma = \\{1, 2\\}$:

        $\mathcal{L}\_{\text{match}}(y\_{1}, \hat{y}\_{1}) + \mathcal{L}_{\text{match}}(y\_{2}, \hat{y}\_{2})$


        For $\sigma = \\{2, 1\\}$:

        $\mathcal{L}\_{\text{match}}(y\_{1}, \hat{y}\_{2}) + \mathcal{L}_{\text{match}}(y\_{2}, \hat{y}\_{1})$

        We will now choose the $\sigma$ that lowers the cost. Upon selecting the permutation that lowers the cost, we eventually get the optimal bipartite matching of the ground truth and the predicted objects.

        That is all fun and games, but how do we chose $\sigma$? This is where **hungarian algorithm** enters.
        
        
        ### MaskFormer
        
       1. **Pixel-level module:**
   - Input image size: $H × W$ (height × width)
   - Feature map: $F \in R^{(C_F × H_S × W_S)}$, where $C_F$ is the number of channels and $S$ is the stride (we use $S = 32$ in this work)
   - Per-pixel embeddings: $E_pixel \in R^{(C_E × H × W)}$, where $C_E$ is the embedding dimension

2. **Transformer module:**
   - Input: Image features F and N learnable positional embeddings (queries)
   - Output: N per-segment embeddings $Q \in R^{(C_Q × N)}$, where $C_Q$ is the embedding dimension

3. **Segmentation module:**
   - Class probability predictions: ${p_i \in Δ^{(K+1)}}^N_{i=1}$, where Δ is the simplex (probability distribution over K+1 classes) and N is the number of segments
   - For mask prediction:
       * An MLP with 2 hidden layers converts per-segment embeddings Q to N mask embeddings E_mask ∈ R^(C_E × N)
       * Binary mask prediction m_i ∈ [0,1]^(H × W) is obtained by taking the dot product between the i-th mask embedding and per-pixel embeddings E_pixel, followed by a sigmoid activation function:
         m_i[h, w] = sigmoid(E_mask[:, i]^T · E_pixel[:, h, w])

**Loss function during training:**
- L_mask-cls: A combination of a cross-entropy classification loss and a binary mask loss L_mask for each predicted segment
- L_mask: A linear combination of a focal loss and a dice loss, multiplied by hyperparameters λ_focal and λ_dice, respectively

And the simplified math explanation:

- Feature map: A low-resolution version of the input image, with a reduced number of channels
- Per-pixel embeddings: A higher-resolution representation of the image, where each pixel is associated with an embedding vector
- Per-segment embeddings: A set of vectors encoding global information about each segment
- Class probability predictions: A probability distribution over classes for each segment
- Binary mask prediction: A matrix representing the shape of each segment in the image, with values between 0 and 1

The model calculates the binary mask prediction by taking the dot product of the i-th mask embedding and per-pixel embeddings, then applying a sigmoid function. The loss function used during training combines classification and mask prediction errors.




    



