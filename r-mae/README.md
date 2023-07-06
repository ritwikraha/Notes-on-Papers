# What is the problem?
TL;DR: What if we could use region-augmented reconstructive Pre-training for vision models starting with Masked Auto Encoders?

“Human perception will group similar scenes and objects together to parse complex scenes and objects”

Faster RCNN has already validated this idea through multiple literature and architecture.

Reconstructive pretraining such as Masked Auto Encoder has proven to be an effective visual model while providing competitive performance.

So the question remains how well do regions improve the performance of reconstructive pretraining models like MAE?

# What is the solution?
The authors of the paper propose to begin with MAE as a representative baseline and explore the use/advantage of pre-computed regions in an MAE style.

This is done in two ways:
- First, the authors create RAE (Region Auto Encoder) which is similar to MAE but with the exception of focusing on regions or region maps as opposed to pixels.
- The authors mention that the trained RAE model can be used to optimize the MAE model parallelly by simply restoring the pixel decoder.

## What other solutions have been considered in the past?

# Wait what is MAE again?

## Why MAE?

# How R-MAE works?

## RAE
The authors mention that before going into why R-MAE is considered they must first establish the use of $x$ to pretrain representations. Where $x$ is an additional signal. There are basically three ways to do this:

1. Feeding $x$ as an input - The more information the system has, the easier it is for it to understand and learn from the data it's working with.
2. Predicting $x$ as a target - The model can learn using 'x' as a guide, but the task can be so hard that the model might learn the training data too well and perform poorly on new data.
3. MAE-style usage of $x$ - Here, '(1-β)×x' is used as the input and 'β×x' is used as the essential target. The 'mask ratio' acts as a flexible control for the difficulty level of the pre-text task. - This method uses 'x' in a way that balances between only using it as input or only as output. A part of 'x' is used as input, and another part as a target to aim for. The 'mask ratio' can be adjusted to control how hard the task is

**We are sold on using regions, but how do we do it?**

### Region Maps

- Regions are first prepared to be an image like
- Each region can be represented by a binary-valued region map with the dimensions of the image
- Each element in the map has a value of either 0 or 1
- Given any partially visible region map (mask ration = $\beta_{R}$) the model will be asked to complete it same as MAE does for pixels.

### Architecture

- RAE, like MAE, consists of two main parts: an encoder and a decoder. These are used for region autoencoding, which means they are used to encode and decode regions of an image.
- The encoder and decoder are built using ViT (Vision Transformer) blocks. There are 'mE' number of 'pE'-dimensional ViT blocks used for the encoder, and 'mD' number of 'pD'-dimensional ViT blocks used for the decoder.
- However, using just a region encoder-decoder pair isn't enough for their goal, which is to develop a pre-trained pixel encoder. A pixel encoder is a tool that understands and interprets pixel information from images.
- So, they keep the encoder part from the MAE system in the RAE system.
- In addition, they use a 'neck' of 'mN' number of ViT blocks to match dimensions and optionally spread information before it is fed into the region decoder.
- This configuration makes good use of the plentiful contextual information found in the pixels to pre-train the encoder. This pre-training helps the encoder to better understand and interpret pixel data.
- For a visual representation of this system, refer to Figure 2 in the original document.
<img width="399" alt="Screenshot 2023-07-07 at 1 29 13 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/6d69996f-5007-416c-9e24-6ada0d17d62f">
In summary:

1. RAE, similar to MAE, uses an encoder and a decoder for region autoencoding.
2. Both parts are constructed using ViT blocks.
3. Just an encoder-decoder pair isn't sufficient, so they keep the MAE encoder in the RAE system.
4. They use an extra 'neck' of ViT blocks to match dimensions and optionally distribute information.
5. This setup effectively uses the plentiful pixel information to pre-train the encoder.

### One-to-many mapping

- Regions in images are considered an additional 'modality' or type of data compared to pixels in a pixel-based MAE system. However, there's a distinct challenge here that can't be fully addressed just by considering regions as another type of data.
- Unlike other data types (like depth or semantic maps), where there is a one-to-one correspondence to pixels, the mapping between images and regions is one-to-many. That means one pixel can belong to many regions, which adds complexity to the task.
- This problem is also encountered in object detection, and the usual solution (as used by R-CNN, a type of convolutional network used for object detection) is to sample and stack regions in the batch axis, processing each region separately.
- In the RAE system, this means each region map would go through the encoder-decoder process individually. So, if there are 'b' images and 'k' regions per image, the network has to be applied 'b×k' times, which can be computationally expensive.
- A simpler alternative might be to combine the 'k' regions in the channel axis. This would allow them to be viewed as a single image for encoding and decoding, and computations could be shared in the intermediate blocks.
- However, unlike natural images that have fixed channel orders (like RGB), randomly sampled regions can appear in any order. So, it would be ideal if the solution still maintains 'permutation equivariance', which means it would work the same no matter the order of the regions.

To summarize:

1. Regions in images present a unique challenge because one pixel can belong to multiple regions.
2. This problem is similar to one found in object detection.
3. The common solution is to process each region separately, but this can be costly.
4. A possible alternative is to combine regions and treat them as a single image, but the order of the regions can vary.
5. An ideal solution would work regardless of the order of the regions.


### Regions as queries – the length variant

The final concept for the model takes inspiration from DETR (DEtection TRansformer) and uses 'object queries' to decode objects. The method and the benefits are broken down as follows:

1. Each region in an image is encoded and pooled into a 1D (one-dimensional) embedding, which is a process that translates high-dimensional data into lower dimensions.

2. These multiple region embeddings are then concatenated (combined) along the sequence length axis to form 'region queries'.

3. These 'region queries' are then used to decode region maps from the output of the pixel encoder (the process is explained further in Figure 2 of the original document).

4. Because Vision Transformer (ViT) blocks operate as 'set operations' with regard to the input, this solution inherently preserves the order of the regions, which is called permutation equivariance.

5. The last decoder block in the process is responsible for spatially expanding the region queries.

6. This decoder has two sets of inputs and thus follows a three-layer design with an additional cross-attention layer that uses outputs from the 'neck' (an intermediate processing stage) to generate keys and values.

7. Unlike standard attention layers that calculate a weighted sum over values to produce the output, the query is expanded by directly adding it to all the values.

<img width="394" alt="Screenshot 2023-07-07 at 1 40 37 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/a4975775-d893-4bfe-836e-e5968d386d68">


8. A small Multilayer Perceptron (MLP) head, which is a type of artificial neural network, is attached afterwards to predict region maps based on these spatially expanded features.

9. This variant of the method alleviates the linear complexity with respect to the number of regions, and still maintains the desired property of permutation equivariance. Because of this, it was chosen as the default method for RAE.

10. As for the loss function used to measure the performance of the model, while ℓ2 loss (a type of regression loss function) is suitable for real-valued pixel predictions, cross-entropy loss is used for binary-valued regions by default.

11. Modeling the problem as a classification task allows for easy balance of weights between foreground and background regions.



