# What is the problem?
So a few days back I was reading this paper, and this line really stuck with me:
“Human perception will group similar scenes and objects together to parse complex scenes and objects”
And then this line
"Meanwhile, in vision, objects can serve as a natural counterpart to words..."
Okay a lot of line stuck with me, but the point is this:
Objects are analogous to words in a visual context.
The paper in question is a recent release from Meta titled 'R-MAE'

<img width="450" alt="Screenshot 2023-07-20 at 12 06 16 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/13c3a70b-3fad-469a-bedc-86c74f0e65fe">

TL;DR: If you don't want to read the paper and want a short one-line summary here you go: 

What if we could use region-augmented reconstructive Pre-training for vision models starting with Masked Auto Encoders?
The paper goes back to a familiar name:
- Faster RCNN has already validated this idea through multiple literature and architecture, meanwhile
- Reconstructive pretraining such as Masked Auto Encoder has proven to be an effective visual model while providing competitive performance.

The real question is how well regions improve the performance of reconstructive pretraining models like MAE.

# What is the solution?
The authors of the paper propose to begin with MAE as a representative baseline and explore the use/advantage of pre-computed regions in an MAE style.

<img width="373" alt="Screenshot 2023-07-21 at 11 46 02 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/6d06ea4b-4871-4773-8346-465ffa88c702">

This is done in two ways:
- First, the authors create RAE (Region Auto Encoder) which is similar to MAE but with the exception of focusing on regions or region maps as opposed to pixels.
- The authors mention that the trained RAE model can be used to optimize the MAE model parallelly by simply restoring the pixel decoder.

## What other solutions have been considered in the past?
But before we get into all of that let us try to revisit some old friends. Why regions? This can't be a new concept in computer vision/ there are hardly a few of them left. So what made the authors think about regions as a useful feature for augmenting the performance of pretraining models?

<img width="349" alt="Screenshot 2023-07-21 at 11 53 14 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/9d3dfd11-3bd5-4a3f-b978-15af05daf714">

1. **Local:** 
   - Usually, in machine learning, entire images are looked at as one whole unit. However, real-world photos are complex, with different details (local contents) found throughout a single scene. Here the paper tells us to study one of the intrinsic properties of regions, local.
   
2. **Motivation from R-CNN series:** 
   - In the past, this has motivated the development of certain methods, like the R-CNN series, which pay particular attention to different regions within an image, called Regions-of-Interest (RoIs).

3. **Contrastive or Siamese learning:** 
   - In other types of learning, such as contrastive or Siamese learning, the 2D details of images are often simplified into global vectors, basically reducing the image into a single, less detailed representation. This is done to compare one image to another. 

4. **Potential downside:** 
   - However, there can be downsides to simplifying the image too much, especially when trying to locate specific things within the image. 
   - As a result, many newer methods have shifted focus to look at contrast within a single image, using details from local parts of the image like points or regions.

5. **Reconstructive methods:** 
   - Reconstructive methods, like denoising autoencoders, maintain the 2D structure of the image. 
   - However, it's not yet clear how using regions could further improve these types of methods.



<img width="346" alt="Screenshot 2023-07-21 at 11 53 23 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/7f1c67b0-ca8e-4a74-9b79-36281d91d0c5">

1. **Object-centric:** 
   - It is now time to visit the other intrinsic property of regions: objects.
   - Another strong reason for considering regions in images comes from the goal to make machine learning for images more like the learning process for natural language.
   - The way we learn language and the way computers learn to recognize images are quite different. 

2. **Natural language and images:** 
   - In natural language, words carry specific, distinct meanings. However, in images, we're dealing with raw signals recorded in pixels, which is less straightforward.

3. **Objects as counterparts to words:** 
   - In visual perception, objects could serve as the equivalent to words in language. This is because, in the visual world, we constantly refer to and interact with objects, which can often be captured by regions within an image.

4. **Bridging the gap:** 
   - By improving how well machines can recognize regions in images (region awareness), the hope is to find new ways to bridge the gap between the way we learn language and the way machines learn to recognize images.
  
Wait why does MAE come into the picture?

## Why MAE?

 But wait, what is MAE?

 ## MAE a.k.a Maksed Auto Encoders a.k.a Scalable Vision Learners

 <img width="459" alt="Screenshot 2023-07-22 at 12 03 54 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/90bcf876-75cc-4966-a810-8d96354d8fe7">
 
The MAE paper, authored by Kaiming He and his team from the Meta AI lab, is a groundbreaking work in the field of computer vision. It introduces the concept of "masked autoencoders" (MAE) as a robust method for self-supervised learning in computer vision. In simple terms, the MAE technique involves hiding random parts of an image and then trying to fill in the missing sections.

1. **Task:** 
   - The task of MAE (Masked Autoencoding) is to hide part of an image and then try to fill in the missing parts by predicting the values of the hidden pixels. 
   - To make this task challenging, a high percentage of the image (e.g., 75%) is typically hidden. 
   - The machine's attempt at reconstruction is compared to the original image to see how accurate it is.

<img width="245" alt="Screenshot 2023-07-22 at 12 07 02 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/88a3cf84-e50c-4bce-b05a-1e6671bb63ab">


2. **Architecture:** 
   - The architecture of MAE works like an autoencoder, a type of machine learning model that tries to recreate its input.
   - The particular type of autoencoder used here, called ViT (Vision Transformer), breaks the image down into patches and treats them like a sequence of tokens (just like words in a sentence).
   - During the training process, some of these "tokens" are removed and the model tries to fill them back in. 
   - After the model is trained, the part of it that does the encoding can be used as a "backbone" for other tasks, basically helping to pre-process images for other types of machine learning tasks.


# How R-MAE works?

## RAE

<img width="337" alt="Screenshot 2023-07-22 at 12 12 21 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/7c3b8556-ed14-4aa8-8ade-60c362517def">

The authors mention that before going into why R-MAE is considered they must first establish the use of $x$ to pretrain representations. Where $x$ is an additional signal. There are basically three ways to do this:

<img width="396" alt="Screenshot 2023-07-22 at 12 13 34 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/acd27368-66b0-4db4-8a94-79af3d2b9958">

1. Feeding $x$ as an input - The more information the system has, the easier it is for it to understand and learn from the data it's working with.
2. Predicting $x$ as a target - The model can learn using 'x' as a guide, but the task can be so hard that the model might learn the training data too well and perform poorly on new data.
3. MAE-style usage of $x$ - Here, '(1-β)×x' is used as the input and 'β×x' is used as the essential target. The 'mask ratio' acts as a flexible control for the difficulty level of the pre-text task. - This method uses 'x' in a way that balances between only using it as input or only as output. A part of 'x' is used as input, and another part as a target to aim for. The 'mask ratio' can be adjusted to control how hard the task is

**We are sold on using regions, but how do we do it?**

### Region Maps

<img width="334" alt="Screenshot 2023-07-22 at 12 14 13 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/1a3389a6-849a-4dab-a974-9762462c044a">

- Regions are first prepared to be an image like
- Each region can be represented by a binary-valued region map with the dimensions of the image
- Each element in the map has a value of either 0 or 1
- Given any partially visible region map (mask ration = $\beta_{R}$) the model will be asked to complete it same as MAE does for pixels.

### Architecture

<img width="333" alt="Screenshot 2023-07-22 at 12 15 10 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/9827dfda-d1a4-44d1-8a1f-c4a4b37750eb">

- RAE, like MAE, consists of two main parts: an encoder and a decoder. These are used for region autoencoding, which means they are used to encode and decode regions of an image.
- The encoder and decoder are built using ViT (Vision Transformer) blocks. There are 'mE' number of 'pE'-dimensional ViT blocks used for the encoder, and 'mD' number of 'pD'-dimensional ViT blocks used for the decoder.
- However, using just a region encoder-decoder pair isn't enough for their goal, which is to develop a pre-trained pixel encoder. A pixel encoder is a tool that understands and interprets pixel information from images.
- So, they keep the encoder part from the MAE system in the RAE system.
- In addition, they use a 'neck' of 'mN' number of ViT blocks to match dimensions and optionally spread information before it is fed into the region decoder.
- This configuration makes good use of the plentiful contextual information found in the pixels to pre-train the encoder. This pre-training helps the encoder to better understand and interpret pixel data.
- For a visual representation of this system, refer to Figure 2 in the original document.
<img width="399" alt="Screenshot 2023-07-07 at 1 29 13 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/6d69996f-5007-416c-9e24-6ada0d17d62f">


### One-to-many mapping

<img width="365" alt="Screenshot 2023-07-22 at 12 15 58 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/1f90b7b4-daf9-4a12-9b0e-398fb85e254e">

- Regions in images are considered an additional 'modality' or type of data compared to pixels in a pixel-based MAE system. However, there's a distinct challenge here that can't be fully addressed just by considering regions as another type of data.
- Unlike other data types (like depth or semantic maps), where there is a one-to-one correspondence to pixels, the mapping between images and regions is one-to-many. That means one pixel can belong to many regions, which adds complexity to the task.
- This problem is also encountered in object detection, and the usual solution (as used by R-CNN, a type of convolutional network used for object detection) is to sample and stack regions in the batch axis, processing each region separately.
- In the RAE system, this means each region map would go through the encoder-decoder process individually. So, if there are 'b' images and 'k' regions per image, the network has to be applied 'b×k' times, which can be computationally expensive.
- A simpler alternative might be to combine the 'k' regions in the channel axis. This would allow them to be viewed as a single image for encoding and decoding, and computations could be shared in the intermediate blocks.
- However, unlike natural images that have fixed channel orders (like RGB), randomly sampled regions can appear in any order. So, it would be ideal if the solution still maintains 'permutation equivariance', which means it would work the same no matter the order of the regions.


### Regions as queries – the length variant

<img width="344" alt="Screenshot 2023-07-22 at 12 16 58 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/172ba04e-dec8-4c96-a3c0-90e42983ea12">


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


### Region Meets MAE

<img width="356" alt="Screenshot 2023-07-22 at 12 17 35 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/65e20cac-e7f3-4a7d-9729-f092747cdcfe">


The Region AutoEncoder (RAE) is fully compatible with the Masked AutoEncoder (MAE), meaning they can work together smoothly. 

The training can be done together by using the pixel encoder from the MAE and applying a combined loss function. The combined loss function is the sum of two individual losses: LI (probably referring to the loss from the MAE or image loss) and LR (likely referring to the region loss from the RAE), with a balance factor λ (lambda) in between. The balance factor λ is set to 1 by default, meaning that both types of loss are considered equally important during training.

In simpler terms:

"The RAE can work well with the MAE. They can be trained together by bringing back the pixel encoder from the MAE and using a combined measurement of mistakes, which considers both types of losses equally."

<img width="694" alt="Screenshot 2023-07-11 at 1 29 23 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/b2234f36-d8bb-4322-a4b6-3593e93a547a">


In the paper Figure 2, provides a visual illustration of the default pre-training pipeline, including what might be less important or 'de-highlighted' parts. The key points mentioned are:

The 'pixel branch' feeds into the 'region branch'. In other words, the process or data flow that deals with pixels contributes to the process or data flow that deals with regions, but not the other way around.

The 'mask' is shared between the two branches. A mask in this context usually means certain data that is used in both the pixel and region processes.

The pipeline is named R-MAE, which stands for Region-aware Masked Autoencoding. This name reflects that it's a process that uses masked autoencoding (a type of machine learning method), but with awareness or consideration of regions in images.


### Implementation details
Although training and implementation details are quite well-detailed and require a deeper look, we shall discuss the overviews of the major concepts in this video. Specifically the source of regions and pre-training data.

1. **Source of regions:** 
   - They used a tool called the Felzenswalb-Huttenlocher (FH) algorithm to identify different parts, or 'regions', of the images. 
   - This tool doesn't need any guidance (it's "unsupervised"), it's fast, and it looks at the whole image. 
   - It's commonly used in many other ways of finding objects in images, like something called selective search.

2. **Pre-training data:** 
   - They prepared their models (RAE and R-MAE) using a collection of images called COCO train2017.
   - They chose COCO because it has many pictures with full scenes and it also has 'ground-truth' regions - that means parts of the image that are accurately labeled, which can be very useful for training. 
   - To generate regions, they used the FH tool at three different 'sizes', namely {500, 1000, 1500}. These sizes also set the smallest 'clusters' or groups of pixels that can be a region.
   - The COCO dataset has fewer images than another popular dataset called ImageNet. So, they ran the training process for a longer time (4,000 rounds, compared to the usual 800) which ends up being about half of what they usually do with the MAE method.

(NOT PART OF VIDEO)
3. **Other pre-training details:**
   - They generally followed the same settings (called hyperparameters) as those used in the MAE method.
   - They set the initial 'learning rate' to 0.0001. The learning rate is like the size of steps taken when the model is learning. A smaller step size can make the learning process more stable and helps maintain good performance.
   - They used a version of their models that deals with 'length'. 
   - They used a specific tool called ViT-B to handle pixels, and a 1-block, 128-dimensional version of ViT for parts called the 'neck', 'region encoder', and 'region decoder'.
   - After the region decoder, they used a simple network called a 3-layer Multilayer Perceptron (MLP) to predict regions.
   - They took 8 regions from each image (and they could repeat some regions), and they set a 'mask ratio' of 0.75. The mask ratio has to do with how much of the image is used for training.
   - They set the weights for two things (λ and background loss) to 1. This means they consider both of these factors equally important.
   - When they used MAE, the part of the model that handles pixels would send information to the part that handles regions, and they would use the same 'random masks' in both parts. Random masks are a way of randomly ignoring some parts of the image during training to help the model learn better.

## Conclusion

<img width="373" alt="Screenshot 2023-07-22 at 12 18 23 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/44647b30-d85e-43b5-99a7-73fb0bce5592">

- The authors presented a new pre-training method called R-MAE. This method focuses on the concept of 'regions' in the Masked AutoEncoder (MAE) model, which is an important concept in visual understanding.
- A lot of experiments and the results show that R-MAE is more 'region-aware'. This means it's better at understanding and using information about regions in images. Because of this, it can improve performance on tasks that involve identifying specific locations in images, such as detection and segmentation tasks.
- One of the key features of R-MAE is that it treats regions as 'queries'. This makes the 'region branch' (the part of the model that handles regions) very efficient, only adding a small amount of additional computational work (1.3% overhead).

But more than all the efficiencies and the metrics involved the paper tries to ask an important question.

"What is the counter part of words in a visual setting?"

The effort is thus to close the gap to natural language processing by learning the visual equivalent of words in computer vision. The effort is to ask how do we enable machines to see they way we do.
