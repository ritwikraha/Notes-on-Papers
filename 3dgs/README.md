# 3d Gaussian Splatting for Real Time Radiance Field Rendering

### What is splatting?

In computer graphics, splatting is a method for rendering volumetric data. It works by projecting each voxel in the volume onto the screen as a small disc, or "splat". The color and opacity of each splat is determined by the value of the voxel at that location.

Splatting is a relatively simple and efficient way to render volumetric data. However, it can produce images with jagged edges and stair-stepping artifacts. To improve the quality of the rendered image, splatting is often combined with other techniques, such as ray tracing or deferred shading.


### Introduction to the paper


#### Traditional Methods: Meshes and Points
- **Meshes and Points**: These are the most common ways to represent 3D scenes. They are straightforward and work well with graphics hardware like GPUs, making them fast for rendering (drawing the scene on your screen).

#### Newer Methods: Neural Radiance Fields (NeRF)
- **Neural Radiance Fields (NeRF)**: This is a newer approach that uses machine learning to create a continuous (smooth and detailed) representation of the scene. It's good for capturing complex details but can be slow and sometimes noisy when rendering.

#### The New Approach: 3D Gaussian Representation
- **3D Gaussian Representation**: The authors introduce a new method that aims to combine the best of both worlds. It uses a mathematical model (Gaussian) to represent the scene in a way that is both detailed and optimized for fast rendering.
  
- **Tile-Based Splatting**: This is a technique they use to make sure the rendering is fast and high-quality. It breaks the scene into smaller pieces (tiles) and processes them quickly.

#### The Result
- The new method claims to offer high-quality visuals and fast rendering times, making it suitable for real-time applications. It also performs well on several well-known datasets, which suggests it's a robust solution.


### What seems to be the problem officer?

<img width="431" alt="Screenshot 2023-09-05 at 12 08 49 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/55dca738-a776-4660-a018-137604e8b99a">


Novel-view synthesis is the task of generating a new image of a scene from a different viewpoint than the ones used to capture the original images, which means it has been pretty relevant for some time now.

The field faces a challenge in balancing the speed of creating 3D scenes with the quality of the visual output. 

### What was tried before?

#### 1. Traditional Scene Reconstruction and Rendering

<img width="694" alt="Screenshot 2023-09-10 at 12 00 09 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/ebe7f934-71d2-46b5-b6ae-f08fcaec91a3">

- Early methods for creating new viewpoints of a scene (called "novel-view synthesis") used a technique called "light fields." Initially, they captured a lot of data points (densely sampled), but later versions allowed for more random and flexible data capture.
  - Imagine trying to recreate a 3D scene from multiple photos. Initially, you'd take photos from every possible angle, making sure you don't miss any spots (dense sampling). Later, you'd allow for photos taken at random angles without any strict pattern (unstructured capture).

- A technique called "Structure-from-Motion" (SfM) came along, which allowed researchers to create new viewpoints just by using a bunch of photos.
  - Structure-from-motion (SfM) is a technique for estimating the 3D structure of a scene from a collection of images. It does this by finding the camera parameters for each image and then using these parameters to reconstruct the 3D points that were visible in the images.

- After SfM, another technique called "multi-view stereo" (MVS) was developed. MVS could create detailed 3D models from photos.
  - Multi-view stereo (MVS) is a technique for estimating the depth of a scene from a collection of images. It does this by finding the correspondences between points in different images and then using these correspondences to estimate the depth of each point.


#### 2. Neural Rendering and Radiance Fields

<img width="739" alt="Screenshot 2023-09-10 at 12 16 14 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/6fe0b86a-0d29-42c2-800b-7734dee49723">

Deep learning, was used early on to create new viewpoints of a scene. They used CNNs to decide how to blend images together and to work with textures. However, using MVS had some problems, and using CNNs sometimes caused flickering in the final image.

<img width="666" alt="Screenshot 2023-09-10 at 12 17 23 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/11009a7e-b399-4fbb-a26d-43b86448cf65">

While NeRFs greatly improved the quality of rendering through importance sampling and positional encoding, the use of an MLP slowed down the speed of rendering. The current state of the art from the field in MipNeRF 360 which again provides excellent quality renderings but is extremely time-consuming.

<img width="779" alt="Screenshot 2023-09-10 at 12 19 02 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/fe0fd922-273d-48a8-aa03-cf59618d6ed3">

To solve this space discretization has been an important consideration in many studies. Two special papers stand out in this context:
- InstantNGP
- Plenoxels

Imagine you're trying to recreate a 3D scene of a forest:
- InstantNGP is like using a map with specific landmarks (hash grid) and another map showing where trees are densely packed (occupancy grid) to quickly understand the forest layout. The simpler neural network is like a basic guide describing tree types and their appearances.
- Plenoxels is like using a 3D model of the forest where only the main trees are shown (sparse voxel grid), and you don't need a guidebook (neural network) because the model is self-explanatory.

Both of the above papers rely on spherical Harmonics. Imagine "Spherical Harmonics" as a special compass:

For InstantNGP, this compass shows the direction of the wind.
For Plenoxels, this compass changes color based on the direction it's pointing, helping to understand the colors in the scene.
<img width="689" alt="Screenshot 2023-09-10 at 12 30 18 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/e82cb897-7e58-4124-aefe-da24165e7392">

The authors point out that the shortcomings of Neural Renderings are either time inefficiency due to training an MLP or rendering quality compromise due to the effect of a structured grid. The better solution in their opinion is to use an unstructured Gaussian to represent scenes.

3. Point-based rendering and Radiance Fields

Yes, marching a ray through the object and recording its color can be memory-consuming. What can be a better way? Using spatial points to represent images. 

Imagine representing an image in a basic geometric form and then optimizing that geometry. 3d points were how it started. Seminal work has been done in identifying splatting-based techniques where points were replaced by elliptic discs, conical sections etc.

Point based alpha blending shares the image formation models with NeRFs

<img width="731" alt="Screenshot 2023-09-10 at 12 52 32 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/3ad82575-d71a-4c04-94a6-4b902ea80527">

<img width="702" alt="Screenshot 2023-09-10 at 12 53 15 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/3f6e97b0-6207-4efe-8fdf-af8f71645a57">

<img width="283" alt="Screenshot 2023-09-10 at 1 02 26 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/eab38957-b4d8-4ddf-a727-ab836b366351">

But the rendering algorithm is different, While NeRFs are a continous representation and requires expesive sampling, Point Based methods are discrete and require strategic optimization.

<img width="279" alt="Screenshot 2023-09-10 at 1 02 10 AM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/f35f315a-a29a-4516-b699-3ecbf9c74deb">

Based on the previous analysis, the researchers want to use a common technique called 𝛼-blending, but with a twist. They sort small portions of the image (called splats) to benefit from 3D-like representations. Their method ensures that things closer to the viewer are shown before things further away, unlike some other methods that don't care about this order. They also adjust the brightness and color of each splat based on its surroundings and handle splats that have different shapes and sizes. All these special touches make the rendering quality excellent and the processing time fast.


### What is the solution?

<img width="414" alt="Screenshot 2023-09-05 at 12 15 30 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/42f868ea-c71f-4f09-b8b4-848661b7623b">

The solution discussed in the text has three main parts. First, they use 3D Gaussians (a type of mathematical function) to create a flexible and detailed 3D model of a scene. They start with basic camera information and initial 3D points to set up these Gaussians. Unlike other methods that need more complex data, they can work well with simpler inputs.

Second, they fine-tune the 3D Gaussians during the process. They adjust various properties like position, transparency, shape, and lighting effects. This makes the 3D model more accurate and detailed. Sometimes, they even add or remove some of these Gaussians to improve the model's quality.

Lastly, they have a fast method to turn this 3D model into a 2D image that you can see on a screen. They use quick calculations on a graphics card (GPU) to do this efficiently. Their method also makes sure that the 3D model looks accurate from different angles and viewpoints. Overall, their approach creates high-quality 3D models that can be easily and quickly turned into images.

<img width="288" alt="Screenshot 2023-09-07 at 11 47 17 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/d15918e8-6ec7-4efb-a684-91d8e9f2b1f5">


