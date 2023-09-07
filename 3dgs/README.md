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


There are some obvious trade-offs between speed and quality in different methods for creating 3D scenes from multiple photos.

### Previous Methods
- **Fast Training but Lower Quality**: Some recent methods can create 3D scenes quickly but don't achieve the high visual quality that state-of-the-art (SOTA) methods can.
  
- **High Quality but Slow**: The best-quality methods, like Mip-NeRF360, can take up to 48 hours to train a model. These methods produce excellent visual results but are not practical for real-time applications.

### Frame Rates
- The faster methods can achieve interactive frame rates of 10-15 frames per second, which is okay but not ideal for real-time, high-resolution rendering.

### The New Solution
- The text hints at a new solution that aims to balance both speed and quality, although it cuts off before going into detail.

In summary, the field faces a challenge in balancing the speed of creating 3D scenes with the quality of the visual output. The new solution aims to address this by offering both fast optimization times and high-quality results.

### What was tried before?

1. Traditional Scene Reconstruction and Rendering
2. Neural Rendering and radiance Fields
3. Point-based rendering and Radiance Fields?

### What is the solution?

<img width="414" alt="Screenshot 2023-09-05 at 12 15 30 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/42f868ea-c71f-4f09-b8b4-848661b7623b">

The solution discussed in the text has three main parts. First, they use 3D Gaussians (a type of mathematical function) to create a flexible and detailed 3D model of a scene. They start with basic camera information and initial 3D points to set up these Gaussians. Unlike other methods that need more complex data, they can work well with simpler inputs.

Second, they fine-tune the 3D Gaussians during the process. They adjust various properties like position, transparency, shape, and lighting effects. This makes the 3D model more accurate and detailed. Sometimes, they even add or remove some of these Gaussians to improve the model's quality.

Lastly, they have a fast method to turn this 3D model into a 2D image that you can see on a screen. They use quick calculations on a graphics card (GPU) to do this efficiently. Their method also makes sure that the 3D model looks accurate from different angles and viewpoints. Overall, their approach creates high-quality 3D models that can be easily and quickly turned into images.

<img width="288" alt="Screenshot 2023-09-07 at 11 47 17 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/d15918e8-6ec7-4efb-a684-91d8e9f2b1f5">


