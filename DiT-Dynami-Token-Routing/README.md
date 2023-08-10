# Annotated DiT
## Effecient Vision Transformers with Dynamic TOken Routing

### Do Transformers see small objects the same way they see big objects?



A recent paper aims at studying how well Vision Transformers can perform when they are able to see the object at different scales. The paper in question is named [DiT: Effecntient Vision Transformers with Dynamic Token Routing](https://arxiv.org/abs/2308.03409)

<img width="390" alt="Screenshot 2023-08-10 at 7 49 14 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/2ecc9b34-7a3f-4b4d-86cc-45dabb2f0e96">

In many recent digital image processing techniques, every piece (or token) of an image is processed in a very fixed manner. This can be problematic because not all objects in a photo are the same: they can be big or small, or some objects can be harder to recognize than others.

To address this, the researchers introduced a new method for a system they call the Dynamic Vision Transformer, or DiT for short. Instead of treating every piece of the image in the same way, DiT adjusts its approach based on the specifics of each piece. Imagine if one is trying to piece together a jigsaw puzzle: some pieces might fit easily, while others need more attention and adjusting. DiT does something similar but for digital images.

In their process, they use gates that are a bit like checkpoints. These gates decide how each image piece is processed, allowing the system to handle the image in multiple ways. By doing this, DiT can be more sensitive to the differences in the objects in the image, such as their size or how easy they are to recognize.

An added advantage? The researchers also designed DiT to be more efficient by setting certain limits or stopping the processing early when it's not needed.

![image](https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/74c27112-3c9a-41ab-873c-29442aadd731)


### What is the problem that is being solved here?

<img width="615" alt="Screenshot 2023-08-10 at 7 53 25 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/66b5119e-dfdd-45c1-b690-092728760ea5">


Let us try to make the problem simple enough for anyone to understand. Imagine you’re looking at a big picture with both an elephant and an ant in it. Now, these networks we’re talking about have a tough time figuring out things in pictures, especially when objects are of different sizes, like our huge elephant and tiny ant.

So, two problems they face are:

	1.	Object Size: Some objects in pictures are big, and some are tiny. Just like in our picture, the elephant takes up a lot more space than the ant. This means the computer sees more “pieces” or “tokens” of the elephant than the ant. These pieces have different details based on the size of the object.
	2.	Detail Needed: Think of it like this - recognizing a person in a picture might need more information than recognizing a football. It’s like trying to tell apart two nearly identical Lego sets. One might need you to look more closely at tiny details, while the other is easier to identify with just a quick glance.

 ### This seems like a well-documented problem, what was tried before?
 
<img width="623" alt="Screenshot 2023-08-10 at 8 00 10 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/001a2ddf-c8e8-4535-9456-e03c8370af15">

 In the past CNNs and recent transformer networks as well follow a sequential methodology. What does that mean?
 In simple terms, it means progressively reducing the spatial size of the feature maps. What could go wrong? You guessed it, the elephant is still recognizable at a smaller scale, but the poor ant is now a speck.

 The paper studies the related works by broadly classifying them into three categories:
<img width="490" alt="Screenshot 2023-08-10 at 8 08 32 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/eeb6ad6c-5eab-47c8-a0c9-dbf0a55490c1">

 1. Transformer Backbones: How PVT, [Pyramid Vision Transformer](https://paperswithcode.com/paper/pyramid-vision-transformer-a-versatile) is an important work that introduces Pyramid Structure to Vision Transformers.
 2. Multi-sclae feature for Dense prediction: This goes back to papers like FPN, RetinaNet, and Masl R-CNN and how they depend on the scale of objects for their performance.
 3. 3. Dynamic Networks: Some CNN architectures as well as [Dynamic ViT](https://arxiv.org/abs/2106.02034) look at token sparsification as an answer, but dynamic token routing is systematically studied through the authors.

 Let us put things into perspective and understand related works with our analogy. Imagine you’re trying to find and label things in a big picture. Now, in this picture, some things are really big like trees and some are really tiny like ants.

	1.	Why multiple-scale features are needed: It’s like using different magnifying glasses for different items in the picture. A big magnifying glass helps you see small things (like ants) clearly, while a smaller magnifying glass might be best for the bigger things (like trees).
	2.	Traditional methods: Think of it as using a set of fixed magnifying glasses (each one has its own zoom level). Some methods, like RetinaNet and Mask R-CNN, already do this. They decide which magnifying glass to use based on the size of the thing you’re trying to find in the picture.
	3.	The new idea (dynamic-scaling networks): Instead of always using the same set of magnifying glasses for every picture, what if you could pick and choose the best ones for each picture? That’s the idea here. For each part of an image, the network dynamically selects the best “magnifying glass” or scale to use.

 
<img width="467" alt="Screenshot 2023-08-10 at 8 10 05 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/1812f17f-97c5-4c80-9019-5584b01eefa3">

Imagine you’re making a robot to sort LEGO bricks by size and color. The robot has a special camera (this is our “vision transformer”) that looks at the LEGO pieces, but sometimes it gets confused when the LEGO pieces are very different in size or color.

	1.	What the paper is doing: They’re trying to make this robot’s camera smarter. They’re adding two special features to help it sort better: one to adjust for size (dynamic-scaling) and one to adjust for color complexity (dynamic-depth).
	2.	Dynamic-Vision-Transformer (DiT): This is the name for their smarter robot camera. It can adjust how it looks at the LEGO bricks on-the-fly.
	3.	The grid-like network: Think of this like a big game board. Each square on the board helps decide how the robot camera should adjust its view. Each square is a decision-making spot, which they call a “token routing module.”
	4.	Token routing module: This is the brainy part of each square on the game board. It decides how to adjust the view for each LEGO piece it’s looking at. It predicts whether it should zoom in, zoom out, or change the focus based on the LEGO’s size and color.
	5.	Two binary gates: This is like having two switches at each decision-making square. One switch (vertical path) decides if the robot should zoom in or out (for size), and the other switch (horizontal path) decides how to adjust for the LEGO color complexity.
	6.	End goal: By doing all this fancy adjusting, the robot camera (or vision transformer) gets better at recognizing and sorting different LEGO pieces by their size and color.

<img width="484" alt="Screenshot 2023-08-10 at 8 13 41 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/42e4386c-ddd8-4ec6-8f07-04e569793315">

### Okay we are sold on Dynamic Token Routing, but how does it work?

To understand how Dynamic Token Routing works and not get sucked into a whirlpool of math and architecture in the process, we must keep our analogy at hand.
Images are distilled into their features (our robots), and these features then roam free on a grid-like structure (our chess board). At every node of that grid, our features (robots) must choose either of the three ways to move, as they get better at choosing, the network gets better at learning.

![IMG-2134](https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/0524b5fa-1dba-4b12-a4ff-ba8724790341)

 If you are at a DL practitioner you are already crying thinking of the shapes of the tensors. The paper is surprisingly very detailed in providing the necessary information.

 Imagine you’re analyzing a photograph using a microscope.

	1.	Dividing the Image: Initially, you take the image and divide it into smaller sections (think of these as zoomed-in areas under your microscope). Each section captures part of the image.
	2.	Resolution Reduction: Instead of viewing the whole photograph at once, you reduce the clarity by a factor. The “H×W×3” simply means the image’s height, width, and the 3 color channels. “HW/4^2 patches” means you are segmenting the photo into smaller chunks, and after this step, each segment is 1/4 its original size.
	3.	Token Routing Space: Think of this as a multi-level laboratory where each level allows you to further analyze and process these segments. At each level, you can reduce the segment size further, making it smaller and simpler to analyze.
	4.	Progressive Downsampling: As you go deeper into these levels, the segments are downsized. By the time you reach the 4th level, your segments are tiny, being only 1/32 of their starting size.
	5.	Transformers & Stages: Within each level of this lab, you use various tools and techniques (referred to as ‘transformers’) to process and understand these segments. The term “stage” here is like a phase or step in the process, and “M” denotes how many of these phases you have in a level.

<img width="723" alt="Screenshot 2023-08-10 at 8 35 19 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/392f449b-909a-4962-a2f6-1295dcb30977">

But what are these stages or phases? What exactly happens to the tokens at the multi-level laboratory or token routing space?

	1.	Grid-Like Structure: Imagine you have a chessboard, and on every square, there’s a tiny lab station (node). Each lab station has a unique response map—sort of like a mini blueprint or a reference guide.
	2.	Response Map (Fi,j): This is a representation at a specific square (i,j) on your chessboard. You use this map as a starting point to decide what steps to take next.
	3.	Tools at each Square: At every square (or stage), you have three main tools:
	•	Patch Embedding Layer (Pi,j): Think of this as a microscope that looks closely at certain parts of your response map.
	•	Transformer Blocks (Bi,j): These are like tiny robots that process and modify the parts of the response map you’re focusing on.
	•	Identity Mapping Layer: This is like a reference guide, ensuring you remember and use the original information from your response map.
	4.	Using the Tools: When working on a square, you combine or mix information from the current square and also possibly from the neighboring squares. You might look left (i,j-1) or up (i-1,j) on the chessboard to borrow information from there. The process of deciding where to get information from and how much to mix is based on a strategy called ‘routing’.
	5.	Skipping Some Steps: Now, sometimes, you don’t need to use all the tools. Depending on your strategy, you might skip some tools on certain squares. When you skip, you put a mask on the response map—sort of like putting a ‘do not touch’ sign on certain parts.


 <img width="724" alt="Screenshot 2023-08-10 at 8 29 13 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/2e565930-34e0-4123-b7ce-f434a5bcd9b7">

A diagram is great, but at some point we need to come to terms with the math. Before we do that let us again come back to a simpler analogy to realize what steps we must take to formulate the equations.

1. Self-Dependent Token Routing:
Imagine again you have a bunch of little robots, called tokens, traveling on paths on a map (your chessboard network). Each robot looks at its own features (sort of like its own specifications or tools it’s carrying) and decides which path it should take.

2. Decision Gates:
Now, there are special checkpoints (binary gates) in the network that help these robots choose their path. These checkpoints are present at every transformer stage and patch embedding module.

3. Making the Decision with a Mask:
At each checkpoint (transformer stage), there’s a decision mask, which is kind of like a traffic light system. It can either show a green light (1) or a red light (0) for each robot, indicating whether it should proceed or stop.

4. Default Setting:
In the beginning, every traffic light is set to green (meaning every element in the decision mask is 1), allowing all robots to move freely.

5. Adapting to the Data:
However, we don’t always want all robots to go in the same direction or use the same path. So, we use a specific mechanism (a linear projection) that looks at the robot’s features and the entire map’s scenario to decide which traffic lights should change from green to red or vice versa.

So we need a learnable parameter that can tune which gate to open. This means we are deciding whether or not to study the image and look for features using our special lab (the transformer block)
We need to predict the probability map based on the feature map. The equation is shown here, w_row(I,j) is the linear projection here.

<img width="221" alt="Screenshot 2023-08-10 at 8 47 05 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/f1dcaa61-9d5a-4541-a099-2ab10c559e5d">

The probability map cannot be left continous since the aim is to do token-specific routing. Tokens are discrete in nature. So Gumbel Softmax is used to sample the binary decision mask from the above probability map

<img width="238" alt="Screenshot 2023-08-10 at 8 49 10 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/8795f440-bf49-44f3-9e2a-d544d2a528c3">

The linear projection distribution w_row(i,j) is formulated as a moving average to maintain a stable gating module for the transformer blocks. w_row(i,j) are updated during back-propogation.

<img width="202" alt="Screenshot 2023-08-10 at 8 53 10 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/dc367911-b0e5-4074-8478-3e1925708b2f">

The above equations were for the row axis of our grid structure, we also need the column axis. This means formulating the learnable parameter which understands which gates to open for the downsampling (zooming in) module
This means a smaller object's feature will be stopped at the beginning of the grid so that the ant doesn't become a speck, while the bigger object's (elephants) feature will be downsampled further. 

Thus for the downsampling or patch embedding layers we have a similar linear projection w_col(i,j) that is maintained using a moving average the same as the row parameter. This is also updated during backpropagation. The column probability distribution and Gumbel softmax are similarly expressed.
<img width="223" alt="Screenshot 2023-08-10 at 8 58 57 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/f5143017-b604-4c62-83fe-8471ea1ae542">

At this stage we should also recap how these equations align with our rudimentary grid like structure.

![IMG-2135](https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/d82feb11-bd70-40fa-8999-d00bc6251b6a)

The actual equation governing the entire gating mechanism is such. 

<img width="458" alt="Screenshot 2023-08-10 at 9 00 40 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/2a8eea80-879d-498c-bbe8-055975c0d777">

The authors also devise a complexity constraint, to have a trade-off between effectiveness and complexity. Essentially they formulate a cost with respect to FLOPs for each stage (transformer and patch embedding). They name this cost C_space, also calculate the cost for PVTv2 (Pyramid Vision Transformer architecture, the baseline that they are improving on), and name it C_base. They create a loss function as a combination of C_base and C_space. Add that loss to the overall loss function of the whole network and jointly optimize it during training.

<img width="469" alt="Screenshot 2023-08-10 at 9 58 20 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/3614bf7f-4945-4202-a6df-61c04fe7eafb">

### But what about validation?

The authors conduct experiments for three task levels.

1. Image Classification on ImageNet-1K
2. Object Detection and Image Segmentation
3. Semantic Segmentation

The authors report a SOTA with comparison against GFLOPs against all three of these tasks.

<img width="478" alt="Screenshot 2023-08-10 at 10 39 00 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/ad44bc00-daa1-4ca2-aef5-a748cffe1810">
<img width="520" alt="Screenshot 2023-08-10 at 10 38 52 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/df447f92-9a0f-4eb8-b8ac-8729095d619b">
<img width="472" alt="Screenshot 2023-08-10 at 10 38 41 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/9d876458-c7b6-4b6a-9182-e5a6292ea9fd">

The authors also report an ablation study where they study the effect of not skipping any gate (fully connected  grid). They report only a slight increase in improvement. They also study random and attention augmented mask genertion as opposed to lernable mask generation as used by their startegy. Thye reprted that their strtaegy was superior in accuracy.


<img width="496" alt="Screenshot 2023-08-10 at 10 42 57 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/8322cbd3-cae1-4ecb-b4fd-cad846193103">

The authors leave us with the idea, that our models must be accomodating to the needs of the data. We must not aim or settle for the one bill fits all architecture but rather strive to find models that serve the smallest regions of the image with the same dexterity that they serve the largest one with. It is their belief that Dynamic Token Routing can be applied to monolithic vision transformers as well as languge domain. 

It is upto us, how we employ our small robots on our custom chess-boards.












