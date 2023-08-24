Decoding Deep Residual Learning for Image Recognition in 2023


<img width="385" alt="Screenshot 2023-08-24 at 12 01 39 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/d80c07d8-1dc1-462a-a9e7-f8006c18b19b">

"Deep Residual Learning for Image Recognition" is a foundational paper in the deep learning community that introduced Residual Networks (ResNets). Let's break it down in simple terms:

Normally, as you add more layers to a deep learning model, it should get better at its task because it can learn more complex features. But in practice, deeper networks often don't perform as well as expected. They can be harder to train due to vanishing and exploding gradient problems.

To address this, the authors introduced a simple yet groundbreaking idea: instead of trying to learn an underlying function directly, why not just learn the difference (or "residual") between the current layer's output and the desired output? This is done using "shortcut connections" or "skip connections" that bypass one or more layers. 

Imagine you're trying to solve a math problem. Instead of calculating the answer from scratch, you use a previous problem's answer and just figure out the difference between the old problem and the new one. This "difference" is the "residual".

ResNets use these residual connections to effectively "skip" layers, making it possible to train very deep networks (like 152 layers deep) without the training process getting stuck. When tested, these ResNet architectures set new performance standards in image recognition tasks.

In everyday language: It's like trying to adjust the settings on your TV. Instead of resetting everything from scratch each time, you just tweak a little from where you left off. This makes it faster and more efficient to get the best picture quality

### The problem 

<img width="394" alt="Screenshot 2023-08-24 at 12 19 55 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/c35668eb-3494-4976-84c8-760e448918d8">

Alright, think of training a deep neural network like trying to pass a whispered message in a game of "Telephone" across a long line of friends. To make it more interesting the first person whispers a puzzle, and a group of progressively smarter friends tries to decode it. But the catch is, of course, the more friends (or layers) you add to the line, the harder it gets to keep the original message intact by the end.

A game of telephone is still not considered a formal framework, so we must once again resort to math.

F is the class of all functions together with weights and parameters, which means that for all functions f\belongs F there exists a set of parameters that can be obtained through training.

Now suppose, the function we are trying to approximate with our data X and labels Y is f*

So we can write f* = argminL(X,y,f) given f \belongs F

Now it can be hypothesized that a more powerful class of functions F, and let us call this F_bar will be better at approximating f*

So with training, we aim to find more and more powerful F_bar that will eventually approximate f*. However with F_bar not including the original F, there is always a possibility of F_bar not approximating f* at all.

So the solution is to nest the function classes as we go forward. 

In other words, this means the progressive function classes must contain the original function class in some respect.

This means making sure some of the friends in the string of friends occasionally hear the original puzzle.

<img width="450" alt="Screenshot 2023-08-24 at 2 30 02 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/7aa6d559-d3ef-4610-8412-4755d3e6e473">


In the context of the paper, they're suggesting that when training deep neural networks, instead of trying to learn a complex function directly, it's sometimes easier to learn the difference or "residual" between the input and the desired output. This is achieved using "shortcut connections" that skip certain layers. These connections help make the learning process easier without adding extra complexity

<img width="357" alt="Screenshot 2023-08-24 at 2 56 06 PM" src="https://github.com/ritwikraha/Notes-on-Papers/assets/44690292/c44db35d-e5f7-4600-9d57-ca49149580dc">


### The Solution


### Why is it still relevant?

**Vanishing Gradient Problem** 

<iframe src="https://www.desmos.com/calculator/igigkoostn?embed" width="500" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>


In deep neural networks, the vanishing gradient problem (VGP) can occur because when you multiply many small numbers together (like when you calculate gradients during backpropagation), the result becomes even smaller, leading to gradients that are too tiny to help the network learn effectively. This happens especially in deeper layers of the network.

Skip connections, also known as residual connections, help address this problem. By allowing information to "skip" layers and directly connect with deeper layers, you're essentially bypassing the multiplication steps that cause the gradients to vanish. This is like giving the robot more direct feedback about its bike-riding without going through lots of small instructions.

When you use skip connections, the activations in the deeper layers get some extra input from shallower layers, which prevents them from becoming exponentially small. This is crucial, especially when using saturating activation functions like tanh, which can make the vanishing gradient problem worse.

In simpler terms, skip connections help keep the gradient information "alive" and prevent it from becoming too small as it travels through the layers during backpropagation. This allows deeper neural networks to learn better and solve more complex tasks effectively.

https://arxiv.org/pdf/1605.06431.pdf
https://arxiv.org/pdf/1805.07477.pdf


References:

https://d2l.ai/chapter_convolutional-modern/resnet.html#residual-blocks
https://ai.stackexchange.com/questions/17764/why-do-resnets-avoid-the-vanishing-gradient-problem#:~:text=The%20VGP%20occurs%20when%20the,almost%20the%20same%20as%20d)
https://www.tensorflow.org/api_docs/python/tfm/vision/layers/ResidualBlock
https://www.desmos.com/calculator/cy3xciahm0




