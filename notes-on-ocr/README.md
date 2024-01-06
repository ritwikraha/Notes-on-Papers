# DB for Dummies

Status: In progress

Wait so DB is not a DataBase?

And I will keep the dad jokes to myself for the rest of this blogpost. So in all seriousness what is DB in image processing?

## DB = Differential Binarization

Differential Binarization is a technique used in the field of text detection from images. But what is it, and why does one need to bother about it?

## What is Binarization?

1,0? Two words too many? Binarization, in simple terms is the process of filtering a signal to 0 and 1, so that it can be represented as a ‘binary’ sequence. For images it refers to the binary map extracted from an image.

**Binarization**: The process of binarization involves converting this probability map \( P \) into a binary map \( B \), which is another matrix of the same size. In the binary map, each element \( B_{i,j} \) can only take one of two possible values: 0 or 1.

**Wait, what’s a** **Probability Map \( P \)**: This is a matrix or 2D array of values that are produced by a segmentation network. The values in this map are probabilities, where each element \( P_{i,j} \) represents the likelihood that the pixel at position \( (i, j) \) is part of a text area. The size of this map is given by the height \( H \) and width \( W \) of the image that was processed by the segmentation network.

The binarization process can be represented by the following equation:

$$
 B_{i,j} = \begin{cases}
1 & \text{if } P_{i,j} \geq t \\
0 & \text{otherwise}
\end{cases} 
$$

Here's what this equation means in plain language:

- Look at each pixel's probability in the map \( P \).
- Compare this probability to the threshold \( t \).
- If the probability is greater than or equal to \( t \), set the corresponding pixel in the binary map \( B \) to 1, meaning it's likely part of a text area.
- If the probability is less than \( t \), set the corresponding pixel in the binary map \( B \) to 0, meaning it's likely not part of a text area.

This binary map can then be used for further processing, like extracting text from an image because it clearly defines which areas are text (marked by 1) and which are not (marked by 0).

![cover-1.png](DB%20for%20Dummies%20b5a3942e79b4490890b94d49369a6410/cover-1.png)

![cover-1-binarized.png](DB%20for%20Dummies%20b5a3942e79b4490890b94d49369a6410/cover-1-binarized.png)

The white region on the right image is the the text (marked by 1) and the rest of the region is not (marked by 0).

## Why Differential Binarization?

Binarization has long been recognized as a useful tool in computer vision, often serving as a fundamental step in processing and analyzing images. At its core, binarization involves converting a grayscale image to a binary image, typically through a straightforward and simple equation. However, a question arises: why is there a need for Differential Binarization in this context? The answer lies in the nature of the binarization function, which is inherently discrete. 

Differential Binarization aims to transform this discrete function into a differentiable one. But why is this transformation necessary? The reason is rooted in the principles of deep learning. In a deep learning architecture, components are generally more effective and yield better performance when they are differentiable. This characteristic allows for more efficient and accurate computations, particularly in the backpropagation process used for training neural networks. Therefore, Differential Binarization aligns the binarization process with the needs of deep learning systems, enabling more seamless integration and improved performance in computer vision tasks.

Updated function for Binarization is as follows:

where \hat{B} is the approximate binary map ;T is the adaptive threshold map learned from the network; k indicates the amplifying factor

[https://files.oaiusercontent.com/file-uXAsbOmk5LmuTViXlg4RNkuc?se=2024-01-04T16%3A43%3A35Z&sp=r&sv=2021-08-06&sr=b&rscc=max-age%3D299%2C%20immutable&rscd=attachment%3B%20filename%3DScreenshot%25202024-01-04%2520at%252012.01.15%25E2%2580%25AFPM.png&sig=VM4r/ACbEqRBhm57JYNnlZmd6P14XaILU%2BPMe30jWyE%3D](https://files.oaiusercontent.com/file-uXAsbOmk5LmuTViXlg4RNkuc?se=2024-01-04T16%3A43%3A35Z&sp=r&sv=2021-08-06&sr=b&rscc=max-age%3D299%2C%20immutable&rscd=attachment%3B%20filename%3DScreenshot%25202024-01-04%2520at%252012.01.15%25E2%2580%25AFPM.png&sig=VM4r/ACbEqRBhm57JYNnlZmd6P14XaILU%2BPMe30jWyE%3D)

\[
\hat{B}*{i,j} = \frac{1}{1 + e^{-k(P*{i,j}-T_{i,j})}}
\]

But why does this work and more importantly when does it work? To understand that - we have to first realise when the above term becomes zero and when does it become 1.

So when would \(\hat{B}_{i,j}\) be zero:

1. The logistic function \(\frac{1}{1 + e^{-x}}\) ranges between 0 and 1 for all real numbers \(x\).
2. The only time a fraction equals zero is when the numerator is zero. However, in this logistic function, the numerator is always 1, so the function itself can never be exactly zero.
3. Since \(\hat{B}*{i,j}\) is a logistic function, it will approach zero as the exponent \(k(P*{i,j}-T_{i,j})\) goes to negative infinity.
4. To make \(k(P_{i,j}-T_{i,j})\) approach negative infinity, \(P_{i,j}\) would have to be much less than \(T_{i,j}\), assuming \(k\) is positive.
5. Practically, \(\hat{B}*{i,j}\) can be considered approximately zero when \(P*{i,j}\) is sufficiently less than \(T_{i,j}\) such that the value of the logistic function is so close to zero it is effectively zero for the purposes of the model or application.

Thus it becomes clear that \(\hat{B}*{i,j}\) will never be exactly zero, but it can be so close to zero that it is effectively zero, which happens when \(P*{i,j}\) is significantly less than \(T_{i,j}\), given that \(k\) is a positive constant. 

Similarly when would be one:

1. When the denominator turns to 1 
2. When the 1 + e^{-x} term results to 0. 
3. This would happen if  *the exponent \(k(P*{i,j}-T_{i,j})\) goes to positive infinity

This *only happens when* \(T_{i,j}\) is significantly less than *\(P*{i,j}\ , given that \(k\) is a positive constant. 

[db_desmos.mov](DB%20for%20Dummies%20b5a3942e79b4490890b94d49369a6410/db_desmos.mov)

But how does this work? What does it all mean with respect to the image we are trying to binarize?

1. **Let us first look at the** **DB Function \( f(x) \)**: 
\(\frac{1}{1 + e^{-x}}\) 
This function is a modified version of the logistic (sigmoid) function. It takes an input \( x \), which is the difference between the probability of a pixel being text \( P_{i,j} \) and the true label \( T_{i,j} \). The true label indicates whether the pixel is actually part of the text (1) or not (0). The constant \( k \) is a factor that can adjust the steepness of the function.
2. **Next we take the** **Binary Cross-Entropy Loss**: This is a common loss function used in binary classification tasks. It measures the difference between the predicted probability \( P_{i,j} \) and the actual label \( T_{i,j} \). The loss is calculated differently for positive and negative labels, which are represented by \( l_+ \) and \( l_- \), respectively.
    1. Binary Cross-Entropy, often used in classification problems, is a loss function that measures the difference between two probability distributions: the true distribution (actual labels) and the predicted distribution (predicted probabilities). Here's a step-by-step explanation with formulas:
    2. The formula for BCE is: 
    \[ \text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i)] \]
    
    Where:
    
    - \( N \) is the number of observations.
    - \( y_i \) is the actual label of the \( i \)-th observation, which can be 0 or 1.
    - \( p_i \) is the predicted probability of the \( i \)-th observation being in class 1.
3. **Loss Equations**:
    - For a positive label (when the pixel is part of the text), the loss \( l_+ \) is calculated using:
    \[ l_+ = -\log \left( \frac{1}{1 + e^{-kx}} \right) \]
    - For a negative label (when the pixel is not part of the text), the loss \( l_- \) is calculated using:
    \[ l_- = -\log \left( 1 - \frac{1}{1 + e^{-kx}} \right) \]
4. Finding the d**erivative of the Losses**: These are used to update the weights of the neural network during training, guiding it towards better performance.
    - The derivative for \( l_+ \) with respect to \( x \) is:
    \[ \frac{\partial l_+}{\partial x} = -kf(x)e^{-kx} \]
    - The derivative for \( l_- \) with respect to \( x \) is:
    \[ \frac{\partial l_-}{\partial x} = kf(x) \]
    - It is important here to realise that the gradient term refers to the degree of change in the difference between the probability map and the threshold map that causes a change in the loss.
    - This gradient (derivative) is scaled by a factor \( k \), making the updates larger and more sensitive to the
    - The function is designed so that the gradient is larger for predictions that are far from the true label, which helps correct wrong predictions more aggressively.

The goal is to have a probability map where the predicted text areas (with high probability) match the actual text areas, and the non-text areas (with low probability) match the actual non-text areas.