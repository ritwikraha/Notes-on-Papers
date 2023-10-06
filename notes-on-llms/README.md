# Notes on LLMs

## What is Fine-Tuning

LLMs can acquire the general abilities for solving various tasks from pre-training. However, an LLM can be further adapted to solve a specific task through a process called fine-tuning.

Imagine training an MMA (mixed martial arts) fighter, you equip them with various skills in boxing, cardio and fighting in general but as they grow and prepare for a specific opponent you equip them with specific skills like grappling, judo and so forth. Training an LLM and then fine-tuning it is somewhat similar.

With Fine-Tuning we must be aware of four key sub-categories:

- Instruction Tuning
- Alignment Tuning
- Parameter Efficient Model Adaptation
- Memory Efficient Model Adaptation

### Instruction Tuning

Instruction tuning is a method used to improve Large Language Models (LLMs). Instead of just giving the model normal data, you give it data formatted in a specific way that's more like natural language instructions. This method is related to other fine-tuning strategies. To do instruction tuning, you first gather or create these special instruction-like data sets. Then, you train the LLM using this data in a typical supervised learning manner. Once done, the LLM becomes better at handling tasks it hasn't seen before, even in different languages.


![image](https://imgur.com/gk5spAx)

#### Formatted Instance Construction 

So how do we make special "instruction-formatted" examples to train LLMs (Large Language Models) to follow instructions? This is like teaching a robot by giving it specific tasks in a detailed manner. There are three ways to make these examples:

1. **Formatting Task Datasets**: Before this fancy method of "instruction tuning", people had some datasets with various tasks like summarizing text or translating. Now, they are turning these into instruction-like tasks. Think of it as converting a set of recipes into a cooking show script. For example, for a question-answering task, they might add a phrase like “Please answer this question” before each example. The model then learns to recognize this instruction and perform the task.

2. **Formatting Daily Chat Data**: Using actual questions and problems people have asked about online. This is like taking real-life questions from a game show and training the robot on them to prepare it for the next round. Some models have used real-world chats and conversations to train and even used these conversations to make the model safer and avoid harmful instructions.

3. **Formatting Synthetic Data**: Instead of using human-made examples, they use LLMs to create new examples. It's like when you ask a robot to make up a new task based on the tasks it already knows. This way, you get a lot more examples without much effort.

**Key Factors for Instance Construction**:

* **Scaling the instructions**: Just like giving a student a broader range of questions can make them smarter, increasing the number and variety of tasks helps the model to generalize better. But after a certain point, adding more tasks doesn’t help much. It's similar to studying for a test - after a certain amount of studying, you might not gain much more knowledge.

* **Formatting design**: How you design and format these instructions is important. Adding examples and demonstrations can help. It’s like a cooking show - seeing a dish being made step-by-step can help more than just reading a recipe. But, adding too much extra information (like the history of an ingredient) might not help and could even confuse.

In the end, what matters most is having a diverse and high-quality set of instructions, not necessarily a huge number of them. It's better to have a varied set of high-quality study materials than to cram with tons of repetitive information. And while there's no perfect way to come up with these training tasks, they can either reuse existing ones or use models to create new ones.

#### Instruction Tuning Strategies

In many ways Instruction Tuning is more efficient than fine-tuning, it does not require expensive training on specific corpus of data.

There are four important aspects to consider while talking about instruction tuning.


**Balancing the Data Distribution**: 
This section is all about ensuring that during training, we don't overemphasize or neglect any specific task. Think of it like preparing a varied diet; you don't want to eat too much of one thing and miss out on other essential nutrients. So, models use strategies like 'examples-proportional mixing' which is like making a smoothie where every ingredient gets an equal portion. However, sometimes they might increase the amount of a particularly beneficial ingredient. They also put a cap on how much of one type of data can be used so that no single type dominates.

**Combining Instruction Tuning and Pre-Training**: 
Imagine training for a marathon. Instead of just doing long runs (pre-training) and then sprints (instruction tuning) separately, some people might find it beneficial to mix both types of training together. In the model training world, this is akin to blending standard pre-training data with task-specific data, so the model gets the benefits of both at the same time.

**Multi-stage Instruction Tuning**: 
Here's an analogy: think of teaching someone to cook. First, you teach them basic recipes (task-formatted instructions) and then let them experiment and create their dishes (daily chat instructions). That's a two-stage process. If we return to basics after the creative part, we reinforce those foundational skills. Similarly, in model training, the process might involve multiple stages with different types of instruction data to ensure the model remembers and performs well.

**Other Practical Tricks**: 
Just like little hacks you might find in everyday tasks (like using vinegar to clean or putting batteries in the fridge to last longer), when training language models, there are also little tips and tricks that can make the training better:

**Summary & Simple Explanation**:

**1. Efficient training for multi-turn chat data**: 
Imagine having a long conversation where you revisit some topics multiple times. Instead of going through the entire conversation every time you want to understand one topic, what if you just focused on the parts that matter for that topic? Similarly, a method called 'Vicuna' lets models train on conversations without constantly revisiting repetitive parts. This makes the training quicker.

*Analogy*: It's like using bookmarks in a long book to quickly find the chapters you want to reread, instead of going through the whole book each time.

**2. Filtering low-quality instructions using LLMs**: 
Not all instructions are good. It's like sifting through a basket of apples and removing the bad ones. Some models use a strategy where they have a powerful AI (like GPT-4) look at some of these instructions and grade them. Once the AI has graded enough of them, these grades can be used to train a system to sift through all the other apples (instructions) and pick out the bad ones.

*Analogy*: Imagine using an expert fruit-picker to teach a new worker how to select the best fruits. Once the new worker has seen enough examples, they can pick good fruits on their own.

**3. Establishing self-identification for LLM**: 
Just like how people introduce themselves at the beginning of a conversation ("Hi, I'm John from TechCorp"), models can be trained to do the same. This makes sure that users know who or what they're talking to. For example, an AI might introduce itself as, "Hello! I'm CHATBOTNAME, created by DEVELOPER."

*Analogy*: It's like wearing a name badge at a conference. Even if someone hasn't met you before, they can quickly learn your name and affiliation.

**4. Other Practical Tricks**: 
Besides the major strategies, there are some smaller tricks that can be helpful, like:
- Joining multiple examples to make the most of the model's capacity.
- Using a special kind of evaluation to check the quality of instructions.
- Rewriting simple instructions to make them more detailed or complex.

*Analogy*: It's like using small hacks in cooking, such as using a squeeze of lemon to keep apples from browning or chilling cookie dough to get thicker cookies. These tricks aren't essential, but they can enhance the final result.
