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


[Imgur](https://imgur.com/gk5spAx)


** Formatted Instance Construction**

So how do we make special "instruction-formatted" examples to train LLMs (Large Language Models) to follow instructions? This is like teaching a robot by giving it specific tasks in a detailed manner. There are three ways to make these examples:

1. **Formatting Task Datasets**: Before this fancy method of "instruction tuning", people had some datasets with various tasks like summarizing text or translating. Now, they are turning these into instruction-like tasks. Think of it as converting a set of recipes into a cooking show script. For example, for a question-answering task, they might add a phrase like “Please answer this question” before each example. The model then learns to recognize this instruction and perform the task.

2. **Formatting Daily Chat Data**: Using actual questions and problems people have asked about online. This is like taking real-life questions from a game show and training the robot on them to prepare it for the next round. Some models have used real-world chats and conversations to train and even used these conversations to make the model safer and avoid harmful instructions.

3. **Formatting Synthetic Data**: Instead of using human-made examples, they use LLMs to create new examples. It's like when you ask a robot to make up a new task based on the tasks it already knows. This way, you get a lot more examples without much effort.

**Key Factors for Instance Construction**:

* **Scaling the instructions**: Just like giving a student a broader range of questions can make them smarter, increasing the number and variety of tasks helps the model to generalize better. But after a certain point, adding more tasks doesn’t help much. It's similar to studying for a test - after a certain amount of studying, you might not gain much more knowledge.

* **Formatting design**: How you design and format these instructions is important. Adding examples and demonstrations can help. It’s like a cooking show - seeing a dish being made step-by-step can help more than just reading a recipe. But, adding too much extra information (like the history of an ingredient) might not help and could even confuse.

In the end, what matters most is having a diverse and high-quality set of instructions, not necessarily a huge number of them. It's better to have a varied set of high-quality study materials than to cram with tons of repetitive information. And while there's no perfect way to come up with these training tasks, they can either reuse existing ones or use models to create new ones.