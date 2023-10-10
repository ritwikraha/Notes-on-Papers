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


<img src="https://imgur.com/gk5spAx" alt="" width="400">

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

In many ways Instruction Tuning is more efficient than fine-tuning, it does not require expensive training on a specific corpus of data.

There are four important aspects to consider while talking about instruction tuning.

<img src="https://imgur.com/vfRnxZ7" alt="" width="400">

**Balancing the Data Distribution**: 
This section is all about ensuring that during training, we don't overemphasize or neglect any specific task. Think of it like preparing a varied diet; you don't want to eat too much of one thing and miss out on other essential nutrients. So, models use strategies like 'examples-proportional mixing' which is like making a smoothie where every ingredient gets an equal portion. However, sometimes they might increase the amount of a particularly beneficial ingredient. They also put a cap on how much of one type of data can be used so that no single type dominates.

**Combining Instruction Tuning and Pre-Training**: 
Imagine training for a marathon. Instead of just doing long runs (pre-training) and then sprints (instruction tuning) separately, some people might find it beneficial to mix both types of training. In the model training world, this is akin to blending standard pre-training data with task-specific data, so the model gets the benefits of both at the same time.

**Multi-stage Instruction Tuning**: 
Here's an analogy: think of teaching someone to cook. First, you teach them basic recipes (task-formatted instructions) and then let them experiment and create their dishes (daily chat instructions). That's a two-stage process. If we return to basics after the creative part, we reinforce those foundational skills. Similarly, in model training, the process might involve multiple stages with different types of instruction data to ensure the model remembers and performs well.


**Other tricks**:

**1. Efficient training for multi-turn chat data**: 
Imagine having a long conversation where you revisit some topics multiple times. Instead of going through the entire conversation every time you want to understand one topic, what if you just focused on the parts that matter for that topic? Similarly, a method called 'Vicuna' lets models train on conversations without constantly revisiting repetitive parts. This makes the training quicker.

*Analogy*: It's like using bookmarks in a long book to quickly find the chapters you want to reread, instead of going through the whole book each time.

**2. Filtering low-quality instructions using LLMs**: 
Not all instructions are good. It's like sifting through a basket of apples and removing the bad ones. Some models use a strategy where they have a powerful AI (like GPT-4) look at some of these instructions and grade them. Once the AI has graded enough of them, these grades can be used to train a system to sift through all the other apples (instructions) and pick out the bad ones.

*Analogy*: Imagine using an expert fruit picker to teach a new worker how to select the best fruits. Once the new worker has seen enough examples, they can pick good fruits on their own.

**3. Establishing self-identification for LLM**: 
Just like how people introduce themselves at the beginning of a conversation ("Hi, I'm John from TechCorp"), models can be trained to do the same. This makes sure that users know who or what they're talking to. For example, an AI might introduce itself as, "Hello! I'm CHATBOTNAME, created by DEVELOPER."

*Analogy*: It's like wearing a name badge at a conference. Even if someone hasn't met you before, they can quickly learn your name and affiliation.

**4. Other Practical Tricks**: 
Besides the major strategies, some smaller tricks can be helpful, like:
- Joining multiple examples to make the most of the model's capacity.
- Using a special kind of evaluation to check the quality of instructions.
- Rewriting simple instructions to make them more detailed or complex.

*Analogy*: It's like using small hacks in cooking, such as using a squeeze of lemon to keep apples from browning or chilling cookie dough to get thicker cookies. These tricks aren't essential, but they can enhance the final result.


### Alignment Tuning


**Problem**:
Large Language Models (LLMs) are like really smart robots that can handle a lot of language tasks. But, like in movies where robots sometimes go rogue, these models can also sometimes misbehave. They might make up things, follow the wrong goals, or even say harmful stuff. One big reason for this is that, during their training, these models were only taught to predict the next word, not what's morally right or wrong.

**Solution - Human Alignment**:
Imagine if we could give these robot models a moral compass, guiding them to behave more like how humans expect. This is the idea of "human alignment". But there's a downside: if we put too many rules, the model might lose some of its smartness, a problem some experts call the "alignment tax".

**Main Alignment Criteria**:

1. **Helpfulness**: 
   * What it means: The model should always try its best to help users. If a user asks a vague question, the model should try to get more details. It should also be sensitive and careful in its replies.
   * Challenge: It's tough to ensure this because everyone's idea of "helpful" might be different.
   
   *Analogy*: It's like a librarian who not only gives you the book you ask for but also suggests related readings or asks more about what you need.

2. **Honesty**: 
   * What it means: The model should always tell the truth. It should not make stuff up. And if it's unsure about something, it should admit it.
   * Benefit: This rule is more straightforward than the others, so it might be easier to implement.
   
   *Analogy*: It's like a friend who, if they don't know the answer to your question, admits they don't know rather than guessing or making something up.

3. **Harmlessness**: 
   * What it means: The model should never say things that can hurt someone or encourage bad actions. For example, if someone asks it how to do something illegal, it should refuse to answer.
   * Challenge: What's considered "harmful" can vary from person to person or from one culture to another.
   
   *Analogy*: It's like a mentor who refuses to give you harmful advice, even if you ask for it.

**How to Achieve Alignment?**:
A cool method some experts use is called "red teaming". It's like a game where experts play the role of hackers trying to make the model say bad things. When they succeed, they then teach the model not to make that mistake again.

*Overall Analogy*: Imagine training a dog. You reward it when it behaves well and correct it when it misbehaves. Human alignment for LLMs is similar, but instead of training dogs, we're training smart robot-like models to behave in line with human expectations.


**Reinforcement Learning from Human Feedback (RLHF)**

---

**Background**:
Let's imagine you're training a pet dog. You want it to listen to your commands and act accordingly. However, sometimes it might not understand or might misbehave. You reward the dog when it behaves well and guide it when it doesn’t. Similarly, Large Language Models (LLMs) are like these pets, and we need to train them to respond well to our commands. To do this, experts use a method called Reinforcement Learning from Human Feedback (RLHF).

---

**How RLHF Works**:
1. **Use a Pre-trained Model**: This is like the natural instincts a pet dog already has before you start training it. LLMs start with some knowledge.
  
2. **Reward Model**: This acts as the reward system. Think of it as the treats you give your dog when it behaves well. This model gives scores (or rewards) to the answers the main model produces based on how good or bad they are.
  
3. **RL Algorithm**: It's like the method you use to train your dog. It helps the model improve by using the scores from the reward model.

---

**Key Steps of RLHF**:
1. **Supervised Fine-tuning**: Initially, you guide the model with specific commands and desired outputs. It's like teaching your dog basic commands like sit, stay, or fetch.

2. **Reward Model Using Human Feedback**: This is where humans rank different answers the model gives, teaching it which responses are good and which are not.

3. **RL Fine-tuning**: Now, we use the rewards from the previous step to improve the model further, making it even better at its tasks. It's like advanced training for your dog after it has mastered the basics.

---

**Practical Tips for RLHF**:

1. **Effective Reward Model Training**:
   * *Size Matters*: Bigger reward models might be better because they can judge the quality of the answers more accurately.
   * *Avoid Overfitting*: You don’t want the model to just memorize stuff. It should think and react intelligently.
   * *Multi-criteria Reward Models*: Just like in school, where you have different teachers for different subjects, having different reward models for different criteria can be helpful.

2. **Effective RL Training**:
   * *Start with a Good Base*: Before advanced training, ensure the model is already pretty good.
   * *Iterative Improvement*: Continuously improve the model by training it in stages, making it better and better.

3. **Efficient RL Training**:
   * *Separate Servers for LLM and Reward Model*: To save time and resources, it's like having two trainers train your dog simultaneously.
   * *Use Beam Search Decoding*: A trick to make the training process faster and more diverse.

---

*Overall Analogy*: RLHF is like an advanced dog training system. You start with a dog that knows some basic stuff, then you train it with specific commands, reward it when it does well, and use those rewards to make it even better. Along the way, you use some tricks and techniques to make sure the dog becomes the best version of itself.


**Non-Reinforcement Learning (Non-RL) Alignment Approaches for LLMs**

---

**Background**:
Let's think of LLMs as students in school. The traditional method (RLHF) was like training them using a complex point system that tells them when they're doing well or not. This method, although effective, was complicated and needed a lot of resources. So, experts thought, "Why not simplify this? Instead of the complex point system, why not teach them directly using a good textbook?"

---

**Limitations of RLHF**:
1. **Multiple Models at Once**: It's like trying to teach several students different subjects all at once - overwhelming!
2. **Complexity of PPO**: Imagine a very complicated scoring system for students, which sometimes gives unpredictable results.

---

**Non-RL Alignment Overview**:
Instead of the complex method, this approach is more straightforward. It directly teaches the LLM (student) using a specially made textbook (alignment dataset). This textbook contains correct answers and safe behaviors, teaching the model directly.

---

**Two Key Steps**:
1. **Creating the Textbook (Alignment Dataset)**: 
   * This can be done by already well-behaved models or using human guidelines.
   * It's like creating a textbook based on what top students or experienced teachers know.
   * Some methods even use previous point-based evaluations to pick the best answers and include them in this textbook.

2. **Teaching Using the Textbook (Fine-tuning with the Dataset)**:
   * This involves teaching the LLM directly from this textbook.
   * Just like how students are taught using textbooks in school. Sometimes, they also have extra tasks, like ranking answers or comparing questions and answers to help them understand better.

---

*Overall Analogy*: If RLHF was like a complicated scoring system to teach students, the non-RL alignment approach is like a direct and simple classroom teaching method. Instead of constantly evaluating and correcting, you give the student a good textbook and guide them through it. It's simpler, direct, and often just as effective.