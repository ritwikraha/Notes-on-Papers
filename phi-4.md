# phi-4 : Deep Dive


## Inference

'''
import transformers

pipeline = transformers.pipeline(
    "text-generation",
    model="microsoft/phi-4",
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a mad scientist dedicated to teaching humanity everything."},
    {"role": "user", "content": "What exactly is a manifold?"},
]

outputs = pipeline(messages, max_new_tokens=128)
print(outputs[0]["generated_text"][-1])
'''