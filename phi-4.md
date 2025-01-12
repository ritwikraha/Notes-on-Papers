# phi-4 : Deep Dive


## Inference

```
from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline(
    "text-generation",
    model="microsoft/phi-4",
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto",
)

# Define conversation messages
messages = [
    {"role": "system", "content": "You are a stand-up comedian who moonlights as a theoretical physicist."},
    {"role": "user", "content": "Can you explain what a manifold is?"},
]

# Generate the response
outputs = generator(messages, max_new_tokens=128)

# Print the generated text
print(outputs[0]["generated_text"][-1])
```