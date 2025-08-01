
# Modern ML Model Architectures

This guide provides clear explanations, diagrams, and practical code for the most important modern machine learning model architectures.

---

## Transformer Models

Transformers are the foundation of most modern NLP and vision models. Their key innovation is the **self-attention mechanism**, which allows the model to weigh the importance of different input tokens.

**Key Concepts:**
- **Self-attention:** Each token attends to every other token, capturing context.
- **Multi-head attention:** Multiple attention mechanisms run in parallel, allowing the model to focus on different parts of the input.
- **Positional encoding:** Since transformers lack recurrence, positional encodings inject information about token order.
- **Encoder-decoder architecture:** Used in tasks like translation (e.g., original Transformer, T5).

**Diagram:**
```
Input Embeddings + Positional Encoding
        │
   ┌───────────────┐
   │ Self-Attention│
   └───────────────┘
        │
   ┌───────────────┐
   │ Feed Forward  │
   └───────────────┘
        │
   (Stacked N times)
```

**PyTorch Example: Transformer Block**
```python
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)
```

---

## BERT and Variants

BERT (Bidirectional Encoder Representations from Transformers) is a transformer encoder trained to deeply understand context in both directions.

**Key Concepts:**
- **Bidirectional training:** Looks at both left and right context.
- **Masked language modeling (MLM):** Randomly masks words and predicts them, forcing deep understanding.
- **Fine-tuning:** Pretrained BERT can be adapted to many tasks with minimal changes.
- **Distilled versions:** Smaller, faster models (e.g., DistilBERT) for efficiency.

**Practical Example: HuggingFace BERT**
```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # (batch_size, seq_len, hidden_dim)
```

**Fine-tuning BERT for Classification**
```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# Train on your dataset using Trainer or PyTorch training loop
```

---

## GPT Models

GPT (Generative Pretrained Transformer) models are decoder-only transformers trained to predict the next token (autoregressive).

**Key Concepts:**
- **Autoregressive generation:** Predicts next token given previous ones.
- **Scaling properties:** Larger models (GPT-2, GPT-3, GPT-4) show emergent abilities.
- **In-context learning:** Can perform tasks by seeing examples in the prompt.
- **Prompt engineering:** Carefully crafting input prompts to guide model behavior.

**Example: Text Generation with GPT-2**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=30)
print(tokenizer.decode(outputs[0]))
```

---

## LLAMA and Open Source LLMs

LLAMA (Large Language Model Meta AI) and similar open-source LLMs (e.g., Falcon, Mistral) are transformer-based models designed for research and deployment.

**Key Concepts:**
- **Architecture:** Similar to GPT, with optimizations for efficiency and scaling.
- **Training methodology:** Trained on large, diverse datasets; often use mixed-precision and distributed training.
- **Fine-tuning:** LoRA, QLoRA, and other parameter-efficient methods allow adaptation to new tasks with less compute.
- **Deployment:** Quantization and model distillation enable running LLMs on consumer hardware.

**Example: Loading LLAMA with HuggingFace**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
inputs = tokenizer("What is the capital of France?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=20)
print(tokenizer.decode(outputs[0]))
```

---

## Multi-Modal Models

Multi-modal models process and generate data from multiple modalities (e.g., text, images, audio).

**Key Concepts:**
- **Text-image models:** CLIP, BLIP, and others learn joint representations of text and images.
- **Cross-modal attention:** Allows the model to align and relate information across modalities.
- **Joint embeddings:** Embeds different modalities into a shared space for comparison or retrieval.
- **Generative capabilities:** Models like DALL-E, Stable Diffusion generate images from text prompts.

**Example: CLIP for Text-Image Similarity**
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
image = Image.open(requests.get("https://images.unsplash.com/photo-1519125323398-675f0ddb6308", stream=True).raw)
inputs = processor(text=["a photo of a dog", "a photo of a cat"], images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)
```

**Example: Generating Images with Stable Diffusion (Diffusers)**
```python
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
image = pipe("A futuristic cityscape at sunset").images[0]
image.save("output.png")
```

---

## Further Reading & Practice

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)

**Practice:**
- Implement a simple transformer from scratch (see annotated PyTorch code above)
- Fine-tune BERT or GPT on a custom dataset
- Try prompt engineering with GPT-3/4 or open-source LLMs
- Use CLIP to search for images by text
- Generate images from text using Stable Diffusion
```
