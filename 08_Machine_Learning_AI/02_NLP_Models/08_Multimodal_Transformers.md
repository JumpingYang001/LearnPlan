# Multimodal Transformers

*Duration: 4 weeks*

## Overview

Multimodal transformers represent a significant evolution in AI, combining information from multiple modalities (text, images, audio, video) to create more comprehensive and versatile models. This section covers vision-language models, text-to-image generation, audio-text transformers, and practical multimodal applications.

## 1. Vision-Language Models

### Understanding Vision-Language Integration

Vision-language models bridge the gap between computer vision and natural language processing, enabling applications like image captioning, visual question answering, and cross-modal retrieval.

```python
import torch
import torch.nn as nn
from transformers import (
    CLIPProcessor, CLIPModel, CLIPTokenizer,
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM
)
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

class VisionLanguageProcessor:
    """Comprehensive vision-language processing toolkit"""
    
    def __init__(self, model_type="clip"):
        self.model_type = model_type
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load vision-language model"""
        
        if self.model_type == "clip":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        elif self.model_type == "blip":
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        elif self.model_type == "blip2":
            self.model = AutoModelForCausalLM.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        self.model.eval()
    
    def encode_image_text_pairs(self, images, texts):
        """Encode image-text pairs for similarity computation"""
        
        if self.model_type == "clip":
            # Process inputs
            inputs = self.processor(
                text=texts, 
                images=images, 
                return_tensors="pt", 
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
            
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            return image_embeds, text_embeds
        
        else:
            raise NotImplementedError(f"Encoding not implemented for {self.model_type}")
    
    def compute_similarity_matrix(self, image_embeds, text_embeds):
        """Compute similarity matrix between images and texts"""
        
        # Scale by learned temperature parameter
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text
    
    def generate_captions(self, images, max_length=50):
        """Generate captions for images"""
        
        if self.model_type in ["blip", "blip2"]:
            captions = []
            
            for image in images:
                inputs = self.processor(image, return_tensors="pt")
                
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, 
                        max_length=max_length,
                        num_beams=5,
                        early_stopping=True
                    )
                
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
            
            return captions
        
        else:
            raise NotImplementedError(f"Caption generation not supported for {self.model_type}")
    
    def answer_visual_questions(self, images, questions):
        """Answer questions about images"""
        
        if self.model_type == "blip2":
            answers = []
            
            for image, question in zip(images, questions):
                inputs = self.processor(image, question, return_tensors="pt")
                
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True
                    )
                
                answer = self.processor.decode(out[0], skip_special_tokens=True)
                answers.append(answer)
            
            return answers
        
        else:
            raise NotImplementedError(f"VQA not supported for {self.model_type}")
    
    def cross_modal_retrieval(self, images, texts, query, query_type="text"):
        """Perform cross-modal retrieval"""
        
        if self.model_type == "clip":
            # Encode all images and texts
            image_embeds, text_embeds = self.encode_image_text_pairs(images, texts)
            
            if query_type == "text":
                # Text query, retrieve similar images
                query_inputs = self.processor(text=[query], return_tensors="pt", padding=True)
                with torch.no_grad():
                    query_embed = self.model.get_text_features(**query_inputs)
                query_embed = query_embed / query_embed.norm(dim=-1, keepdim=True)
                
                # Compute similarities with images
                similarities = (query_embed @ image_embeds.T).squeeze(0)
                
            elif query_type == "image":
                # Image query, retrieve similar texts
                query_inputs = self.processor(images=[query], return_tensors="pt")
                with torch.no_grad():
                    query_embed = self.model.get_image_features(**query_inputs)
                query_embed = query_embed / query_embed.norm(dim=-1, keepdim=True)
                
                # Compute similarities with texts
                similarities = (query_embed @ text_embeds.T).squeeze(0)
            
            # Get top matches
            top_indices = similarities.argsort(descending=True)
            top_similarities = similarities[top_indices]
            
            return top_indices, top_similarities
        
        else:
            raise NotImplementedError(f"Cross-modal retrieval not supported for {self.model_type}")

# Example usage and demonstration
def vision_language_demo():
    """Demonstrate vision-language model capabilities"""
    
    # Load sample images (you would load actual images)
    image_urls = [
        "https://example.com/cat.jpg",  # Replace with actual URLs
        "https://example.com/dog.jpg",
        "https://example.com/car.jpg",
    ]
    
    # For demo, create dummy images
    dummy_images = [Image.new('RGB', (224, 224), color=(i*80, i*80, i*80)) for i in range(3)]
    
    # Sample texts
    texts = [
        "a cute cat sitting on a windowsill",
        "a golden retriever playing in the park",
        "a red sports car on a highway"
    ]
    
    # Initialize CLIP model
    vl_processor = VisionLanguageProcessor("clip")
    
    # Encode image-text pairs
    image_embeds, text_embeds = vl_processor.encode_image_text_pairs(dummy_images, texts)
    
    # Compute similarity matrix
    img_to_text_sim, text_to_img_sim = vl_processor.compute_similarity_matrix(image_embeds, text_embeds)
    
    print("Image-to-Text Similarity Matrix:")
    print(img_to_text_sim.numpy())
    
    # Cross-modal retrieval example
    query = "a furry animal"
    top_indices, similarities = vl_processor.cross_modal_retrieval(
        dummy_images, texts, query, query_type="text"
    )
    
    print(f"\nTop matches for query '{query}':")
    for i, (idx, sim) in enumerate(zip(top_indices[:2], similarities[:2])):
        print(f"  {i+1}. Image {idx}: similarity = {sim:.3f}")
    
    return vl_processor

# demo_processor = vision_language_demo()
```

### Advanced Vision-Language Architectures

```python
class CustomVisionLanguageModel(nn.Module):
    """Custom vision-language model with advanced features"""
    
    def __init__(self, vision_model_name="vit-base-patch16", 
                 text_model_name="bert-base-uncased",
                 projection_dim=512):
        super().__init__()
        
        # Vision encoder
        from transformers import ViTModel
        self.vision_encoder = ViTModel.from_pretrained(f"google/{vision_model_name}")
        
        # Text encoder
        from transformers import BertModel
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        
        # Projection layers
        self.vision_projection = nn.Linear(self.vision_encoder.config.hidden_size, projection_dim)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_vision(self, pixel_values):
        """Encode vision input"""
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_embeds = vision_outputs.last_hidden_state[:, 0]  # CLS token
        vision_embeds = self.vision_projection(vision_embeds)
        return vision_embeds
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text input"""
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state[:, 0]  # CLS token
        text_embeds = self.text_projection(text_embeds)
        return text_embeds
    
    def cross_modal_fusion(self, vision_embeds, text_embeds):
        """Perform cross-modal fusion using attention"""
        
        # Add sequence dimension for attention
        vision_seq = vision_embeds.unsqueeze(1)  # [batch, 1, dim]
        text_seq = text_embeds.unsqueeze(1)  # [batch, 1, dim]
        
        # Cross-attention: vision attends to text
        vision_attended, _ = self.cross_attention(vision_seq, text_seq, text_seq)
        vision_attended = vision_attended.squeeze(1)
        
        # Cross-attention: text attends to vision
        text_attended, _ = self.cross_attention(text_seq, vision_seq, vision_seq)
        text_attended = text_attended.squeeze(1)
        
        # Concatenate and fuse
        fused = torch.cat([vision_attended, text_attended], dim=-1)
        fused = self.fusion_layer(fused)
        
        return fused
    
    def forward(self, pixel_values, input_ids, attention_mask, return_embeddings=False):
        """Forward pass"""
        
        # Encode modalities
        vision_embeds = self.encode_vision(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        
        if return_embeddings:
            return vision_embeds, text_embeds
        
        # Normalize embeddings
        vision_embeds = vision_embeds / vision_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarity matrix
        logit_scale = self.temperature.exp()
        logits_per_image = logit_scale * vision_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text
    
    def contrastive_loss(self, logits_per_image, logits_per_text):
        """Compute contrastive loss"""
        
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=logits_per_image.device)
        
        loss_img = nn.functional.cross_entropy(logits_per_image, labels)
        loss_txt = nn.functional.cross_entropy(logits_per_text, labels)
        
        return (loss_img + loss_txt) / 2

class VisionLanguageTrainer:
    """Trainer for vision-language models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        self.scheduler = None
    
    def create_data_loader(self, image_text_pairs, batch_size=32, image_size=224):
        """Create data loader for vision-language training"""
        
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        
        class VisionLanguageDataset(Dataset):
            def __init__(self, pairs, processor, transform=None):
                self.pairs = pairs
                self.processor = processor
                self.transform = transform or transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            
            def __len__(self):
                return len(self.pairs)
            
            def __getitem__(self, idx):
                image, text = self.pairs[idx]
                
                # Process image
                if hasattr(image, 'convert'):
                    image = image.convert('RGB')
                pixel_values = self.transform(image)
                
                # Process text
                text_inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77
                )
                
                return {
                    'pixel_values': pixel_values,
                    'input_ids': text_inputs['input_ids'].squeeze(0),
                    'attention_mask': text_inputs['attention_mask'].squeeze(0)
                }
        
        # Use CLIP processor for text tokenization
        from transformers import CLIPTokenizer
        processor = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        dataset = VisionLanguageDataset(image_text_pairs, processor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            logits_per_image, logits_per_text = self.model(
                pixel_values, input_ids, attention_mask
            )
            
            # Compute loss
            loss = self.model.contrastive_loss(logits_per_image, logits_per_text)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, data_loader):
        """Evaluate model performance"""
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits_per_image, logits_per_text = self.model(
                    pixel_values, input_ids, attention_mask
                )
                
                loss = self.model.contrastive_loss(logits_per_image, logits_per_text)
                total_loss += loss.item()
                
                # Calculate accuracy (image-text matching)
                batch_size = pixel_values.shape[0]
                labels = torch.arange(batch_size, device=self.device)
                
                img_pred = logits_per_image.argmax(dim=-1)
                txt_pred = logits_per_text.argmax(dim=-1)
                
                correct_predictions += (img_pred == labels).sum().item()
                correct_predictions += (txt_pred == labels).sum().item()
                total_samples += 2 * batch_size
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy

# Example training setup
def setup_vision_language_training():
    """Setup vision-language model training"""
    
    # Create custom model
    model = CustomVisionLanguageModel()
    
    # Create trainer
    trainer = VisionLanguageTrainer(model)
    
    # Create dummy training data
    dummy_pairs = [
        (Image.new('RGB', (224, 224), color=(255, 0, 0)), "a red image"),
        (Image.new('RGB', (224, 224), color=(0, 255, 0)), "a green image"),
        (Image.new('RGB', (224, 224), color=(0, 0, 255)), "a blue image"),
    ] * 100  # Repeat for larger dataset
    
    # Create data loader
    data_loader = trainer.create_data_loader(dummy_pairs, batch_size=4)
    
    print(f"Created training setup with {len(dummy_pairs)} image-text pairs")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, trainer, data_loader

# model, trainer, data_loader = setup_vision_language_training()
```

## 2. Text-to-Image Generation

### Diffusion Models and Text-to-Image

```python
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from PIL import Image
import matplotlib.pyplot as plt

class TextToImageGenerator:
    """Advanced text-to-image generation system"""
    
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        self.model_name = model_name
        self.pipe = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
    
    def load_model(self):
        """Load text-to-image model"""
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe.to(self.device)
        
        # Enable memory efficient attention
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
    
    def generate_image(self, prompt, negative_prompt="", 
                      num_inference_steps=50, guidance_scale=7.5,
                      height=512, width=512, seed=None):
        """Generate image from text prompt"""
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        with torch.autocast(self.device):
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator
            ).images[0]
        
        return image
    
    def generate_variations(self, prompt, num_variations=4, seed_base=None):
        """Generate multiple variations of the same prompt"""
        
        variations = []
        seeds = []
        
        for i in range(num_variations):
            seed = (seed_base + i) if seed_base else None
            image = self.generate_image(prompt, seed=seed)
            variations.append(image)
            seeds.append(seed)
        
        return variations, seeds
    
    def prompt_engineering(self, base_prompt, style_modifiers=None, quality_modifiers=None):
        """Engineer prompts for better image generation"""
        
        if style_modifiers is None:
            style_modifiers = [
                "photorealistic", "digital art", "oil painting", 
                "watercolor", "sketch", "3D render"
            ]
        
        if quality_modifiers is None:
            quality_modifiers = [
                "high quality", "detailed", "masterpiece", 
                "4K", "ultra realistic", "professional photography"
            ]
        
        # Create enhanced prompts
        enhanced_prompts = []
        
        for style in style_modifiers[:3]:  # Limit to 3 styles
            for quality in quality_modifiers[:2]:  # Limit to 2 quality terms
                enhanced_prompt = f"{base_prompt}, {style}, {quality}"
                enhanced_prompts.append(enhanced_prompt)
        
        return enhanced_prompts
    
    def batch_generate(self, prompts, output_dir="generated_images"):
        """Generate images for multiple prompts"""
        
        from pathlib import Path
        Path(output_dir).mkdir(exist_ok=True)
        
        generated_images = []
        
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                image = self.generate_image(prompt)
                
                # Save image
                filename = f"image_{i:03d}.png"
                filepath = Path(output_dir) / filename
                image.save(filepath)
                
                generated_images.append({
                    'prompt': prompt,
                    'image': image,
                    'filepath': filepath
                })
                
            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                continue
        
        return generated_images
    
    def evaluate_image_quality(self, image, prompt):
        """Evaluate generated image quality"""
        
        # Simple quality metrics (in practice, would use more sophisticated methods)
        import numpy as np
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate basic statistics
        metrics = {
            'mean_brightness': np.mean(img_array),
            'std_brightness': np.std(img_array),
            'contrast': np.std(img_array) / np.mean(img_array) if np.mean(img_array) > 0 else 0,
            'color_diversity': len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)),
            'prompt_length': len(prompt.split()),
        }
        
        # Simple quality score (0-1)
        quality_score = min(1.0, (
            min(metrics['contrast'], 1.0) * 0.3 +
            min(metrics['color_diversity'] / 1000, 1.0) * 0.3 +
            min(metrics['prompt_length'] / 20, 1.0) * 0.2 +
            (1.0 - abs(metrics['mean_brightness'] - 128) / 128) * 0.2
        ))
        
        metrics['quality_score'] = quality_score
        
        return metrics

class AdvancedTextToImageFeatures:
    """Advanced features for text-to-image generation"""
    
    def __init__(self, generator):
        self.generator = generator
    
    def controlnet_generation(self, prompt, control_image, control_type="canny"):
        """Generate images with ControlNet guidance"""
        
        try:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            
            # Load ControlNet
            if control_type == "canny":
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
                )
            elif control_type == "pose":
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
                )
            else:
                raise ValueError(f"Unsupported control type: {control_type}")
            
            # Create ControlNet pipeline
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.generator.model_name,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe.to(self.generator.device)
            
            # Generate image with control
            image = pipe(
                prompt=prompt,
                image=control_image,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            return image
            
        except ImportError:
            print("ControlNet not available. Install diffusers with ControlNet support.")
            return None
    
    def inpainting_generation(self, prompt, base_image, mask_image):
        """Generate images with inpainting"""
        
        try:
            from diffusers import StableDiffusionInpaintPipeline
            
            # Load inpainting pipeline
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe.to(self.generator.device)
            
            # Generate inpainted image
            image = pipe(
                prompt=prompt,
                image=base_image,
                mask_image=mask_image,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            return image
            
        except ImportError:
            print("Inpainting pipeline not available.")
            return None
    
    def style_transfer_generation(self, content_prompt, style_prompt):
        """Generate images with style transfer"""
        
        # Combine prompts for style transfer effect
        combined_prompt = f"{content_prompt} in the style of {style_prompt}"
        
        # Use higher guidance scale for stronger style adherence
        image = self.generator.generate_image(
            combined_prompt,
            guidance_scale=12.0,
            num_inference_steps=75
        )
        
        return image
    
    def progressive_generation(self, initial_prompt, refinement_steps):
        """Generate image through progressive refinement"""
        
        current_prompt = initial_prompt
        images = []
        
        for i, refinement in enumerate(refinement_steps):
            print(f"Refinement step {i+1}: {refinement}")
            
            # Add refinement to prompt
            current_prompt = f"{current_prompt}, {refinement}"
            
            # Generate with increasing quality
            image = self.generator.generate_image(
                current_prompt,
                num_inference_steps=50 + i * 10,
                guidance_scale=7.5 + i * 0.5
            )
            
            images.append({
                'step': i + 1,
                'prompt': current_prompt,
                'image': image
            })
        
        return images

# Example usage and demonstrations
def text_to_image_demo():
    """Demonstrate text-to-image generation capabilities"""
    
    # Initialize generator
    generator = TextToImageGenerator()
    
    # Basic generation
    basic_prompt = "a serene mountain landscape at sunset, digital art"
    print(f"Generating image for: {basic_prompt}")
    
    try:
        image = generator.generate_image(basic_prompt, seed=42)
        print("✓ Basic generation successful")
    except Exception as e:
        print(f"✗ Basic generation failed: {e}")
        # Create dummy image for demo
        image = Image.new('RGB', (512, 512), color=(100, 150, 200))
    
    # Prompt engineering
    enhanced_prompts = generator.prompt_engineering(
        "a magical forest",
        style_modifiers=["fantasy art", "digital painting"],
        quality_modifiers=["detailed", "atmospheric"]
    )
    
    print(f"\nGenerated {len(enhanced_prompts)} enhanced prompts:")
    for i, prompt in enumerate(enhanced_prompts[:3]):
        print(f"  {i+1}. {prompt}")
    
    # Generate variations
    variations, seeds = generator.generate_variations(
        "a futuristic cityscape", num_variations=3, seed_base=100
    )
    
    print(f"\nGenerated {len(variations)} variations with seeds: {seeds}")
    
    # Evaluate quality
    metrics = generator.evaluate_image_quality(image, basic_prompt)
    print(f"\nImage quality metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    return generator, image, variations

# generator, sample_image, variations = text_to_image_demo()
```

## 3. Audio-Text Transformers

### Speech Recognition and Generation

```python
import torch
import torchaudio
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    WhisperProcessor, WhisperForConditionalGeneration,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
)
import librosa
import numpy as np

class AudioTextProcessor:
    """Comprehensive audio-text processing system"""
    
    def __init__(self):
        self.asr_model = None
        self.asr_processor = None
        self.tts_model = None
        self.tts_processor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_asr_model(self, model_name="openai/whisper-base"):
        """Load automatic speech recognition model"""
        
        if "whisper" in model_name:
            self.asr_processor = WhisperProcessor.from_pretrained(model_name)
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        else:
            self.asr_processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.asr_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        self.asr_model.to(self.device)
        self.asr_model.eval()
    
    def load_tts_model(self, model_name="microsoft/speecht5_tts"):
        """Load text-to-speech model"""
        
        self.tts_processor = SpeechT5Processor.from_pretrained(model_name)
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        
        # Load vocoder
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        self.tts_model.to(self.device)
        self.vocoder.to(self.device)
        self.tts_model.eval()
        self.vocoder.eval()
    
    def preprocess_audio(self, audio_path, target_sr=16000):
        """Preprocess audio file for model input"""
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio, sr
    
    def speech_to_text(self, audio_path, language="en"):
        """Convert speech to text"""
        
        if self.asr_model is None:
            self.load_asr_model()
        
        # Preprocess audio
        audio, sr = self.preprocess_audio(audio_path)
        
        # Process with model
        inputs = self.asr_processor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            if "whisper" in self.asr_model.config.name_or_path:
                # Whisper model
                generated_ids = self.asr_model.generate(inputs["input_features"])
                transcription = self.asr_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
            else:
                # Wav2Vec2 model
                logits = self.asr_model(inputs.input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.asr_processor.batch_decode(predicted_ids)[0]
        
        return transcription
    
    def text_to_speech(self, text, speaker_embeddings=None):
        """Convert text to speech"""
        
        if self.tts_model is None:
            self.load_tts_model()
        
        # Process text
        inputs = self.tts_processor(text=text, return_tensors="pt").to(self.device)
        
        # Load default speaker embeddings if not provided
        if speaker_embeddings is None:
            # Create default speaker embeddings
            speaker_embeddings = torch.zeros(1, 512).to(self.device)
        
        with torch.no_grad():
            # Generate speech
            speech = self.tts_model.generate_speech(
                inputs["input_ids"], 
                speaker_embeddings, 
                vocoder=self.vocoder
            )
        
        return speech.cpu().numpy()
    
    def batch_transcribe(self, audio_files):
        """Transcribe multiple audio files"""
        
        transcriptions = []
        
        for audio_file in audio_files:
            try:
                transcription = self.speech_to_text(audio_file)
                transcriptions.append({
                    'file': audio_file,
                    'transcription': transcription,
                    'success': True
                })
            except Exception as e:
                transcriptions.append({
                    'file': audio_file,
                    'transcription': '',
                    'success': False,
                    'error': str(e)
                })
        
        return transcriptions
    
    def audio_text_alignment(self, audio_path, text):
        """Align audio with text for forced alignment"""
        
        # Load audio
        audio, sr = self.preprocess_audio(audio_path)
        
        # Get ASR model predictions
        inputs = self.asr_processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.asr_model, 'generate'):
                # For generation-based models
                outputs = self.asr_model.generate(
                    inputs["input_features"], 
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Simple alignment (frame-level)
                alignment = self.compute_alignment(outputs, text)
            else:
                # For CTC-based models
                logits = self.asr_model(inputs.input_values).logits
                alignment = self.compute_ctc_alignment(logits, text)
        
        return alignment
    
    def compute_alignment(self, outputs, target_text):
        """Compute alignment between audio and text"""
        
        # Simplified alignment computation
        # In practice, would use more sophisticated methods like CTC alignment
        
        generated_ids = outputs.sequences[0]
        decoded_text = self.asr_processor.decode(generated_ids, skip_special_tokens=True)
        
        # Create basic word-level alignment
        words = target_text.split()
        alignment = []
        
        for i, word in enumerate(words):
            alignment.append({
                'word': word,
                'start_time': i * 0.5,  # Simplified timing
                'end_time': (i + 1) * 0.5,
                'confidence': 0.8  # Placeholder confidence
            })
        
        return alignment
    
    def compute_ctc_alignment(self, logits, target_text):
        """Compute CTC-based alignment"""
        
        # Get predicted tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode predictions
        predicted_text = self.asr_processor.decode(predicted_ids[0])
        
        # Simple word alignment
        words = target_text.split()
        frame_duration = len(logits[0]) / len(words) if words else 1
        
        alignment = []
        for i, word in enumerate(words):
            start_frame = int(i * frame_duration)
            end_frame = int((i + 1) * frame_duration)
            
            alignment.append({
                'word': word,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_frame * 0.02,  # Assuming 20ms frames
                'end_time': end_frame * 0.02
            })
        
        return alignment

class AdvancedAudioTextFeatures:
    """Advanced audio-text processing features"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def multilingual_asr(self, audio_path, languages=None):
        """Multilingual automatic speech recognition"""
        
        if languages is None:
            languages = ["en", "es", "fr", "de", "it"]
        
        results = {}
        
        # Load multilingual Whisper model
        self.processor.load_asr_model("openai/whisper-base")
        
        for lang in languages:
            try:
                # Force language for Whisper
                audio, sr = self.processor.preprocess_audio(audio_path)
                inputs = self.processor.asr_processor(
                    audio, sampling_rate=sr, return_tensors="pt"
                ).to(self.processor.device)
                
                # Generate with language specification
                forced_decoder_ids = self.processor.asr_processor.get_decoder_prompt_ids(
                    language=lang, task="transcribe"
                )
                
                generated_ids = self.processor.asr_model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_decoder_ids
                )
                
                transcription = self.processor.asr_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                
                results[lang] = transcription
                
            except Exception as e:
                results[lang] = f"Error: {e}"
        
        return results
    
    def voice_activity_detection(self, audio_path, frame_duration=0.02):
        """Detect voice activity in audio"""
        
        # Load audio
        audio, sr = self.processor.preprocess_audio(audio_path)
        
        # Simple VAD using energy-based detection
        frame_length = int(frame_duration * sr)
        hop_length = frame_length // 2
        
        # Compute short-time energy
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
        
        # Threshold-based VAD
        energy = np.array(energy)
        threshold = np.mean(energy) * 0.1  # Adjust threshold as needed
        
        voice_activity = energy > threshold
        
        # Create segments
        segments = []
        in_speech = False
        start_time = 0
        
        for i, is_voice in enumerate(voice_activity):
            time = i * frame_duration
            
            if is_voice and not in_speech:
                start_time = time
                in_speech = True
            elif not is_voice and in_speech:
                segments.append({
                    'start': start_time,
                    'end': time,
                    'duration': time - start_time
                })
                in_speech = False
        
        return segments
    
    def speaker_diarization(self, audio_path, num_speakers=None):
        """Perform speaker diarization"""
        
        # Simplified speaker diarization
        # In practice, would use specialized models like pyannote.audio
        
        # Get voice activity segments
        segments = self.voice_activity_detection(audio_path)
        
        # Assign speakers (simplified clustering)
        if num_speakers is None:
            num_speakers = min(len(segments) // 3, 4)  # Estimate speakers
        
        # Simple speaker assignment based on temporal clustering
        speaker_segments = []
        
        for i, segment in enumerate(segments):
            speaker_id = i % num_speakers  # Simplified assignment
            speaker_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'speaker': f"Speaker_{speaker_id}",
                'duration': segment['duration']
            })
        
        return speaker_segments
    
    def audio_emotion_recognition(self, audio_path):
        """Recognize emotions in audio"""
        
        # Load audio features
        audio, sr = self.processor.preprocess_audio(audio_path)
        
        # Extract acoustic features
        features = self.extract_acoustic_features(audio, sr)
        
        # Simple emotion classification (placeholder)
        emotions = ["neutral", "happy", "sad", "angry", "fear", "surprise"]
        
        # Random classification for demo (would use trained model)
        emotion_scores = np.random.dirichlet(np.ones(len(emotions)))
        
        emotion_results = dict(zip(emotions, emotion_scores))
        predicted_emotion = max(emotion_results, key=emotion_results.get)
        
        return {
            'predicted_emotion': predicted_emotion,
            'emotion_scores': emotion_results,
            'features': features
        }
    
    def extract_acoustic_features(self, audio, sr):
        """Extract acoustic features from audio"""
        
        features = {}
        
        # Spectral features
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=magnitude))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=magnitude))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
        features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(S=magnitude)
        features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo
        
        return features

# Example usage and demonstration
def audio_text_demo():
    """Demonstrate audio-text processing capabilities"""
    
    # Initialize processor
    processor = AudioTextProcessor()
    
    # Initialize advanced features
    advanced_features = AdvancedAudioTextFeatures(processor)
    
    print("Audio-Text Processing Demo")
    print("=" * 50)
    
    # Create dummy audio for demo (in practice, would use real audio files)
    dummy_audio_path = "demo_audio.wav"
    
    # Generate dummy audio
    sr = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sr * duration), False)
    # Simple sine wave as dummy audio
    dummy_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Save dummy audio
    import scipy.io.wavfile as wavfile
    wavfile.write(dummy_audio_path, sr, (dummy_audio * 32767).astype(np.int16))
    
    try:
        # Voice Activity Detection
        print("\n1. Voice Activity Detection:")
        vad_segments = advanced_features.voice_activity_detection(dummy_audio_path)
        print(f"Found {len(vad_segments)} voice segments")
        for i, segment in enumerate(vad_segments[:3]):
            print(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
        
        # Speaker Diarization
        print("\n2. Speaker Diarization:")
        speaker_segments = advanced_features.speaker_diarization(dummy_audio_path, num_speakers=2)
        print(f"Identified {len(set(s['speaker'] for s in speaker_segments))} speakers")
        for segment in speaker_segments[:3]:
            print(f"  {segment['speaker']}: {segment['start']:.2f}s - {segment['end']:.2f}s")
        
        # Emotion Recognition
        print("\n3. Emotion Recognition:")
        emotion_results = advanced_features.audio_emotion_recognition(dummy_audio_path)
        print(f"Predicted emotion: {emotion_results['predicted_emotion']}")
        print("Emotion scores:")
        for emotion, score in emotion_results['emotion_scores'].items():
            print(f"  {emotion}: {score:.3f}")
        
        # Acoustic Features
        print(f"\n4. Acoustic Features:")
        features = emotion_results['features']
        print(f"  Spectral centroid: {features['spectral_centroid']:.2f}")
        print(f"  Zero crossing rate: {features['zero_crossing_rate']:.4f}")
        print(f"  Tempo: {features['tempo']:.1f} BPM")
        
    except Exception as e:
        print(f"Demo error: {e}")
    
    finally:
        # Clean up dummy file
        import os
        if os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)
    
    return processor, advanced_features

# processor, advanced_features = audio_text_demo()
```

## 4. Learning Objectives

By the end of this section, you should be able to:
- **Understand** multimodal transformer architectures and cross-modal attention mechanisms
- **Implement** vision-language models for image-text tasks
- **Deploy** text-to-image generation systems with advanced features
- **Build** audio-text processing pipelines for speech recognition and synthesis
- **Apply** multimodal models to real-world applications
- **Evaluate** multimodal model performance across different modalities

### Self-Assessment Checklist

□ Can explain cross-modal attention and multimodal fusion techniques  
□ Can implement CLIP-style vision-language models from scratch  
□ Can use diffusion models for text-to-image generation  
□ Can build speech recognition and text-to-speech systems  
□ Can apply ControlNet and other advanced generation techniques  
□ Can evaluate multimodal model performance  
□ Can design end-to-end multimodal applications  

## 5. Practical Exercises

**Exercise 1: Custom Vision-Language Model**
```python
# TODO: Build a custom vision-language model
# Implement cross-modal attention and contrastive learning
# Train on image-caption pairs and evaluate on retrieval tasks
```

**Exercise 2: Text-to-Image Application**
```python
# TODO: Create a complete text-to-image application
# Include prompt engineering, style transfer, and quality evaluation
# Implement batch generation and user interface
```

**Exercise 3: Multimodal Chat System**
```python
# TODO: Build a multimodal chat system
# Combine text, image, and audio processing
# Enable users to interact using multiple modalities
```

## 6. Study Materials

### Essential Papers
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- [DALL·E 2: Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)

### Multimodal Datasets
- **Vision-Language**: COCO Captions, Flickr30k, Conceptual Captions, Visual Genome
- **Text-to-Image**: LAION-5B, CC12M, YFCC100M
- **Audio-Text**: LibriSpeech, Common Voice, AudioCaps, Clotho

### Tools and Libraries
```bash
pip install transformers diffusers accelerate
pip install torch torchvision torchaudio
pip install opencv-python pillow matplotlib
pip install librosa scipy soundfile
pip install datasets huggingface_hub
```
