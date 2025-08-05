# Project: Content Generation System

*Duration: 4 weeks*

## Objective

Build a comprehensive content generation system that assists content creators by providing AI-powered writing assistance with controllable style, tone, and quality evaluation. This project integrates multiple NLP techniques including text generation, style transfer, content evaluation, and user interface design.

## System Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │    │   Style Control  │    │  Quality Check  │
│   - Topic       │───▶│   - Tone         │───▶│  - Coherence    │
│   - Keywords    │    │   - Style        │    │  - Relevance    │
│   - Length      │    │   - Formality    │    │  - Grammar      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Content Gen    │    │   Post-Process   │    │   Final Output  │
│  - GPT/BERT     │───▶│   - Filter       │───▶│  - Formatted    │
│  - Fine-tuned   │    │   - Refine       │    │  - Evaluated    │
│  - Multi-modal  │    │   - Enhance      │    │  - Optimized    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 1. Core Content Generation Engine

### Multi-Model Generation System

```python
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    GPT2LMHeadModel, BertTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    BloomForCausalLM, BloomTokenizerFast
)
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import re
import json
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

class ContentType(Enum):
    """Supported content types"""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_DOC = "technical_doc"
    MARKETING_COPY = "marketing_copy"
    NEWS_REPORT = "news_report"

class StyleType(Enum):
    """Writing styles"""
    FORMAL = "formal"
    CASUAL = "casual"
    ACADEMIC = "academic"
    CREATIVE = "creative"
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    PERSUASIVE = "persuasive"
    INFORMATIVE = "informative"

class ToneType(Enum):
    """Writing tones"""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    ENTHUSIASTIC = "enthusiastic"
    EMPATHETIC = "empathetic"
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    OPTIMISTIC = "optimistic"

@dataclass
class GenerationConfig:
    """Configuration for content generation"""
    content_type: ContentType
    style: StyleType
    tone: ToneType
    target_length: int = 500
    keywords: List[str] = None
    topics: List[str] = None
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512
    min_length: int = 50
    
class ContentGenerationEngine:
    """Advanced content generation with style and tone control"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.models = {}
        self.tokenizers = {}
        
        # Style and tone prompts
        self.style_prompts = {
            StyleType.FORMAL: "Write in a formal, professional manner with proper grammar and structure.",
            StyleType.CASUAL: "Write in a casual, relaxed tone as if talking to a friend.",
            StyleType.ACADEMIC: "Write in an academic style with scholarly language and citations.",
            StyleType.CREATIVE: "Write creatively with vivid descriptions and engaging narrative.",
            StyleType.PROFESSIONAL: "Write professionally for business communication.",
            StyleType.CONVERSATIONAL: "Write in a conversational style as if speaking directly to the reader.",
            StyleType.PERSUASIVE: "Write persuasively to convince and influence the reader.",
            StyleType.INFORMATIVE: "Write informatively to educate and explain clearly."
        }
        
        self.tone_prompts = {
            ToneType.NEUTRAL: "Maintain a neutral, balanced tone throughout.",
            ToneType.FRIENDLY: "Use a warm, friendly, and approachable tone.",
            ToneType.AUTHORITATIVE: "Write with authority and confidence on the subject.",
            ToneType.ENTHUSIASTIC: "Express enthusiasm and excitement about the topic.",
            ToneType.EMPATHETIC: "Show understanding and empathy toward the audience.",
            ToneType.HUMOROUS: "Include appropriate humor and light-hearted elements.",
            ToneType.SERIOUS: "Maintain a serious, thoughtful tone throughout.",
            ToneType.OPTIMISTIC: "Focus on positive aspects and hopeful outcomes."
        }
        
        # Content type templates
        self.content_templates = {
            ContentType.ARTICLE: {
                "structure": ["introduction", "main_points", "conclusion"],
                "style_guide": "Follow journalistic principles with clear structure and facts."
            },
            ContentType.BLOG_POST: {
                "structure": ["hook", "main_content", "call_to_action"],
                "style_guide": "Engage readers with personal insights and actionable advice."
            },
            ContentType.SOCIAL_MEDIA: {
                "structure": ["hook", "main_message", "hashtags"],
                "style_guide": "Be concise, engaging, and include relevant hashtags."
            },
            ContentType.EMAIL: {
                "structure": ["subject", "greeting", "body", "closing"],
                "style_guide": "Be clear, concise, and action-oriented."
            },
            ContentType.CREATIVE_WRITING: {
                "structure": ["setting", "characters", "plot", "resolution"],
                "style_guide": "Focus on narrative, character development, and vivid descriptions."
            },
            ContentType.TECHNICAL_DOC: {
                "structure": ["overview", "details", "examples", "summary"],
                "style_guide": "Be precise, clear, and include technical details and examples."
            },
            ContentType.MARKETING_COPY: {
                "structure": ["attention", "interest", "desire", "action"],
                "style_guide": "Focus on benefits, create urgency, and include clear calls-to-action."
            },
            ContentType.NEWS_REPORT: {
                "structure": ["headline", "lead", "body", "conclusion"],
                "style_guide": "Follow the inverted pyramid structure with facts first."
            }
        }
        
        # Generation statistics
        self.generation_stats = defaultdict(int)
        
    async def load_models(self):
        """Load required models for content generation"""
        
        print("Loading content generation models...")
        
        # Primary generation model (GPT-2 for demo, could be GPT-3.5/4, LLaMA, etc.)
        model_configs = [
            ("gpt2-medium", "primary"),
            ("google/flan-t5-base", "conditioning"),
        ]
        
        for model_name, model_type in model_configs:
            try:
                print(f"Loading {model_name}...")
                
                if "t5" in model_name.lower():
                    tokenizer = T5Tokenizer.from_pretrained(model_name)
                    model = T5ForConditionalGeneration.from_pretrained(model_name)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add padding token if missing
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model.to(self.device)
                model.eval()
                
                self.models[model_type] = model
                self.tokenizers[model_type] = tokenizer
                
                print(f"✓ {model_name} loaded successfully")
                
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
    
    def create_style_conditioning_prompt(self, config: GenerationConfig, 
                                       topic: str) -> str:
        """Create a conditioning prompt for style and tone control"""
        
        # Base prompt structure
        prompt_parts = []
        
        # Add content type guidance
        content_template = self.content_templates[config.content_type]
        prompt_parts.append(f"Task: Create a {config.content_type.value}")
        prompt_parts.append(f"Topic: {topic}")
        
        # Add style guidance
        style_instruction = self.style_prompts[config.style]
        prompt_parts.append(f"Style: {style_instruction}")
        
        # Add tone guidance
        tone_instruction = self.tone_prompts[config.tone]
        prompt_parts.append(f"Tone: {tone_instruction}")
        
        # Add structure guidance
        structure = content_template["structure"]
        prompt_parts.append(f"Structure: {', '.join(structure)}")
        
        # Add keywords if provided
        if config.keywords:
            prompt_parts.append(f"Keywords to include: {', '.join(config.keywords)}")
        
        # Add length guidance
        prompt_parts.append(f"Target length: approximately {config.target_length} words")
        
        # Content type specific guidance
        prompt_parts.append(f"Guidelines: {content_template['style_guide']}")
        
        # Combine into final prompt
        conditioning_prompt = "\n".join(prompt_parts)
        conditioning_prompt += f"\n\nContent:\n"
        
        return conditioning_prompt
    
    async def generate_with_style_control(self, config: GenerationConfig, 
                                        topic: str) -> Dict[str, Any]:
        """Generate content with style and tone control"""
        
        start_time = time.time()
        
        # Create conditioning prompt
        conditioning_prompt = self.create_style_conditioning_prompt(config, topic)
        
        # Primary generation
        primary_model = self.models["primary"]
        primary_tokenizer = self.tokenizers["primary"]
        
        # Tokenize input
        inputs = primary_tokenizer(
            conditioning_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "do_sample": True,
            "pad_token_id": primary_tokenizer.eos_token_id,
            "use_cache": True
        }
        
        # Generate content
        with torch.no_grad():
            outputs = primary_model.generate(**inputs, **generation_kwargs)
        
        # Decode generated content
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = primary_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up generated text
        cleaned_text = self.post_process_generated_text(generated_text, config)
        
        generation_time = time.time() - start_time
        
        # Update statistics
        self.generation_stats[f"{config.content_type.value}_{config.style.value}"] += 1
        
        result = {
            "generated_text": cleaned_text,
            "conditioning_prompt": conditioning_prompt,
            "config": config,
            "generation_time": generation_time,
            "word_count": len(cleaned_text.split()),
            "metadata": {
                "model_used": "primary",
                "tokens_generated": len(generated_ids),
                "effective_length": len(cleaned_text.split())
            }
        }
        
        return result
    
    def post_process_generated_text(self, text: str, config: GenerationConfig) -> str:
        """Post-process generated text for quality and style"""
        
        # Remove extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        # Content type specific post-processing
        if config.content_type == ContentType.SOCIAL_MEDIA:
            # Ensure it's concise
            sentences = text.split('.')
            if len(sentences) > 3:
                text = '. '.join(sentences[:3]) + '.'
            
            # Add hashtags if missing and keywords provided
            if config.keywords and '#' not in text:
                hashtags = [f"#{kw.replace(' ', '')}" for kw in config.keywords[:3]]
                text += f"\n\n{' '.join(hashtags)}"
        
        elif config.content_type == ContentType.EMAIL:
            # Ensure proper email structure
            if not text.startswith(('Dear', 'Hello', 'Hi')):
                text = "Dear Reader,\n\n" + text
            
            if not text.endswith(('Sincerely,', 'Best regards,', 'Thank you,')):
                text += "\n\nBest regards,"
        
        elif config.content_type == ContentType.ARTICLE:
            # Ensure proper article structure
            paragraphs = text.split('\n\n')
            if len(paragraphs) < 3:
                # Add more structure if needed
                text = self.enhance_article_structure(text, config)
        
        return text
    
    def enhance_article_structure(self, text: str, config: GenerationConfig) -> str:
        """Enhance article structure for better readability"""
        
        paragraphs = text.split('\n\n')
        
        # Ensure introduction
        if len(paragraphs) >= 1:
            intro = paragraphs[0]
            if len(intro.split()) < 30:  # Short intro, enhance it
                intro += " This article explores the key aspects and implications of this important topic."
                paragraphs[0] = intro
        
        # Ensure conclusion if article is long enough
        if len(paragraphs) >= 3:
            conclusion = paragraphs[-1]
            if not any(word in conclusion.lower() for word in ['conclusion', 'summary', 'finally', 'in summary']):
                conclusion = "In conclusion, " + conclusion
                paragraphs[-1] = conclusion
        
        return '\n\n'.join(paragraphs)
    
    async def generate_multiple_variants(self, config: GenerationConfig, 
                                      topic: str, num_variants: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple variants of content for comparison"""
        
        variants = []
        
        for i in range(num_variants):
            # Slightly vary the temperature for diversity
            variant_config = config
            variant_config.temperature = config.temperature + (i * 0.1)
            
            variant_result = await self.generate_with_style_control(variant_config, topic)
            variant_result["variant_id"] = i + 1
            variants.append(variant_result)
        
        return variants
    
    def get_content_suggestions(self, topic: str, content_type: ContentType) -> Dict[str, List[str]]:
        """Get content suggestions based on topic and type"""
        
        suggestions = {
            "keywords": [],
            "subtopics": [],
            "angles": [],
            "structure_tips": []
        }
        
        # Basic keyword extraction (in production, use more sophisticated NLP)
        topic_words = topic.lower().split()
        
        # Content type specific suggestions
        if content_type == ContentType.ARTICLE:
            suggestions["structure_tips"] = [
                "Start with a compelling headline",
                "Include an engaging introduction",
                "Use subheadings to break up content",
                "End with a strong conclusion"
            ]
            suggestions["angles"] = [
                f"The complete guide to {topic}",
                f"Top trends in {topic}",
                f"How {topic} is changing",
                f"The future of {topic}"
            ]
        
        elif content_type == ContentType.BLOG_POST:
            suggestions["structure_tips"] = [
                "Start with a personal hook",
                "Share actionable insights",
                "Include personal experiences",
                "End with a call-to-action"
            ]
            suggestions["angles"] = [
                f"My experience with {topic}",
                f"5 lessons about {topic}",
                f"Why {topic} matters",
                f"Getting started with {topic}"
            ]
        
        elif content_type == ContentType.SOCIAL_MEDIA:
            suggestions["structure_tips"] = [
                "Start with an attention-grabbing hook",
                "Keep it concise and engaging",
                "Include relevant hashtags",
                "Add a call-to-action"
            ]
            suggestions["angles"] = [
                f"Quick tip about {topic}",
                f"Did you know about {topic}?",
                f"The truth about {topic}",
                f"Why I love {topic}"
            ]
        
        # Generate basic keywords (simplified)
        suggestions["keywords"] = [topic] + topic_words
        
        return suggestions

class ContentQualityEvaluator:
    """Evaluate the quality of generated content"""
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_content(self, content: str, config: GenerationConfig) -> Dict[str, float]:
        """Comprehensive content quality evaluation"""
        
        evaluation_results = {}
        
        # 1. Basic metrics
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        evaluation_results.update({
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": word_count / max(sentence_count, 1),
        })
        
        # 2. Readability metrics
        readability_score = self.calculate_readability(content)
        evaluation_results["readability_score"] = readability_score
        
        # 3. Style consistency
        style_score = self.evaluate_style_consistency(content, config.style)
        evaluation_results["style_consistency"] = style_score
        
        # 4. Tone analysis
        tone_score = self.evaluate_tone_consistency(content, config.tone)
        evaluation_results["tone_consistency"] = tone_score
        
        # 5. Content structure
        structure_score = self.evaluate_structure(content, config.content_type)
        evaluation_results["structure_score"] = structure_score
        
        # 6. Keyword usage
        if config.keywords:
            keyword_score = self.evaluate_keyword_usage(content, config.keywords)
            evaluation_results["keyword_usage"] = keyword_score
        
        # 7. Grammar and fluency (simplified)
        grammar_score = self.evaluate_grammar(content)
        evaluation_results["grammar_score"] = grammar_score
        
        # 8. Overall quality score
        quality_scores = [
            readability_score, style_score, tone_score, 
            structure_score, grammar_score
        ]
        if config.keywords:
            quality_scores.append(evaluation_results["keyword_usage"])
        
        evaluation_results["overall_quality"] = np.mean(quality_scores)
        
        return evaluation_results
    
    def calculate_readability(self, content: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        
        words = content.split()
        sentences = [s for s in content.split('.') if s.strip()]
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simplified syllable counting
        syllable_count = sum(self.count_syllables(word) for word in words)
        avg_syllables_per_word = syllable_count / len(words)
        
        # Simplified Flesch Reading Ease formula
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        return max(0, min(100, readability)) / 100
    
    def count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def evaluate_style_consistency(self, content: str, style: StyleType) -> float:
        """Evaluate style consistency (simplified)"""
        
        content_lower = content.lower()
        
        # Style-specific indicators
        style_indicators = {
            StyleType.FORMAL: {
                'positive': ['therefore', 'however', 'furthermore', 'consequently'],
                'negative': ['gonna', 'wanna', 'yeah', 'ok']
            },
            StyleType.CASUAL: {
                'positive': ['really', 'pretty', 'quite', 'actually'],
                'negative': ['subsequently', 'furthermore', 'nonetheless']
            },
            StyleType.ACADEMIC: {
                'positive': ['research', 'study', 'analysis', 'evidence'],
                'negative': ['i think', 'in my opinion', 'obviously']
            },
            StyleType.CREATIVE: {
                'positive': ['vivid', 'imagine', 'beautiful', 'stunning'],
                'negative': ['data shows', 'statistics', 'metrics']
            }
        }
        
        if style not in style_indicators:
            return 0.5  # Neutral score for undefined styles
        
        indicators = style_indicators[style]
        positive_count = sum(1 for word in indicators['positive'] if word in content_lower)
        negative_count = sum(1 for word in indicators['negative'] if word in content_lower)
        
        # Calculate style score
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return 0.5
        
        return positive_count / total_indicators
    
    def evaluate_tone_consistency(self, content: str, tone: ToneType) -> float:
        """Evaluate tone consistency (simplified)"""
        
        content_lower = content.lower()
        
        # Tone-specific indicators
        tone_indicators = {
            ToneType.FRIENDLY: {
                'positive': ['welcome', 'happy', 'glad', 'enjoy', 'please'],
                'negative': ['must', 'required', 'mandatory', 'strictly']
            },
            ToneType.AUTHORITATIVE: {
                'positive': ['clearly', 'definitely', 'certainly', 'proven'],
                'negative': ['maybe', 'perhaps', 'possibly', 'might']
            },
            ToneType.ENTHUSIASTIC: {
                'positive': ['amazing', 'fantastic', 'excited', 'wonderful'],
                'negative': ['boring', 'mundane', 'ordinary', 'dull']
            }
        }
        
        if tone not in tone_indicators:
            return 0.5
        
        indicators = tone_indicators[tone]
        positive_count = sum(1 for word in indicators['positive'] if word in content_lower)
        negative_count = sum(1 for word in indicators['negative'] if word in content_lower)
        
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return 0.5
        
        return positive_count / total_indicators
    
    def evaluate_structure(self, content: str, content_type: ContentType) -> float:
        """Evaluate content structure"""
        
        paragraphs = content.split('\n\n')
        paragraph_count = len(paragraphs)
        
        # Content type specific structure evaluation
        if content_type == ContentType.ARTICLE:
            # Articles should have at least 3 paragraphs
            if paragraph_count >= 3:
                return 1.0
            elif paragraph_count == 2:
                return 0.7
            else:
                return 0.3
        
        elif content_type == ContentType.SOCIAL_MEDIA:
            # Social media should be concise
            if paragraph_count <= 2 and len(content.split()) <= 100:
                return 1.0
            else:
                return 0.5
        
        elif content_type == ContentType.EMAIL:
            # Check for email elements
            has_greeting = any(greeting in content.lower() for greeting in ['dear', 'hello', 'hi'])
            has_closing = any(closing in content.lower() for closing in ['sincerely', 'regards', 'thank you'])
            
            if has_greeting and has_closing:
                return 1.0
            elif has_greeting or has_closing:
                return 0.7
            else:
                return 0.3
        
        # Default structure evaluation
        return min(1.0, paragraph_count / 3)
    
    def evaluate_keyword_usage(self, content: str, keywords: List[str]) -> float:
        """Evaluate keyword usage"""
        
        content_lower = content.lower()
        keywords_used = sum(1 for keyword in keywords if keyword.lower() in content_lower)
        
        return keywords_used / len(keywords) if keywords else 1.0
    
    def evaluate_grammar(self, content: str) -> float:
        """Simple grammar evaluation (placeholder for more sophisticated analysis)"""
        
        # Basic grammar checks
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        grammar_score = 1.0
        
        for sentence in sentences:
            # Check capitalization
            if sentence and not sentence[0].isupper():
                grammar_score -= 0.1
            
            # Check for very short sentences
            if len(sentence.split()) < 3:
                grammar_score -= 0.05
            
            # Check for very long sentences
            if len(sentence.split()) > 40:
                grammar_score -= 0.1
        
        return max(0, grammar_score)

# Example usage and demo
async def content_generation_demo():
    """Demonstrate the content generation system"""
    
    print("Content Generation System Demo")
    print("=" * 50)
    
    # Initialize system
    engine = ContentGenerationEngine()
    evaluator = ContentQualityEvaluator()
    
    # Load models
    await engine.load_models()
    
    # Test different content types and styles
    test_configs = [
        GenerationConfig(
            content_type=ContentType.BLOG_POST,
            style=StyleType.CASUAL,
            tone=ToneType.FRIENDLY,
            target_length=300,
            keywords=["artificial intelligence", "future", "technology"]
        ),
        GenerationConfig(
            content_type=ContentType.ARTICLE,
            style=StyleType.PROFESSIONAL,
            tone=ToneType.AUTHORITATIVE,
            target_length=500,
            keywords=["climate change", "environment", "sustainability"]
        ),
        GenerationConfig(
            content_type=ContentType.SOCIAL_MEDIA,
            style=StyleType.CASUAL,
            tone=ToneType.ENTHUSIASTIC,
            target_length=100,
            keywords=["innovation", "startup"]
        )
    ]
    
    topics = [
        "The Future of Artificial Intelligence",
        "Climate Change Solutions",
        "Starting a Tech Company"
    ]
    
    results = []
    
    for i, (config, topic) in enumerate(zip(test_configs, topics)):
        print(f"\n{i+1}. Generating {config.content_type.value} - {config.style.value} style")
        print(f"   Topic: {topic}")
        
        # Generate content
        result = await engine.generate_with_style_control(config, topic)
        
        # Evaluate quality
        evaluation = evaluator.evaluate_content(result["generated_text"], config)
        result["evaluation"] = evaluation
        
        results.append(result)
        
        # Display results
        print(f"   Generated ({result['word_count']} words):")
        print(f"   {result['generated_text'][:200]}...")
        print(f"   Quality Score: {evaluation['overall_quality']:.2f}")
        print(f"   Generation Time: {result['generation_time']:.2f}s")
        print(f"   Readability: {evaluation['readability_score']:.2f}")
        print(f"   Style Consistency: {evaluation['style_consistency']:.2f}")
    
    # Generate content suggestions
    print(f"\n4. Content Suggestions for 'Machine Learning':")
    suggestions = engine.get_content_suggestions("Machine Learning", ContentType.ARTICLE)
    
    for category, items in suggestions.items():
        print(f"   {category.title()}: {items[:3]}")
    
    # Generate multiple variants
    print(f"\n5. Multiple Variants Generation:")
    variants = await engine.generate_multiple_variants(
        test_configs[0], 
        "The Impact of Remote Work", 
        num_variants=2
    )
    
    for variant in variants:
        print(f"   Variant {variant['variant_id']}:")
        print(f"   {variant['generated_text'][:150]}...")
        print(f"   Quality: {variant.get('evaluation', {}).get('overall_quality', 'N/A')}")
    
    return engine, evaluator, results

# Run the demo
# engine, evaluator, results = await content_generation_demo()
```

## 2. User Interface and Experience

### Streamlit Web Application

```python
import streamlit as st
import asyncio
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import json

class ContentGenerationUI:
    """Streamlit UI for content generation system"""
    
    def __init__(self):
        self.engine = None
        self.evaluator = None
        
        # Session state initialization
        if 'generated_content' not in st.session_state:
            st.session_state.generated_content = []
        if 'generation_history' not in st.session_state:
            st.session_state.generation_history = []
    
    async def initialize_system(self):
        """Initialize the content generation system"""
        if self.engine is None:
            with st.spinner("Initializing AI models..."):
                self.engine = ContentGenerationEngine()
                self.evaluator = ContentQualityEvaluator()
                await self.engine.load_models()
            st.success("✅ AI models loaded successfully!")
        
        return self.engine, self.evaluator
    
    def render_sidebar_controls(self):
        """Render sidebar controls for content configuration"""
        
        st.sidebar.title("Content Configuration")
        
        # Content type selection
        content_type = st.sidebar.selectbox(
            "Content Type",
            options=[ct.value for ct in ContentType],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Style selection
        style = st.sidebar.selectbox(
            "Writing Style",
            options=[st.value for st in StyleType],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Tone selection
        tone = st.sidebar.selectbox(
            "Tone",
            options=[tt.value for tt in ToneType],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Advanced settings
        st.sidebar.subheader("Advanced Settings")
        
        target_length = st.sidebar.slider(
            "Target Length (words)",
            min_value=50,
            max_value=1000,
            value=300,
            step=50
        )
        
        temperature = st.sidebar.slider(
            "Creativity (Temperature)",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Higher values make output more creative but less focused"
        )
        
        top_p = st.sidebar.slider(
            "Focus (Top-p)",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Lower values make output more focused"
        )
        
        # Keywords input
        keywords_input = st.sidebar.text_area(
            "Keywords (one per line)",
            placeholder="artificial intelligence\nmachine learning\nfuture"
        )
        
        keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()] if keywords_input else []
        
        config = GenerationConfig(
            content_type=ContentType(content_type),
            style=StyleType(style),
            tone=ToneType(tone),
            target_length=target_length,
            keywords=keywords,
            temperature=temperature,
            top_p=top_p
        )
        
        return config
    
    def render_main_interface(self, config):
        """Render main content generation interface"""
        
        st.title("🤖 AI Content Generation System")
        st.markdown("Generate high-quality content with controllable style and tone")
        
        # Topic input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            topic = st.text_input(
                "Topic or Title",
                placeholder="Enter your content topic...",
                help="Describe what you want to write about"
            )
        
        with col2:
            generate_variants = st.checkbox("Generate variants", value=False)
            num_variants = st.selectbox("Number of variants", [1, 2, 3], index=0) if generate_variants else 1
        
        # Content suggestions
        if topic:
            with st.expander("💡 Content Suggestions", expanded=False):
                suggestions = self.engine.get_content_suggestions(topic, config.content_type)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Structure Tips:**")
                    for tip in suggestions["structure_tips"]:
                        st.write(f"• {tip}")
                
                with col2:
                    st.write("**Content Angles:**")
                    for angle in suggestions["angles"]:
                        st.write(f"• {angle}")
        
        # Generation button
        if st.button("🚀 Generate Content", type="primary", disabled=not topic):
            return self.generate_content(config, topic, num_variants)
        
        return None
    
    async def generate_content(self, config, topic, num_variants=1):
        """Generate content and display results"""
        
        with st.spinner(f"Generating content... ({num_variants} variant{'s' if num_variants > 1 else ''})"):
            
            if num_variants == 1:
                result = await self.engine.generate_with_style_control(config, topic)
                results = [result]
            else:
                results = await self.engine.generate_multiple_variants(config, topic, num_variants)
            
            # Evaluate each result
            for result in results:
                evaluation = self.evaluator.evaluate_content(result["generated_text"], config)
                result["evaluation"] = evaluation
            
            # Store in session state
            st.session_state.generated_content = results
            st.session_state.generation_history.append({
                'timestamp': datetime.now(),
                'topic': topic,
                'config': config,
                'results': results
            })
            
            return results
    
    def render_results(self, results):
        """Render generation results"""
        
        if not results:
            return
        
        st.subheader("📄 Generated Content")
        
        for i, result in enumerate(results):
            with st.container():
                if len(results) > 1:
                    st.markdown(f"### Variant {i+1}")
                
                # Content display
                st.markdown("**Generated Text:**")
                st.text_area(
                    "Content",
                    value=result["generated_text"],
                    height=300,
                    key=f"content_{i}",
                    label_visibility="collapsed"
                )
                
                # Metrics and evaluation
                col1, col2, col3, col4 = st.columns(4)
                
                evaluation = result.get("evaluation", {})
                
                with col1:
                    st.metric("Word Count", result["word_count"])
                
                with col2:
                    st.metric("Quality Score", f"{evaluation.get('overall_quality', 0):.2f}")
                
                with col3:
                    st.metric("Generation Time", f"{result['generation_time']:.2f}s")
                
                with col4:
                    st.metric("Readability", f"{evaluation.get('readability_score', 0):.2f}")
                
                # Detailed evaluation
                with st.expander("📊 Detailed Evaluation", expanded=False):
                    self.render_evaluation_details(evaluation)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"📋 Copy to Clipboard", key=f"copy_{i}"):
                        st.write("Content copied! (Note: Use browser copy functionality)")
                
                with col2:
                    if st.button(f"💾 Save to File", key=f"save_{i}"):
                        self.save_content_to_file(result)
                
                with col3:
                    if st.button(f"🔄 Regenerate Similar", key=f"regen_{i}"):
                        # Would trigger regeneration with similar parameters
                        st.info("Regeneration feature coming soon!")
                
                st.divider()
    
    def render_evaluation_details(self, evaluation):
        """Render detailed evaluation metrics"""
        
        if not evaluation:
            st.write("No evaluation data available")
            return
        
        # Create radar chart for quality metrics
        metrics = ['readability_score', 'style_consistency', 'tone_consistency', 
                  'structure_score', 'grammar_score']
        
        available_metrics = {k: v for k, v in evaluation.items() if k in metrics}
        
        if available_metrics:
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(available_metrics.values()),
                theta=[m.replace('_', ' ').title() for m in available_metrics.keys()],
                fill='toself',
                name='Quality Metrics'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Content Quality Metrics"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Quality Metrics:**")
            for key, value in evaluation.items():
                if isinstance(value, (int, float)):
                    st.write(f"• {key.replace('_', ' ').title()}: {value:.3f}")
        
        with col2:
            if 'keyword_usage' in evaluation:
                st.write("**Keyword Analysis:**")
                st.write(f"• Keywords Used: {evaluation['keyword_usage']:.1%}")
    
    def render_analytics_dashboard(self):
        """Render analytics dashboard"""
        
        if not st.session_state.generation_history:
            st.info("No generation history available yet. Generate some content first!")
            return
        
        st.subheader("📈 Analytics Dashboard")
        
        # Prepare data
        history_data = []
        for entry in st.session_state.generation_history:
            for result in entry['results']:
                history_data.append({
                    'timestamp': entry['timestamp'],
                    'topic': entry['topic'],
                    'content_type': entry['config'].content_type.value,
                    'style': entry['config'].style.value,
                    'tone': entry['config'].tone.value,
                    'word_count': result['word_count'],
                    'quality_score': result.get('evaluation', {}).get('overall_quality', 0),
                    'generation_time': result['generation_time']
                })
        
        df = pd.DataFrame(history_data)
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Generations", len(df))
        
        with col2:
            st.metric("Avg Quality Score", f"{df['quality_score'].mean():.2f}")
        
        with col3:
            st.metric("Avg Word Count", f"{df['word_count'].mean():.0f}")
        
        with col4:
            st.metric("Avg Generation Time", f"{df['generation_time'].mean():.2f}s")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality by content type
            fig = px.box(df, x='content_type', y='quality_score', 
                        title='Quality Score by Content Type')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Generation time by style
            fig = px.scatter(df, x='word_count', y='generation_time', 
                           color='style', title='Generation Time vs Word Count')
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent generations table
        st.subheader("Recent Generations")
        recent_df = df.tail(10)[['timestamp', 'topic', 'content_type', 'quality_score', 'word_count']]
        st.dataframe(recent_df, use_container_width=True)
    
    def save_content_to_file(self, result):
        """Save generated content to file"""
        
        content_data = {
            'generated_text': result['generated_text'],
            'metadata': result.get('metadata', {}),
            'evaluation': result.get('evaluation', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Create downloadable file
        json_str = json.dumps(content_data, indent=2)
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name=f"generated_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def run(self):
        """Run the Streamlit application"""
        
        st.set_page_config(
            page_title="AI Content Generation System",
            page_icon="🤖",
            layout="wide"
        )
        
        # Initialize system
        try:
            engine, evaluator = asyncio.run(self.initialize_system())
            self.engine = engine
            self.evaluator = evaluator
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            return
        
        # Sidebar controls
        config = self.render_sidebar_controls()
        
        # Main interface
        tab1, tab2, tab3 = st.tabs(["🎯 Generate", "📊 Analytics", "ℹ️ About"])
        
        with tab1:
            results = self.render_main_interface(config)
            
            # Display results if available
            if st.session_state.generated_content:
                self.render_results(st.session_state.generated_content)
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
            st.markdown("""
            ## About this Content Generation System
            
            This AI-powered content generation system helps you create high-quality content with:
            
            ### Features:
            - **Multiple Content Types**: Articles, blog posts, social media, emails, and more
            - **Style Control**: Choose from formal, casual, academic, creative, and other styles
            - **Tone Adjustment**: Set the emotional tone from friendly to authoritative
            - **Quality Evaluation**: Comprehensive metrics for content assessment
            - **Multiple Variants**: Generate different versions for comparison
            
            ### How it Works:
            1. **Input Configuration**: Set your content type, style, tone, and keywords
            2. **AI Generation**: Advanced language models create content based on your specifications
            3. **Quality Evaluation**: Automated assessment of readability, style consistency, and structure
            4. **Refinement**: Post-processing ensures content meets quality standards
            
            ### Technologies Used:
            - **Language Models**: GPT-2, T5, and other transformer architectures
            - **Quality Metrics**: Custom evaluation algorithms for style and tone
            - **User Interface**: Streamlit for interactive web application
            """)

# Run the application
if __name__ == "__main__":
    app = ContentGenerationUI()
    app.run()
```

## 3. API Integration and Deployment

### RESTful API Service

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
import logging

# API Models
class ContentGenerationRequest(BaseModel):
    topic: str = Field(..., description="Content topic or title")
    content_type: str = Field(..., description="Type of content to generate")
    style: str = Field(..., description="Writing style")
    tone: str = Field(..., description="Writing tone")
    target_length: int = Field(300, ge=50, le=2000, description="Target word count")
    keywords: Optional[List[str]] = Field(None, description="Keywords to include")
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    num_variants: int = Field(1, ge=1, le=5)

class ContentGenerationResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    generation_time: float
    metadata: Dict[str, Any]

class ContentEvaluationRequest(BaseModel):
    content: str
    content_type: str
    style: str
    tone: str
    keywords: Optional[List[str]] = None

class ContentEvaluationResponse(BaseModel):
    evaluation: Dict[str, float]
    suggestions: List[str]

# FastAPI application
app = FastAPI(
    title="Content Generation API",
    description="AI-powered content generation with style and tone control",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
content_engine = None
content_evaluator = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global content_engine, content_evaluator
    
    logging.info("Initializing content generation system...")
    
    content_engine = ContentGenerationEngine()
    content_evaluator = ContentQualityEvaluator()
    
    await content_engine.load_models()
    
    logging.info("Content generation system initialized")

@app.post("/generate", response_model=ContentGenerationResponse)
async def generate_content(request: ContentGenerationRequest):
    """Generate content based on specifications"""
    
    try:
        # Create configuration
        config = GenerationConfig(
            content_type=ContentType(request.content_type),
            style=StyleType(request.style),
            tone=ToneType(request.tone),
            target_length=request.target_length,
            keywords=request.keywords,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Generate content
        if request.num_variants == 1:
            result = await content_engine.generate_with_style_control(config, request.topic)
            results = [result]
        else:
            results = await content_engine.generate_multiple_variants(
                config, request.topic, request.num_variants
            )
        
        # Evaluate results
        for result in results:
            evaluation = content_evaluator.evaluate_content(result["generated_text"], config)
            result["evaluation"] = evaluation
        
        # Calculate total generation time
        total_time = sum(result["generation_time"] for result in results)
        
        return ContentGenerationResponse(
            success=True,
            results=results,
            generation_time=total_time,
            metadata={
                "num_variants": len(results),
                "config": config.__dict__
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=ContentEvaluationResponse)
async def evaluate_content(request: ContentEvaluationRequest):
    """Evaluate content quality"""
    
    try:
        config = GenerationConfig(
            content_type=ContentType(request.content_type),
            style=StyleType(request.style),
            tone=ToneType(request.tone),
            keywords=request.keywords
        )
        
        evaluation = content_evaluator.evaluate_content(request.content, config)
        
        # Generate suggestions based on evaluation
        suggestions = []
        
        if evaluation.get('readability_score', 1) < 0.5:
            suggestions.append("Consider simplifying sentence structure for better readability")
        
        if evaluation.get('style_consistency', 1) < 0.6:
            suggestions.append(f"Content could better match the {request.style} style")
        
        if evaluation.get('structure_score', 1) < 0.7:
            suggestions.append("Consider improving content structure and organization")
        
        if request.keywords and evaluation.get('keyword_usage', 1) < 0.5:
            suggestions.append("Consider including more of the specified keywords")
        
        return ContentEvaluationResponse(
            evaluation=evaluation,
            suggestions=suggestions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suggestions/{content_type}")
async def get_content_suggestions(content_type: str, topic: str):
    """Get content structure and keyword suggestions"""
    
    try:
        suggestions = content_engine.get_content_suggestions(topic, ContentType(content_type))
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    
    return {
        "models_loaded": list(content_engine.models.keys()) if content_engine else [],
        "supported_content_types": [ct.value for ct in ContentType],
        "supported_styles": [st.value for st in StyleType],
        "supported_tones": [tt.value for tt in ToneType]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    return {
        "status": "healthy",
        "models_loaded": content_engine is not None,
        "evaluator_loaded": content_evaluator is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 4. Learning Objectives

By the end of this project, you should be able to:
- **Build** end-to-end content generation systems
- **Implement** style and tone control mechanisms
- **Design** quality evaluation frameworks
- **Create** user-friendly interfaces for AI systems
- **Deploy** production-ready content generation APIs
- **Integrate** multiple NLP techniques effectively

### Self-Assessment Checklist

□ Can control content generation style and tone  
□ Can implement quality evaluation metrics  
□ Can build user interfaces for AI systems  
□ Can create RESTful APIs for content generation  
□ Can integrate multiple AI models effectively  
□ Can deploy scalable content generation systems  
□ Can optimize generation for different content types  

## 5. Practical Exercises

**Exercise 1: Enhanced Style Control**
```python
# TODO: Implement advanced style transfer techniques
# Add support for custom writing styles
# Include style fine-tuning capabilities
```

**Exercise 2: Content Planning System**
```python
# TODO: Build content planning and outlining features
# Add content calendar integration
# Include SEO optimization suggestions
```

**Exercise 3: Multi-language Support**
```python
# TODO: Extend system to support multiple languages
# Add translation capabilities
# Include cultural adaptation features
```

## 6. Study Materials

### Essential Resources
- **Controllable Text Generation**: https://arxiv.org/abs/1909.05858
- **CTRL: A Conditional Transformer**: https://arxiv.org/abs/1909.05858
- **Content Quality Evaluation**: Various academic papers on automatic text evaluation
- **Streamlit Documentation**: https://docs.streamlit.io/

### Advanced Topics
- **Fine-tuning for Style**: Domain adaptation techniques
- **Reinforcement Learning**: RLHF for content quality
- **Multi-modal Generation**: Combining text with images
- **Personalization**: User-specific content adaptation

## 7. Extensions and Improvements

### Potential Enhancements
1. **Advanced Style Models**: Train specialized models for each style
2. **Content Templates**: Pre-built templates for common content types
3. **SEO Integration**: Automatic SEO optimization and keyword density analysis
4. **Plagiarism Detection**: Integration with plagiarism checking services
5. **A/B Testing**: Built-in A/B testing for content variants
6. **Analytics Integration**: Connect with Google Analytics and social media metrics
7. **Voice and Brand**: Custom brand voice training and consistency checks
8. **Collaborative Features**: Multi-user content creation and review workflows
