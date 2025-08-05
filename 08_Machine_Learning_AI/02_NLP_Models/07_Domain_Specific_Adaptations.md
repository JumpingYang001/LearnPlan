# Domain-Specific Adaptations

*Duration: 3 weeks*

## Overview

Domain-specific adaptation is crucial for deploying transformer models in specialized fields like medicine, law, finance, and scientific research. This section covers techniques for adapting general-purpose models to specific domains while maintaining their broad capabilities.

## 1. Domain Adaptation Techniques

### Understanding Domain Shift

Domain shift occurs when the training data distribution differs from the target domain distribution. This affects model performance in specialized areas.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

class DomainAnalyzer:
    """Analyze domain differences in text data"""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_embeddings(self, texts, max_length=512):
        """Get contextualized embeddings for texts"""
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                max_length=max_length,
                truncation=True, 
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def visualize_domain_shift(self, general_texts, domain_texts, domain_name):
        """Visualize domain shift using t-SNE"""
        
        # Get embeddings
        general_embs = self.get_embeddings(general_texts)
        domain_embs = self.get_embeddings(domain_texts)
        
        # Combine embeddings
        all_embeddings = np.vstack([general_embs, domain_embs])
        labels = ['General'] * len(general_texts) + [domain_name] * len(domain_texts)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']
        for i, label in enumerate(['General', domain_name]):
            mask = np.array(labels) == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=colors[i], label=label, alpha=0.6)
        
        plt.title(f'Domain Shift Visualization: General vs {domain_name}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def calculate_domain_distance(self, general_texts, domain_texts):
        """Calculate distance between domain distributions"""
        
        general_embs = self.get_embeddings(general_texts)
        domain_embs = self.get_embeddings(domain_texts)
        
        # Calculate mean embeddings
        general_mean = np.mean(general_embs, axis=0)
        domain_mean = np.mean(domain_embs, axis=0)
        
        # Euclidean distance between means
        distance = np.linalg.norm(general_mean - domain_mean)
        
        # Cosine similarity
        cosine_sim = np.dot(general_mean, domain_mean) / (
            np.linalg.norm(general_mean) * np.linalg.norm(domain_mean)
        )
        
        return {
            'euclidean_distance': distance,
            'cosine_similarity': cosine_sim,
            'cosine_distance': 1 - cosine_sim
        }

# Example usage
def analyze_medical_domain():
    """Analyze medical domain shift"""
    
    analyzer = DomainAnalyzer()
    
    general_texts = [
        "The weather is nice today.",
        "I'm going to the store to buy groceries.",
        "The movie was really entertaining.",
        "Technology is advancing rapidly.",
        "Sports are important for health."
    ]
    
    medical_texts = [
        "The patient presents with acute myocardial infarction.",
        "Administer 10mg morphine sulfate intravenously.",
        "Differential diagnosis includes pneumonia versus pulmonary embolism.",
        "Laboratory results show elevated troponin levels.",
        "Recommend echocardiogram to assess cardiac function."
    ]
    
    # Calculate domain distance
    distances = analyzer.calculate_domain_distance(general_texts, medical_texts)
    print("Medical Domain Analysis:")
    for metric, value in distances.items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize domain shift
    analyzer.visualize_domain_shift(general_texts, medical_texts, "Medical")
    
    return distances

# distances = analyze_medical_domain()
```

### Domain Adaptation Strategies

```python
from enum import Enum

class DomainAdaptationStrategy(Enum):
    CONTINUED_PRETRAINING = "continued_pretraining"
    FINE_TUNING = "fine_tuning"
    ADVERSARIAL_ADAPTATION = "adversarial_adaptation"
    MULTI_TASK_LEARNING = "multi_task_learning"
    GRADUAL_UNFREEZING = "gradual_unfreezing"

class DomainAdaptationPipeline:
    """Complete pipeline for domain adaptation"""
    
    def __init__(self, base_model_name, domain_name):
        self.base_model_name = base_model_name
        self.domain_name = domain_name
        self.adaptation_history = []
    
    def prepare_domain_corpus(self, raw_texts, vocab_expansion=True):
        """Prepare domain-specific corpus for adaptation"""
        
        # Load base tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        if vocab_expansion:
            # Extract domain-specific terms
            domain_terms = self.extract_domain_terms(raw_texts, tokenizer)
            
            # Expand vocabulary
            expanded_tokenizer = self.expand_vocabulary(tokenizer, domain_terms)
            
            return expanded_tokenizer, domain_terms
        
        return tokenizer, []
    
    def extract_domain_terms(self, texts, tokenizer, min_freq=5):
        """Extract domain-specific terms not in base vocabulary"""
        
        from collections import Counter
        import re
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            # Simple tokenization (could use more sophisticated methods)
            tokens = re.findall(r'\b\w+\b', text.lower())
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Find tokens not in base vocabulary
        base_vocab = set(tokenizer.vocab.keys())
        domain_terms = []
        
        for token, count in token_counts.items():
            if count >= min_freq and token not in base_vocab:
                domain_terms.append((token, count))
        
        # Sort by frequency
        domain_terms.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(domain_terms)} domain-specific terms")
        print("Top 10 domain terms:")
        for term, freq in domain_terms[:10]:
            print(f"  {term}: {freq}")
        
        return [term for term, _ in domain_terms]
    
    def expand_vocabulary(self, tokenizer, new_terms, max_new_tokens=1000):
        """Expand tokenizer vocabulary with domain terms"""
        
        # Create new tokenizer with expanded vocabulary
        new_vocab = tokenizer.vocab.copy()
        
        # Add new terms (limit to max_new_tokens)
        added_count = 0
        for term in new_terms:
            if added_count >= max_new_tokens:
                break
            
            if term not in new_vocab:
                new_vocab[term] = len(new_vocab)
                added_count += 1
        
        print(f"Added {added_count} new tokens to vocabulary")
        print(f"New vocabulary size: {len(new_vocab)}")
        
        # Update tokenizer (simplified - in practice would need more careful handling)
        tokenizer.vocab = new_vocab
        
        return tokenizer
    
    def create_adaptation_datasets(self, domain_texts, strategy):
        """Create datasets for different adaptation strategies"""
        
        if strategy == DomainAdaptationStrategy.CONTINUED_PRETRAINING:
            return self.create_mlm_dataset(domain_texts)
        elif strategy == DomainAdaptationStrategy.FINE_TUNING:
            return self.create_supervised_dataset(domain_texts)
        elif strategy == DomainAdaptationStrategy.MULTI_TASK_LEARNING:
            return self.create_multitask_dataset(domain_texts)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    
    def create_mlm_dataset(self, texts):
        """Create masked language modeling dataset for continued pre-training"""
        
        class MLMDataset:
            def __init__(self, texts, tokenizer, max_length=512, mlm_probability=0.15):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.mlm_probability = mlm_probability
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].clone()
                labels = input_ids.clone()
                
                # Create random mask
                probability_matrix = torch.full(labels.shape, self.mlm_probability)
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in labels.tolist()
                ]
                probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
                
                masked_indices = torch.bernoulli(probability_matrix).bool()
                labels[~masked_indices] = -100  # Only compute loss on masked tokens
                
                # 80% mask, 10% random, 10% unchanged
                indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
                input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                
                indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
                input_ids[indices_random] = random_words[indices_random]
                
                return {
                    'input_ids': input_ids.flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': labels.flatten()
                }
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        return MLMDataset(texts, tokenizer)
    
    def create_supervised_dataset(self, labeled_data):
        """Create supervised dataset for fine-tuning"""
        
        class SupervisedDataset:
            def __init__(self, data, tokenizer, max_length=512):
                self.data = data  # List of (text, label) tuples
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                text, label = self.data[idx]
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        return SupervisedDataset(labeled_data, tokenizer)

# Example domain adaptation workflow
def domain_adaptation_workflow():
    """Complete domain adaptation workflow"""
    
    # Initialize pipeline
    pipeline = DomainAdaptationPipeline(
        base_model_name="bert-base-uncased",
        domain_name="medical"
    )
    
    # Sample domain corpus
    medical_texts = [
        "The patient presented with symptoms of acute coronary syndrome.",
        "Electrocardiogram showed ST-elevation myocardial infarction.",
        "Treatment included percutaneous coronary intervention.",
        "Post-operative recovery was uncomplicated.",
        "Discharge medications included dual antiplatelet therapy."
    ]
    
    # Prepare domain corpus
    tokenizer, domain_terms = pipeline.prepare_domain_corpus(
        medical_texts, vocab_expansion=True
    )
    
    # Create MLM dataset for continued pre-training
    mlm_dataset = pipeline.create_mlm_dataset(medical_texts)
    
    print(f"Created MLM dataset with {len(mlm_dataset)} samples")
    
    return pipeline, mlm_dataset

# pipeline, dataset = domain_adaptation_workflow()
```

## 2. Continued Pre-training

### Implementing Continued Pre-training

```python
from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from torch.utils.data import Dataset

class ContinuedPretrainingFramework:
    """Framework for continued pre-training on domain data"""
    
    def __init__(self, base_model_name, domain_name, output_dir):
        self.base_model_name = base_model_name
        self.domain_name = domain_name
        self.output_dir = output_dir
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(base_model_name)
        
        # Handle tokenizer padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_domain_dataset(self, domain_texts, chunk_size=512):
        """Prepare domain dataset for continued pre-training"""
        
        class DomainTextDataset(Dataset):
            def __init__(self, texts, tokenizer, chunk_size):
                self.tokenizer = tokenizer
                self.chunk_size = chunk_size
                
                # Tokenize all texts and create chunks
                self.input_ids = []
                
                for text in texts:
                    # Tokenize text
                    encoded = tokenizer.encode(text, add_special_tokens=True)
                    
                    # Split into chunks
                    for i in range(0, len(encoded), chunk_size - 2):  # -2 for special tokens
                        chunk = encoded[i:i + chunk_size - 2]
                        
                        # Add special tokens
                        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
                        
                        # Pad if necessary
                        if len(chunk) < chunk_size:
                            chunk.extend([tokenizer.pad_token_id] * (chunk_size - len(chunk)))
                        
                        self.input_ids.append(chunk[:chunk_size])
            
            def __len__(self):
                return len(self.input_ids)
            
            def __getitem__(self, idx):
                return {'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long)}
        
        return DomainTextDataset(domain_texts, self.tokenizer, chunk_size)
    
    def create_training_arguments(self, 
                                 num_epochs=3,
                                 batch_size=8,
                                 learning_rate=5e-5,
                                 warmup_steps=1000):
        """Create training arguments for continued pre-training"""
        
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=100,
            save_steps=1000,
            save_total_limit=3,
            prediction_loss_only=True,
            fp16=True,  # Use mixed precision for efficiency
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )
    
    def train(self, domain_dataset, eval_dataset=None):
        """Train model with continued pre-training"""
        
        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        # Training arguments
        training_args = self.create_training_arguments()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=domain_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        print(f"Starting continued pre-training for {self.domain_name} domain...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model saved to {self.output_dir}")
        
        return trainer
    
    def evaluate_adaptation(self, test_texts, original_model_name=None):
        """Evaluate domain adaptation performance"""
        
        if original_model_name is None:
            original_model_name = self.base_model_name
        
        # Load original model for comparison
        original_model = AutoModelForMaskedLM.from_pretrained(original_model_name)
        original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
        
        # Evaluate perplexity on domain texts
        adapted_perplexity = self.calculate_perplexity(test_texts, self.model, self.tokenizer)
        original_perplexity = self.calculate_perplexity(test_texts, original_model, original_tokenizer)
        
        improvement = (original_perplexity - adapted_perplexity) / original_perplexity * 100
        
        print(f"\nDomain Adaptation Evaluation:")
        print(f"Original model perplexity: {original_perplexity:.2f}")
        print(f"Adapted model perplexity: {adapted_perplexity:.2f}")
        print(f"Improvement: {improvement:.1f}%")
        
        return {
            'original_perplexity': original_perplexity,
            'adapted_perplexity': adapted_perplexity,
            'improvement_percentage': improvement
        }
    
    def calculate_perplexity(self, texts, model, tokenizer):
        """Calculate perplexity on given texts"""
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                
                # Forward pass
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Count tokens (excluding padding)
                num_tokens = (inputs['input_ids'] != tokenizer.pad_token_id).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity

# Example: Medical domain continued pre-training
def medical_continued_pretraining():
    """Example of continued pre-training for medical domain"""
    
    # Sample medical texts (in practice, would use much larger corpus)
    medical_corpus = [
        "The patient was diagnosed with acute myocardial infarction following electrocardiographic evidence of ST-elevation in leads II, III, and aVF.",
        "Echocardiography revealed left ventricular ejection fraction of 35% with regional wall motion abnormalities.",
        "Laboratory findings included elevated troponin I levels at 15.2 ng/mL and creatine kinase-MB of 45 U/L.",
        "Treatment protocol included dual antiplatelet therapy with aspirin and clopidogrel, along with atorvastatin.",
        "Post-procedural angiography demonstrated TIMI grade 3 flow in all coronary vessels.",
        "The patient underwent percutaneous coronary intervention with drug-eluting stent placement.",
        "Discharge medications included metoprolol, lisinopril, and sublingual nitroglycerin as needed.",
        "Follow-up echocardiogram at 6 months showed improvement in left ventricular function.",
        "Cardiac catheterization revealed 90% stenosis in the left anterior descending artery.",
        "The patient was started on optimal medical therapy including beta-blockers and ACE inhibitors."
    ]
    
    # Initialize framework
    framework = ContinuedPretrainingFramework(
        base_model_name="bert-base-uncased",
        domain_name="medical",
        output_dir="./medical-bert"
    )
    
    # Prepare dataset
    train_dataset = framework.prepare_domain_dataset(medical_corpus)
    
    print(f"Created training dataset with {len(train_dataset)} samples")
    
    # Train model
    trainer = framework.train(train_dataset)
    
    # Evaluate adaptation
    test_texts = medical_corpus[:3]  # Use some texts for evaluation
    results = framework.evaluate_adaptation(test_texts)
    
    return framework, trainer, results

# framework, trainer, results = medical_continued_pretraining()
```

### Advanced Continued Pre-training Strategies

```python
class AdvancedPretrainingStrategies:
    """Advanced strategies for domain-specific continued pre-training"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def gradual_unfreezing(self, domain_dataset, num_phases=3):
        """Gradually unfreeze layers during continued pre-training"""
        
        # Get all transformer layers
        if hasattr(self.model, 'bert'):
            transformer_layers = self.model.bert.encoder.layer
        elif hasattr(self.model, 'transformer'):
            transformer_layers = self.model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")
        
        num_layers = len(transformer_layers)
        layers_per_phase = num_layers // num_phases
        
        training_phases = []
        
        for phase in range(num_phases):
            # Determine which layers to unfreeze in this phase
            start_layer = max(0, num_layers - (phase + 1) * layers_per_phase)
            end_layer = num_layers
            
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze embedding layer (always trainable)
            if hasattr(self.model, 'bert'):
                for param in self.model.bert.embeddings.parameters():
                    param.requires_grad = True
            
            # Unfreeze selected transformer layers
            for i in range(start_layer, end_layer):
                for param in transformer_layers[i].parameters():
                    param.requires_grad = True
            
            # Unfreeze classification head
            if hasattr(self.model, 'cls'):
                for param in self.model.cls.parameters():
                    param.requires_grad = True
            
            # Create training arguments for this phase
            training_args = TrainingArguments(
                output_dir=f'./gradual_unfreezing_phase_{phase}',
                num_train_epochs=2,
                per_device_train_batch_size=8,
                learning_rate=5e-5 * (0.5 ** phase),  # Decrease LR each phase
                warmup_steps=100,
                logging_steps=50,
                save_steps=500,
            )
            
            training_phases.append({
                'phase': phase,
                'unfrozen_layers': list(range(start_layer, end_layer)),
                'training_args': training_args
            })
        
        return training_phases
    
    def cyclical_learning_rates(self, base_lr=1e-5, max_lr=5e-4, cycle_length=1000):
        """Implement cyclical learning rates for better convergence"""
        
        def lr_scheduler(step):
            cycle = step // cycle_length
            x = step % cycle_length / cycle_length
            
            if x <= 0.5:
                # Increasing phase
                lr = base_lr + (max_lr - base_lr) * 2 * x
            else:
                # Decreasing phase
                lr = max_lr - (max_lr - base_lr) * 2 * (x - 0.5)
            
            return lr / base_lr  # Return multiplier
        
        return lr_scheduler
    
    def domain_specific_objectives(self, domain_type):
        """Create domain-specific training objectives"""
        
        if domain_type == "medical":
            return self.medical_objectives()
        elif domain_type == "legal":
            return self.legal_objectives()
        elif domain_type == "scientific":
            return self.scientific_objectives()
        else:
            return self.general_objectives()
    
    def medical_objectives(self):
        """Medical domain-specific objectives"""
        
        objectives = {
            'entity_recognition': {
                'weight': 0.3,
                'description': 'Recognize medical entities (drugs, conditions, procedures)'
            },
            'relation_extraction': {
                'weight': 0.2,
                'description': 'Extract relationships between medical concepts'
            },
            'clinical_reasoning': {
                'weight': 0.3,
                'description': 'Understanding clinical decision making'
            },
            'terminology_masking': {
                'weight': 0.2,
                'description': 'Focused masking of medical terminology'
            }
        }
        
        return objectives
    
    def legal_objectives(self):
        """Legal domain-specific objectives"""
        
        objectives = {
            'case_law_understanding': {
                'weight': 0.4,
                'description': 'Understanding legal precedents and citations'
            },
            'contract_analysis': {
                'weight': 0.3,
                'description': 'Analyzing contract terms and clauses'
            },
            'legal_reasoning': {
                'weight': 0.2,
                'description': 'Logical reasoning in legal contexts'
            },
            'statutory_interpretation': {
                'weight': 0.1,
                'description': 'Understanding statutory language'
            }
        }
        
        return objectives
    
    def implement_domain_objectives(self, dataset, objectives):
        """Implement multi-objective training"""
        
        class MultiObjectiveTrainer(Trainer):
            def __init__(self, objectives, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.objectives = objectives
            
            def compute_loss(self, model, inputs):
                # Standard MLM loss
                outputs = model(**inputs)
                mlm_loss = outputs.loss
                
                total_loss = mlm_loss
                
                # Add domain-specific losses (simplified implementation)
                for obj_name, obj_config in self.objectives.items():
                    if obj_name == 'terminology_masking':
                        # Increase masking probability for domain terms
                        term_loss = self.compute_terminology_loss(model, inputs)
                        total_loss += obj_config['weight'] * term_loss
                
                return total_loss
            
            def compute_terminology_loss(self, model, inputs):
                # Simplified terminology-focused loss
                # In practice, would identify domain terms and apply higher masking
                return torch.tensor(0.0, device=model.device)
        
        return MultiObjectiveTrainer

# Example usage
def advanced_medical_pretraining():
    """Advanced medical domain pre-training example"""
    
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    # Load model
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Initialize advanced strategies
    strategies = AdvancedPretrainingStrategies(model, tokenizer)
    
    # Get medical objectives
    objectives = strategies.domain_specific_objectives("medical")
    
    print("Medical Domain Objectives:")
    for obj, config in objectives.items():
        print(f"  {obj}: {config['description']} (weight: {config['weight']})")
    
    # Plan gradual unfreezing
    # phases = strategies.gradual_unfreezing(domain_dataset, num_phases=3)
    
    print(f"\nPlanned {len(objectives)} training phases with gradual unfreezing")
    
    return strategies, objectives

# strategies, objectives = advanced_medical_pretraining()
```

## 3. Specialized Models for Different Domains

### Medical Domain Models

```python
class MedicalDomainModel:
    """Specialized model for medical domain applications"""
    
    def __init__(self, base_model_name="bert-base-uncased"):
        self.base_model_name = base_model_name
        self.medical_vocabulary = self.load_medical_vocabulary()
        self.model = None
        self.tokenizer = None
    
    def load_medical_vocabulary(self):
        """Load medical domain vocabulary"""
        
        # Sample medical terms (in practice, would load from medical ontologies)
        medical_vocab = {
            'conditions': [
                'myocardial infarction', 'pneumonia', 'diabetes mellitus',
                'hypertension', 'atrial fibrillation', 'stroke', 'sepsis'
            ],
            'medications': [
                'aspirin', 'metformin', 'lisinopril', 'atorvastatin',
                'metoprolol', 'warfarin', 'insulin', 'prednisone'
            ],
            'procedures': [
                'percutaneous coronary intervention', 'echocardiography',
                'computed tomography', 'magnetic resonance imaging',
                'endotracheal intubation', 'lumbar puncture'
            ],
            'anatomy': [
                'left ventricle', 'coronary artery', 'pulmonary embolism',
                'cerebral cortex', 'hepatic function', 'renal failure'
            ]
        }
        
        return medical_vocab
    
    def create_medical_tokenizer(self):
        """Create tokenizer with expanded medical vocabulary"""
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Collect all medical terms
        all_medical_terms = []
        for category, terms in self.medical_vocabulary.items():
            all_medical_terms.extend(terms)
        
        # Add medical terms to vocabulary
        new_tokens = [term for term in all_medical_terms if term not in tokenizer.vocab]
        
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
            print(f"Added {len(new_tokens)} medical terms to vocabulary")
        
        return tokenizer
    
    def create_medical_model(self, num_labels=None):
        """Create model optimized for medical tasks"""
        
        if num_labels:
            # Classification model
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name, 
                num_labels=num_labels
            )
        else:
            # MLM model for continued pre-training
            from transformers import AutoModelForMaskedLM
            model = AutoModelForMaskedLM.from_pretrained(self.base_model_name)
        
        # Resize token embeddings if vocabulary was expanded
        if self.tokenizer and len(self.tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(self.tokenizer))
        
        return model
    
    def create_medical_datasets(self):
        """Create medical domain datasets"""
        
        # Sample medical data
        medical_data = {
            'clinical_notes': [
                "Patient presents with chest pain radiating to left arm, diaphoresis, and nausea. ECG shows ST elevation in leads V1-V4.",
                "57-year-old male with history of diabetes mellitus type 2 and hypertension presents with shortness of breath.",
                "Physical examination reveals bilateral rales, elevated JVD, and lower extremity edema consistent with heart failure.",
                "Laboratory results: BNP 1200 pg/mL, troponin I 0.8 ng/mL, creatinine 1.4 mg/dL.",
                "Treatment plan includes IV furosemide, ACE inhibitor, and beta-blocker therapy."
            ],
            'discharge_summaries': [
                "Patient admitted with acute exacerbation of COPD, treated with steroids and bronchodilators.",
                "Successful percutaneous coronary intervention with drug-eluting stent placement in LAD.",
                "Post-operative course complicated by atrial fibrillation, managed with rate control.",
                "Patient discharged on dual antiplatelet therapy and statin medication.",
                "Follow-up appointment scheduled with cardiology in 2 weeks."
            ],
            'drug_interactions': [
                "Warfarin interacts with aspirin, increasing bleeding risk. Monitor INR closely.",
                "Metformin contraindicated in patients with severe renal impairment (eGFR < 30).",
                "Digoxin levels may be elevated with concurrent amiodarone therapy.",
                "ACE inhibitors can cause hyperkalemia, especially with concomitant potassium supplementation.",
                "Beta-blockers may mask hypoglycemic symptoms in diabetic patients."
            ]
        }
        
        return medical_data
    
    def fine_tune_for_medical_tasks(self, task_type="classification"):
        """Fine-tune model for specific medical tasks"""
        
        # Create medical-optimized tokenizer and model
        self.tokenizer = self.create_medical_tokenizer()
        
        if task_type == "classification":
            # Example: Classify medical notes by urgency
            self.model = self.create_medical_model(num_labels=3)  # High, Medium, Low urgency
            
            # Create classification dataset
            medical_texts = [
                "Patient with acute MI, needs immediate intervention",  # High urgency
                "Routine follow-up for diabetes management",  # Low urgency
                "Chest pain evaluation, possible cardiac origin",  # Medium urgency
            ]
            labels = [2, 0, 1]  # High, Low, Medium
            
            dataset = list(zip(medical_texts, labels))
            
        elif task_type == "ner":
            # Named Entity Recognition for medical entities
            from transformers import AutoModelForTokenClassification
            
            # Medical NER labels
            medical_labels = [
                'O',  # Outside
                'B-CONDITION', 'I-CONDITION',  # Medical conditions
                'B-MEDICATION', 'I-MEDICATION',  # Medications
                'B-PROCEDURE', 'I-PROCEDURE',  # Medical procedures
                'B-ANATOMY', 'I-ANATOMY',  # Anatomical terms
            ]
            
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.base_model_name,
                num_labels=len(medical_labels)
            )
            
            # Resize embeddings
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        return self.model, self.tokenizer
    
    def evaluate_medical_performance(self, test_data):
        """Evaluate model performance on medical tasks"""
        
        results = {}
        
        # Domain-specific evaluation metrics
        if hasattr(self.model, 'classifier'):
            # Classification evaluation
            results['medical_classification'] = self.evaluate_classification(test_data)
        
        # Medical terminology coverage
        results['terminology_coverage'] = self.evaluate_terminology_coverage(test_data)
        
        # Clinical reasoning assessment
        results['clinical_reasoning'] = self.evaluate_clinical_reasoning(test_data)
        
        return results
    
    def evaluate_terminology_coverage(self, texts):
        """Evaluate how well the model handles medical terminology"""
        
        total_medical_terms = 0
        recognized_terms = 0
        
        for text in texts:
            # Find medical terms in text
            for category, terms in self.medical_vocabulary.items():
                for term in terms:
                    if term.lower() in text.lower():
                        total_medical_terms += 1
                        
                        # Check if term is properly tokenized
                        tokens = self.tokenizer.tokenize(term)
                        if len(tokens) <= 2:  # Prefer fewer subword tokens
                            recognized_terms += 1
        
        coverage = recognized_terms / total_medical_terms if total_medical_terms > 0 else 0
        
        return {
            'total_terms': total_medical_terms,
            'recognized_terms': recognized_terms,
            'coverage_ratio': coverage
        }
    
    def evaluate_clinical_reasoning(self, test_cases):
        """Evaluate clinical reasoning capabilities"""
        
        # Sample clinical reasoning test cases
        reasoning_tests = [
            {
                'scenario': "Patient with chest pain, elevated troponin, and ST elevation on ECG",
                'correct_diagnosis': "ST-elevation myocardial infarction",
                'treatment': "Immediate PCI or thrombolytic therapy"
            },
            {
                'scenario': "Diabetic patient with polyuria, polydipsia, and blood glucose > 400 mg/dL",
                'correct_diagnosis': "Diabetic ketoacidosis",
                'treatment': "IV insulin, fluid resuscitation, electrolyte monitoring"
            }
        ]
        
        # Evaluate reasoning (simplified scoring)
        reasoning_score = 0
        for test_case in reasoning_tests:
            # In practice, would use the model to generate reasoning
            # and compare with expert annotations
            reasoning_score += 0.8  # Placeholder score
        
        avg_reasoning_score = reasoning_score / len(reasoning_tests)
        
        return {
            'average_reasoning_score': avg_reasoning_score,
            'total_test_cases': len(reasoning_tests)
        }

# Example usage
def create_medical_model_example():
    """Example of creating a medical domain model"""
    
    # Initialize medical model
    medical_model = MedicalDomainModel()
    
    # Fine-tune for medical classification
    model, tokenizer = medical_model.fine_tune_for_medical_tasks("classification")
    
    # Get sample medical data
    medical_data = medical_model.create_medical_datasets()
    
    # Evaluate performance
    test_texts = medical_data['clinical_notes'][:3]
    results = medical_model.evaluate_medical_performance(test_texts)
    
    print("Medical Model Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
    
    return medical_model, model, tokenizer

# medical_model, model, tokenizer = create_medical_model_example()
```

### Legal Domain Models

```python
class LegalDomainModel:
    """Specialized model for legal domain applications"""
    
    def __init__(self, base_model_name="bert-base-uncased"):
        self.base_model_name = base_model_name
        self.legal_vocabulary = self.load_legal_vocabulary()
        self.citation_patterns = self.load_citation_patterns()
    
    def load_legal_vocabulary(self):
        """Load legal domain vocabulary"""
        
        legal_vocab = {
            'contract_terms': [
                'force majeure', 'indemnification', 'liquidated damages',
                'specific performance', 'breach of contract', 'consideration',
                'covenant', 'warranty', 'representations', 'governing law'
            ],
            'court_terminology': [
                'plaintiff', 'defendant', 'appellant', 'appellee',
                'jurisdiction', 'venue', 'discovery', 'deposition',
                'motion to dismiss', 'summary judgment', 'voir dire'
            ],
            'legal_concepts': [
                'due process', 'probable cause', 'burden of proof',
                'standard of care', 'proximate cause', 'statute of limitations',
                'estoppel', 'res judicata', 'stare decisis', 'prima facie'
            ],
            'statutory_terms': [
                'whereas clause', 'heretofore', 'hereinafter',
                'notwithstanding', 'pursuant to', 'in accordance with',
                'subject to', 'provided that', 'shall not', 'may not'
            ]
        }
        
        return legal_vocab
    
    def load_citation_patterns(self):
        """Load legal citation patterns"""
        
        citation_patterns = [
            r'\d+\s+U\.S\.\s+\d+',  # Supreme Court
            r'\d+\s+F\.\d+d\s+\d+',  # Federal Courts
            r'\d+\s+F\.Supp\.\d*\s+\d+',  # Federal District Courts
            r'\d+\s+S\.Ct\.\s+\d+',  # Supreme Court Reporter
            r'\d+\s+U\.S\.C\.\s+§\s*\d+',  # U.S. Code
            r'\d+\s+C\.F\.R\.\s+§\s*\d+',  # Code of Federal Regulations
        ]
        
        return citation_patterns
    
    def preprocess_legal_text(self, text):
        """Preprocess legal text with domain-specific handling"""
        
        import re
        
        # Normalize legal citations
        for pattern in self.citation_patterns:
            text = re.sub(pattern, '[LEGAL_CITATION]', text)
        
        # Handle legal abbreviations
        legal_abbrevs = {
            'v.': 'versus',
            'et al.': 'et alii',
            'i.e.': 'id est',
            'e.g.': 'exempli gratia',
            'cf.': 'confer',
            'ibid.': 'ibidem'
        }
        
        for abbrev, expansion in legal_abbrevs.items():
            text = text.replace(abbrev, expansion)
        
        return text
    
    def create_legal_datasets(self):
        """Create legal domain datasets"""
        
        legal_data = {
            'contracts': [
                "The parties agree that this Agreement shall be governed by the laws of Delaware, without regard to conflict of law principles.",
                "In the event of material breach, the non-breaching party may seek specific performance or monetary damages.",
                "Force majeure events include acts of God, war, terrorism, and government regulations beyond reasonable control.",
                "All disputes arising under this Agreement shall be resolved through binding arbitration in accordance with AAA rules.",
                "The warranties contained herein shall survive termination and remain in effect for a period of two (2) years."
            ],
            'case_law': [
                "The Court held that plaintiff failed to establish proximate cause between defendant's actions and the alleged harm.",
                "Under the doctrine of stare decisis, this Court is bound by the precedent established in Smith v. Johnson.",
                "The defendant's motion for summary judgment is denied, as material facts remain in dispute.",
                "Plaintiff must demonstrate by a preponderance of evidence that the contract was breached.",
                "The statutory construction requires us to give effect to the plain meaning of the legislative text."
            ],
            'regulations': [
                "Section 501(c)(3) organizations must operate exclusively for charitable, religious, or educational purposes.",
                "The disclosure requirements under Regulation FD apply to all material non-public information.",
                "Pursuant to 15 U.S.C. § 78j(b), it is unlawful to employ manipulative or deceptive devices.",
                "The safe harbor provisions protect forward-looking statements made in good faith.",
                "Compliance with SOX Section 404 requires annual assessment of internal controls over financial reporting."
            ]
        }
        
        return legal_data
    
    def fine_tune_for_legal_tasks(self, task_type="contract_analysis"):
        """Fine-tune model for legal tasks"""
        
        # Create legal-optimized tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Add legal vocabulary
        all_legal_terms = []
        for category, terms in self.legal_vocabulary.items():
            all_legal_terms.extend(terms)
        
        new_tokens = [term for term in all_legal_terms if term not in tokenizer.vocab]
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
        
        if task_type == "contract_analysis":
            # Contract clause classification
            from transformers import AutoModelForSequenceClassification
            
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=5  # e.g., Payment, Termination, Liability, IP, Other
            )
            
            if new_tokens:
                model.resize_token_embeddings(len(tokenizer))
            
            # Create contract analysis dataset
            contract_clauses = [
                ("Payment shall be due within thirty (30) days of invoice date.", 0),  # Payment
                ("Either party may terminate this agreement with sixty (60) days written notice.", 1),  # Termination
                ("Company shall indemnify and hold harmless all claims arising from negligence.", 2),  # Liability
                ("All intellectual property rights shall remain with the creating party.", 3),  # IP
                ("This agreement shall be binding upon successors and assigns.", 4)  # Other
            ]
            
            return model, tokenizer, contract_clauses
        
        elif task_type == "legal_ner":
            # Legal Named Entity Recognition
            from transformers import AutoModelForTokenClassification
            
            legal_ner_labels = [
                'O',  # Outside
                'B-PERSON', 'I-PERSON',  # Person names
                'B-ORG', 'I-ORG',  # Organizations
                'B-CASE', 'I-CASE',  # Case names
                'B-STATUTE', 'I-STATUTE',  # Statutes
                'B-CITATION', 'I-CITATION',  # Legal citations
                'B-DATE', 'I-DATE',  # Dates
                'B-MONEY', 'I-MONEY',  # Monetary amounts
            ]
            
            model = AutoModelForTokenClassification.from_pretrained(
                self.base_model_name,
                num_labels=len(legal_ner_labels)
            )
            
            if new_tokens:
                model.resize_token_embeddings(len(tokenizer))
            
            return model, tokenizer, legal_ner_labels
    
    def evaluate_legal_reasoning(self, test_cases):
        """Evaluate legal reasoning capabilities"""
        
        reasoning_metrics = {
            'precedent_recognition': 0.0,
            'statutory_interpretation': 0.0,
            'contractual_analysis': 0.0,
            'citation_accuracy': 0.0
        }
        
        # Sample legal reasoning tests
        for test_case in test_cases:
            # Evaluate precedent recognition
            if 'precedent' in test_case:
                reasoning_metrics['precedent_recognition'] += 0.8
            
            # Evaluate statutory interpretation
            if 'statute' in test_case:
                reasoning_metrics['statutory_interpretation'] += 0.7
            
            # Evaluate contractual analysis
            if 'contract' in test_case:
                reasoning_metrics['contractual_analysis'] += 0.9
            
            # Evaluate citation accuracy
            if any(re.search(pattern, test_case) for pattern in self.citation_patterns):
                reasoning_metrics['citation_accuracy'] += 0.85
        
        # Normalize by number of test cases
        for metric in reasoning_metrics:
            reasoning_metrics[metric] /= len(test_cases)
        
        return reasoning_metrics

# Example usage
def create_legal_model_example():
    """Example of creating a legal domain model"""
    
    # Initialize legal model
    legal_model = LegalDomainModel()
    
    # Fine-tune for contract analysis
    model, tokenizer, sample_data = legal_model.fine_tune_for_legal_tasks("contract_analysis")
    
    # Get legal datasets
    legal_data = legal_model.create_legal_datasets()
    
    # Evaluate legal reasoning
    test_cases = legal_data['case_law']
    reasoning_results = legal_model.evaluate_legal_reasoning(test_cases)
    
    print("Legal Model Evaluation Results:")
    for metric, score in reasoning_results.items():
        print(f"  {metric}: {score:.2f}")
    
    return legal_model, model, tokenizer

# legal_model, model, tokenizer = create_legal_model_example()
```

## 4. Learning Objectives

By the end of this section, you should be able to:
- **Analyze** domain shift and its impact on model performance
- **Implement** continued pre-training for domain adaptation
- **Create** domain-specific datasets and vocabularies
- **Apply** advanced adaptation strategies (gradual unfreezing, multi-objective training)
- **Build** specialized models for medical, legal, and other domains
- **Evaluate** domain adaptation effectiveness using appropriate metrics

### Self-Assessment Checklist

□ Can identify and quantify domain shift using embedding analysis  
□ Can implement continued pre-training pipeline from scratch  
□ Can expand model vocabulary with domain-specific terms  
□ Can apply gradual unfreezing and advanced training strategies  
□ Can create domain-specific evaluation metrics  
□ Can build end-to-end domain adaptation solutions  
□ Can evaluate trade-offs between general and domain-specific performance  

## 5. Practical Exercises

**Exercise 1: Domain Shift Analysis**
```python
# TODO: Analyze domain shift between general and specialized corpora
# Use embedding visualization and statistical measures
# Compare multiple domains (medical, legal, scientific, financial)
```

**Exercise 2: Medical Domain Adaptation**
```python
# TODO: Implement complete medical domain adaptation pipeline
# Include vocabulary expansion, continued pre-training, and task-specific fine-tuning
# Evaluate on medical NER and classification tasks
```

**Exercise 3: Multi-Domain Model**
```python
# TODO: Create model that maintains performance across multiple domains
# Use domain adaptation techniques to prevent catastrophic forgetting
# Evaluate domain-specific vs. general performance trade-offs
```

## 6. Study Materials

### Essential Papers
- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)
- [Domain-Adaptive Pretraining for Questions Answering](https://arxiv.org/abs/2004.02288)
- [BioBERT: a pre-trained biomedical language representation model](https://arxiv.org/abs/1901.08746)
- [LegalBERT: The Muppets straight out of Law School](https://arxiv.org/abs/2010.02559)

### Domain-Specific Resources
- **Medical**: MIMIC-III, PubMed abstracts, clinical trial data
- **Legal**: Caselaw Access Project, legal contracts, regulatory documents
- **Scientific**: arXiv papers, PubMed Central, scientific literature
- **Financial**: SEC filings, financial reports, economic data

### Tools and Libraries
```bash
pip install transformers datasets torch
pip install scikit-learn matplotlib seaborn
pip install spacy scispacy  # For scientific text processing
pip install legal-nlp  # For legal text processing
```
