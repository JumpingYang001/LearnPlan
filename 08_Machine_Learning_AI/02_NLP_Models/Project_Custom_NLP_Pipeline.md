# Project: Custom NLP Pipeline

*Duration: 4-6 weeks*

## Project Overview

Build a comprehensive, production-ready NLP system tailored for a specific domain (e.g., medical, legal, financial, or e-commerce). This project involves implementing multiple NLP tasks, fine-tuning models, creating robust evaluation metrics, and deploying a scalable pipeline with monitoring capabilities.

## Learning Objectives

By completing this project, you will:
- Design and implement end-to-end NLP pipelines
- Master fine-tuning techniques for domain-specific applications
- Develop comprehensive evaluation and monitoring systems
- Create production-ready NLP services with proper API design
- Understand model versioning and deployment strategies
- Implement data preprocessing and augmentation techniques

## Project Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Preprocessing   │───▶│   Model Store   │
│  • Raw Text     │    │  • Cleaning      │    │  • Base Models  │
│  • Documents    │    │  • Tokenization  │    │  • Fine-tuned   │
│  • APIs         │    │  • Augmentation  │    │  • Versioning   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│   NLP Pipeline   │◀───│   Fine-tuning   │
│  • Performance  │    │  • NER           │    │  • Domain Data  │
│  • Drift        │    │  • Sentiment     │    │  • Evaluation   │
│  • Alerts       │    │  • Classification│    │  • Optimization │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │    API Layer     │
                       │  • REST API      │
                       │  • WebSocket     │
                       │  • Documentation │
                       └──────────────────┘
```

## Implementation Phase 1: Data Pipeline & Preprocessing

### 1.1 Data Collection and Management

```python
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from pathlib import Path
import logging
from typing import List, Dict, Optional
import json

class DataManager:
    """Manages data collection, preprocessing, and versioning"""
    
    def __init__(self, domain: str, data_dir: Path):
        self.domain = domain
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(f'DataManager-{self.domain}')
        return logger
    
    def collect_domain_data(self, sources: Dict[str, str]) -> DatasetDict:
        """Collect data from various domain-specific sources"""
        datasets = {}
        
        for split, source_path in sources.items():
            self.logger.info(f"Loading {split} data from {source_path}")
            
            if source_path.endswith('.json'):
                with open(source_path) as f:
                    data = json.load(f)
            elif source_path.endswith('.csv'):
                data = pd.read_csv(source_path).to_dict('records')
            
            datasets[split] = Dataset.from_list(data)
        
        return DatasetDict(datasets)
    
    def preprocess_text(self, text: str) -> str:
        """Domain-specific text preprocessing"""
        # Remove special characters, normalize whitespace
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Domain-specific preprocessing
        if self.domain == 'medical':
            # Normalize medical abbreviations
            text = self.normalize_medical_terms(text)
        elif self.domain == 'legal':
            # Handle legal citations and references
            text = self.normalize_legal_terms(text)
        
        return text
    
    def normalize_medical_terms(self, text: str) -> str:
        """Normalize medical terminology"""
        medical_abbrev = {
            'mg': 'milligrams',
            'ml': 'milliliters',
            'bp': 'blood pressure',
            'hr': 'heart rate'
        }
        
        for abbrev, full_form in medical_abbrev.items():
            text = re.sub(rf'\b{abbrev}\b', full_form, text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_legal_terms(self, text: str) -> str:
        """Normalize legal terminology"""
        # Example legal normalization
        text = re.sub(r'\bvs?\.\b', 'versus', text, flags=re.IGNORECASE)
        return text

# Example usage
data_manager = DataManager(domain='medical', data_dir=Path('./data'))
sources = {
    'train': './data/medical_train.json',
    'validation': './data/medical_val.json',
    'test': './data/medical_test.json'
}
dataset = data_manager.collect_domain_data(sources)
```

### 1.2 Data Augmentation Techniques

```python
import random
from transformers import pipeline
import nltk
from nltk.corpus import wordnet

class DataAugmentation:
    """Implements various data augmentation techniques"""
    
    def __init__(self):
        self.paraphraser = pipeline('text2text-generation', 
                                  model='tuner007/pegasus_paraphrase')
        nltk.download('wordnet', quiet=True)
    
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """Replace n random words with their synonyms"""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            
            # Get synonyms using WordNet
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name().lower() != word.lower():
                        synonyms.append(lemma.name().replace('_', ' '))
            
            if synonyms:
                words[idx] = random.choice(synonyms)
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert n random synonyms at random positions"""
        words = text.split()
        
        for _ in range(n):
            if not words:
                break
                
            # Get a random word and its synonym
            random_word = random.choice(words)
            synonyms = []
            
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    if lemma.name().lower() != random_word.lower():
                        synonyms.append(lemma.name().replace('_', ' '))
            
            if synonyms:
                synonym = random.choice(synonyms)
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, synonym)
        
        return ' '.join(words)
    
    def back_translation(self, text: str) -> str:
        """Paraphrase using pre-trained model"""
        try:
            result = self.paraphraser(f"paraphrase: {text}")
            return result[0]['generated_text']
        except:
            return text
    
    def augment_dataset(self, dataset: Dataset, 
                       augment_ratio: float = 0.3) -> Dataset:
        """Augment dataset with various techniques"""
        augmented_data = []
        
        for example in dataset:
            # Original example
            augmented_data.append(example)
            
            # Add augmented versions
            if random.random() < augment_ratio:
                text = example['text']
                
                # Apply random augmentation
                aug_choice = random.choice([
                    'synonym', 'insertion', 'paraphrase'
                ])
                
                if aug_choice == 'synonym':
                    aug_text = self.synonym_replacement(text)
                elif aug_choice == 'insertion':
                    aug_text = self.random_insertion(text)
                else:
                    aug_text = self.back_translation(text)
                
                aug_example = example.copy()
                aug_example['text'] = aug_text
                augmented_data.append(aug_example)
        
        return Dataset.from_list(augmented_data)

# Example usage
augmenter = DataAugmentation()
augmented_train = augmenter.augment_dataset(dataset['train'], augment_ratio=0.2)
```

## Implementation Phase 2: Model Fine-tuning

### 2.1 Multi-task Fine-tuning Framework

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, TrainingArguments, Trainer
)
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import wandb

class MultiTaskNLPPipeline:
    """Multi-task NLP pipeline with fine-tuning capabilities"""
    
    def __init__(self, base_model: str, tasks: List[str]):
        self.base_model = base_model
        self.tasks = tasks
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.models = {}
        self.metrics = {}
        
        # Initialize Weights & Biases for experiment tracking
        wandb.init(project="custom-nlp-pipeline")
    
    def setup_models(self):
        """Initialize models for different tasks"""
        for task in self.tasks:
            if task in ['sentiment', 'classification']:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.base_model, num_labels=self._get_num_labels(task)
                )
            elif task == 'ner':
                model = AutoModelForTokenClassification.from_pretrained(
                    self.base_model, num_labels=self._get_ner_labels()
                )
            
            self.models[task] = model
    
    def _get_num_labels(self, task: str) -> int:
        """Get number of labels for classification tasks"""
        label_mapping = {
            'sentiment': 3,  # positive, negative, neutral
            'classification': 5  # domain-specific categories
        }
        return label_mapping.get(task, 2)
    
    def _get_ner_labels(self) -> int:
        """Get number of NER labels"""
        # B-I-O format for domain entities
        return 9  # Example: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
    
    def tokenize_data(self, examples, task: str):
        """Tokenize data for specific task"""
        if task in ['sentiment', 'classification']:
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
        elif task == 'ner':
            # Handle token-level labels for NER
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512,
                is_split_into_words=True,
                return_tensors='pt'
            )
            # Align labels with tokenized words
            tokenized['labels'] = self._align_labels(
                examples['labels'], tokenized
            )
            return tokenized
    
    def _align_labels(self, labels, tokenized_inputs):
        """Align NER labels with tokenized inputs"""
        aligned_labels = []
        for i, label_seq in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_label = []
            
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_label.append(-100)  # Special token
                elif word_idx != previous_word_idx:
                    aligned_label.append(label_seq[word_idx])
                else:
                    aligned_label.append(-100)  # Subword
                previous_word_idx = word_idx
            
            aligned_labels.append(aligned_label)
        
        return torch.tensor(aligned_labels)
    
    def fine_tune_task(self, task: str, train_dataset, eval_dataset, 
                      epochs: int = 3, batch_size: int = 16):
        """Fine-tune model for specific task"""
        
        # Tokenize datasets
        train_tokenized = train_dataset.map(
            lambda x: self.tokenize_data(x, task),
            batched=True
        )
        eval_tokenized = eval_dataset.map(
            lambda x: self.tokenize_data(x, task),
            batched=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/{task}',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs/{task}',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="wandb"
        )
        
        # Trainer
        trainer = Trainer(
            model=self.models[task],
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            compute_metrics=lambda p: self.compute_metrics(p, task)
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(f'./models/{task}')
        
        return trainer
    
    def compute_metrics(self, eval_pred, task: str):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        
        if task in ['sentiment', 'classification']:
            predictions = predictions.argmax(axis=-1)
            
            # Filter out -100 labels
            mask = labels != -100
            predictions = predictions[mask]
            labels = labels[mask]
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted')
            }
        
        elif task == 'ner':
            predictions = predictions.argmax(axis=-1)
            
            # Remove ignored index (special tokens)
            predictions = predictions[labels != -100].flatten()
            labels = labels[labels != -100].flatten()
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted')
            }

# Example usage
pipeline = MultiTaskNLPPipeline(
    base_model='bert-base-uncased',
    tasks=['sentiment', 'ner', 'classification']
)
pipeline.setup_models()

# Fine-tune each task
for task in pipeline.tasks:
    print(f"Fine-tuning {task}...")
    trainer = pipeline.fine_tune_task(
        task=task,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        epochs=3
    )
```

## Implementation Phase 3: Evaluation and Monitoring

### 3.1 Comprehensive Evaluation Framework

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from datetime import datetime

class NLPEvaluator:
    """Comprehensive evaluation framework for NLP models"""
    
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.results = {}
        self.benchmark_results = {}
    
    def evaluate_classification(self, y_true: List, y_pred: List, 
                              labels: List[str]) -> Dict:
        """Evaluate classification performance"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'support': support.tolist()
        }
    
    def evaluate_ner(self, y_true: List[List], y_pred: List[List], 
                     labels: List[str]) -> Dict:
        """Evaluate NER performance at entity level"""
        true_entities = self._extract_entities(y_true, labels)
        pred_entities = self._extract_entities(y_pred, labels)
        
        # Entity-level evaluation
        tp = len(true_entities & pred_entities)
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_entities': len(true_entities),
            'predicted_entities': len(pred_entities),
            'correct_entities': tp
        }
    
    def _extract_entities(self, sequences: List[List], labels: List[str]) -> set:
        """Extract entities from BIO-tagged sequences"""
        entities = set()
        
        for seq_idx, sequence in enumerate(sequences):
            current_entity = None
            entity_start = None
            
            for token_idx, label_idx in enumerate(sequence):
                if label_idx < len(labels):
                    label = labels[label_idx]
                    
                    if label.startswith('B-'):
                        # Start of new entity
                        if current_entity:
                            entities.add((seq_idx, entity_start, token_idx - 1, current_entity))
                        current_entity = label[2:]
                        entity_start = token_idx
                    elif label.startswith('I-') and current_entity:
                        # Continue current entity
                        continue
                    else:
                        # End of entity
                        if current_entity:
                            entities.add((seq_idx, entity_start, token_idx - 1, current_entity))
                        current_entity = None
            
            # Handle entity at end of sequence
            if current_entity:
                entities.add((seq_idx, entity_start, len(sequence) - 1, current_entity))
        
        return entities
    
    def benchmark_speed(self, model, test_data: List[str], 
                       batch_sizes: List[int] = [1, 8, 16, 32]) -> Dict:
        """Benchmark model inference speed"""
        import time
        
        speed_results = {}
        
        for batch_size in batch_sizes:
            times = []
            
            # Create batches
            batches = [test_data[i:i + batch_size] 
                      for i in range(0, len(test_data), batch_size)]
            
            for batch in batches[:10]:  # Test on first 10 batches
                start_time = time.time()
                _ = model(batch)
                end_time = time.time()
                
                batch_time = end_time - start_time
                per_sample_time = batch_time / len(batch)
                times.append(per_sample_time)
            
            speed_results[batch_size] = {
                'mean_time_per_sample': np.mean(times),
                'std_time_per_sample': np.std(times),
                'throughput_samples_per_sec': 1.0 / np.mean(times)
            }
        
        return speed_results
    
    def generate_evaluation_report(self, model_results: Dict, 
                                 output_path: str = './evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_results': model_results,
            'summary': self._generate_summary(model_results)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self._plot_results(model_results)
        
        return report
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics"""
        summary = {}
        
        for task, metrics in results.items():
            if 'f1_score' in metrics:
                summary[task] = {
                    'f1_score': metrics['f1_score'],
                    'accuracy': metrics.get('accuracy', 'N/A'),
                    'status': 'Good' if metrics['f1_score'] > 0.8 else 'Needs Improvement'
                }
        
        return summary
    
    def _plot_results(self, results: Dict):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NLP Pipeline Evaluation Results', fontsize=16)
        
        # F1 scores by task
        tasks = list(results.keys())
        f1_scores = [results[task].get('f1_score', 0) for task in tasks]
        
        axes[0, 0].bar(tasks, f1_scores)
        axes[0, 0].set_title('F1 Scores by Task')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confusion matrix (if available)
        if 'classification' in results and 'confusion_matrix' in results['classification']:
            cm = np.array(results['classification']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1])
            axes[0, 1].set_title('Confusion Matrix - Classification')
        
        # Speed benchmarks (if available)
        if 'speed_benchmark' in results:
            speed_data = results['speed_benchmark']
            batch_sizes = list(speed_data.keys())
            throughputs = [speed_data[bs]['throughput_samples_per_sec'] for bs in batch_sizes]
            
            axes[1, 0].plot(batch_sizes, throughputs, marker='o')
            axes[1, 0].set_title('Throughput vs Batch Size')
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('Samples/Second')
        
        plt.tight_layout()
        plt.savefig('./evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
evaluator = NLPEvaluator(['sentiment', 'ner', 'classification'])

# Evaluate models
model_results = {}
for task in pipeline.tasks:
    # Get predictions (pseudo-code)
    y_true, y_pred = get_test_predictions(pipeline.models[task], test_dataset)
    
    if task in ['sentiment', 'classification']:
        results = evaluator.evaluate_classification(y_true, y_pred, class_labels)
    elif task == 'ner':
        results = evaluator.evaluate_ner(y_true, y_pred, ner_labels)
    
    model_results[task] = results

# Generate report
report = evaluator.generate_evaluation_report(model_results)
```

### 3.2 Production Monitoring System

```python
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any
import json
import asyncio
from dataclasses import dataclass
import aiohttp

@dataclass
class ModelMetrics:
    """Data class for model performance metrics"""
    task: str
    latency: float
    throughput: float
    accuracy: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime

class ProductionMonitor:
    """Production monitoring system for NLP pipeline"""
    
    def __init__(self, alert_thresholds: Dict[str, float]):
        self.alert_thresholds = alert_thresholds
        self.metrics_history = []
        self.alerts = []
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline_monitor.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('ProductionMonitor')
    
    async def monitor_model_performance(self, model, test_batch: List[str]) -> ModelMetrics:
        """Monitor single model performance"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()
        
        # Run inference
        try:
            results = await self._async_inference(model, test_batch)
            success_rate = 1.0
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            success_rate = 0.0
            results = []
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        # Calculate metrics
        latency = end_time - start_time
        throughput = len(test_batch) / latency if latency > 0 else 0
        memory_usage = end_memory - start_memory
        cpu_usage = end_cpu - start_cpu
        
        metrics = ModelMetrics(
            task=model.task if hasattr(model, 'task') else 'unknown',
            latency=latency,
            throughput=throughput,
            accuracy=success_rate,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            timestamp=datetime.now()
        )
        
        # Check for alerts
        await self._check_alerts(metrics)
        
        return metrics
    
    async def _async_inference(self, model, batch: List[str]):
        """Async wrapper for model inference"""
        # This would be implemented based on your specific model interface
        return model(batch)
    
    async def _check_alerts(self, metrics: ModelMetrics):
        """Check if metrics exceed alert thresholds"""
        alerts = []
        
        if metrics.latency > self.alert_thresholds.get('latency', 5.0):
            alerts.append(f"High latency: {metrics.latency:.2f}s")
        
        if metrics.throughput < self.alert_thresholds.get('min_throughput', 10):
            alerts.append(f"Low throughput: {metrics.throughput:.2f} samples/s")
        
        if metrics.memory_usage > self.alert_thresholds.get('memory', 1000):
            alerts.append(f"High memory usage: {metrics.memory_usage:.2f}MB")
        
        if metrics.accuracy < self.alert_thresholds.get('min_accuracy', 0.8):
            alerts.append(f"Low accuracy: {metrics.accuracy:.2f}")
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert, metrics)
    
    async def _send_alert(self, message: str, metrics: ModelMetrics):
        """Send alert notifications"""
        alert_data = {
            'timestamp': metrics.timestamp.isoformat(),
            'task': metrics.task,
            'message': message,
            'metrics': {
                'latency': metrics.latency,
                'throughput': metrics.throughput,
                'memory_usage': metrics.memory_usage,
                'accuracy': metrics.accuracy
            }
        }
        
        # Log alert
        self.logger.warning(f"ALERT: {message}")
        self.alerts.append(alert_data)
        
        # Send to external monitoring service (e.g., Slack, PagerDuty)
        await self._send_external_alert(alert_data)
    
    async def _send_external_alert(self, alert_data: Dict):
        """Send alert to external service"""
        webhook_url = "YOUR_SLACK_WEBHOOK_URL"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json={
                    'text': f"🚨 NLP Pipeline Alert: {alert_data['message']}"
                }) as response:
                    if response.status == 200:
                        self.logger.info("Alert sent successfully")
                    else:
                        self.logger.error("Failed to send alert")
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    def generate_monitoring_report(self, hours: int = 24) -> Dict:
        """Generate monitoring report for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No metrics found for specified period"}
        
        # Aggregate metrics
        avg_latency = sum(m.latency for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_accuracy = sum(m.accuracy for m in recent_metrics) / len(recent_metrics)
        
        report = {
            'period_hours': hours,
            'total_requests': len(recent_metrics),
            'average_metrics': {
                'latency': avg_latency,
                'throughput': avg_throughput,
                'accuracy': avg_accuracy
            },
            'alerts_count': len([a for a in self.alerts 
                               if datetime.fromisoformat(a['timestamp']) >= cutoff_time]),
            'uptime_percentage': self._calculate_uptime(recent_metrics)
        }
        
        return report
    
    def _calculate_uptime(self, metrics: List[ModelMetrics]) -> float:
        """Calculate system uptime percentage"""
        successful_requests = sum(1 for m in metrics if m.accuracy > 0)
        return (successful_requests / len(metrics)) * 100 if metrics else 0

# Example usage
monitor = ProductionMonitor(alert_thresholds={
    'latency': 2.0,
    'min_throughput': 20,
    'memory': 500,
    'min_accuracy': 0.85
})

# Monitor all models in pipeline
async def monitor_pipeline():
    test_batch = ["Sample text for monitoring"] * 10
    
    for task, model in pipeline.models.items():
        metrics = await monitor.monitor_model_performance(model, test_batch)
        monitor.metrics_history.append(metrics)
        print(f"Task: {task}, Latency: {metrics.latency:.2f}s, "
              f"Throughput: {metrics.throughput:.2f} samples/s")

# Run monitoring
# asyncio.run(monitor_pipeline())
```

## API Layer Implementation

### REST API with FastAPI

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime
import asyncio

app = FastAPI(
    title="Custom NLP Pipeline API",
    description="Production-ready NLP pipeline with multiple tasks",
    version="1.0.0"
)

class TextInput(BaseModel):
    text: str
    task: Optional[str] = "classification"

class BatchTextInput(BaseModel):
    texts: List[str]
    task: Optional[str] = "classification"

class PredictionResponse(BaseModel):
    task: str
    predictions: List[Dict]
    processing_time: float
    timestamp: str

# Global pipeline instance
nlp_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global nlp_pipeline
    nlp_pipeline = MultiTaskNLPPipeline(
        base_model='bert-base-uncased',
        tasks=['sentiment', 'ner', 'classification']
    )
    nlp_pipeline.setup_models()
    print("NLP Pipeline initialized successfully")

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: TextInput, background_tasks: BackgroundTasks):
    """Single text prediction endpoint"""
    start_time = time.time()
    
    try:
        if input_data.task not in nlp_pipeline.tasks:
            raise HTTPException(
                status_code=400, 
                detail=f"Task {input_data.task} not supported. Available: {nlp_pipeline.tasks}"
            )
        
        # Get model for the task
        model = nlp_pipeline.models[input_data.task]
        
        # Make prediction
        prediction = await _make_prediction(model, [input_data.text], input_data.task)
        
        processing_time = time.time() - start_time
        
        # Log metrics in background
        background_tasks.add_task(
            _log_prediction_metrics, 
            input_data.task, processing_time, len(input_data.text)
        )
        
        return PredictionResponse(
            task=input_data.task,
            predictions=prediction,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=PredictionResponse)
async def predict_batch(input_data: BatchTextInput, background_tasks: BackgroundTasks):
    """Batch text prediction endpoint"""
    start_time = time.time()
    
    try:
        if input_data.task not in nlp_pipeline.tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Task {input_data.task} not supported. Available: {nlp_pipeline.tasks}"
            )
        
        # Get model for the task
        model = nlp_pipeline.models[input_data.task]
        
        # Make predictions
        predictions = await _make_prediction(model, input_data.texts, input_data.task)
        
        processing_time = time.time() - start_time
        
        # Log metrics in background
        background_tasks.add_task(
            _log_prediction_metrics,
            input_data.task, processing_time, len(input_data.texts)
        )
        
        return PredictionResponse(
            task=input_data.task,
            predictions=predictions,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_tasks": nlp_pipeline.tasks if nlp_pipeline else []
    }

@app.get("/metrics")
async def get_metrics():
    """Get current system metrics"""
    if not hasattr(app.state, 'monitor'):
        return {"error": "Monitoring not initialized"}
    
    return monitor.generate_monitoring_report(hours=1)

async def _make_prediction(model, texts: List[str], task: str) -> List[Dict]:
    """Make prediction using the appropriate model"""
    # This would be implemented based on your specific model interface
    # For demonstration, returning mock predictions
    
    predictions = []
    for text in texts:
        if task == 'sentiment':
            pred = {'text': text, 'sentiment': 'positive', 'confidence': 0.95}
        elif task == 'ner':
            pred = {'text': text, 'entities': [{'text': 'example', 'label': 'ORG'}]}
        else:
            pred = {'text': text, 'category': 'example', 'confidence': 0.90}
        
        predictions.append(pred)
    
    return predictions

async def _log_prediction_metrics(task: str, processing_time: float, batch_size: int):
    """Log prediction metrics for monitoring"""
    # Implementation would log to your monitoring system
    print(f"Logged metrics - Task: {task}, Time: {processing_time:.3f}s, Batch: {batch_size}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Deployment and Production Considerations

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Requirements File

```txt
# requirements.txt
torch>=1.9.0
transformers>=4.10.0
fastapi>=0.70.0
uvicorn>=0.15.0
datasets>=1.12.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
wandb>=0.12.0
aiohttp>=3.8.0
psutil>=5.8.0
nltk>=3.6.0
```

## Project Deliverables

### 1. Code Repository Structure
```
custom-nlp-pipeline/
├── src/
│   ├── data/
│   │   ├── data_manager.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── pipeline.py
│   │   └── fine_tuning.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── monitoring.py
│   └── api/
│       ├── main.py
│       └── models.py
├── tests/
├── docs/
├── docker/
├── requirements.txt
└── README.md
```

### 2. Documentation
- API documentation with Swagger/OpenAPI
- Model training and evaluation reports
- Deployment and scaling guides
- Performance benchmarking results

### 3. Evaluation Reports
- Comprehensive model performance analysis
- A/B testing results comparing different approaches
- Resource usage and cost analysis
- Production monitoring dashboard

This comprehensive project will give you hands-on experience with end-to-end NLP system development, from data preprocessing to production deployment, with robust monitoring and evaluation capabilities.
