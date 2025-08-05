# Evaluation and Benchmarking

*Duration: 3 weeks*

## Overview

Proper evaluation and benchmarking are crucial for understanding model performance, comparing different approaches, and ensuring reliable deployment. This section covers comprehensive evaluation methodologies, standard benchmarks, and best practices for NLP model assessment.

## 1. NLP Evaluation Metrics

### Text Classification Metrics

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from scipy import stats
import torch
import torch.nn.functional as F

class ClassificationEvaluator:
    """Comprehensive evaluation for text classification tasks"""
    
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.metrics_history = []
    
    def compute_basic_metrics(self, y_true, y_pred, average='weighted'):
        """Compute basic classification metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        return metrics
    
    def compute_per_class_metrics(self, y_true, y_pred):
        """Compute per-class metrics"""
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return report
    
    def compute_confusion_matrix(self, y_true, y_pred):
        """Compute and visualize confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return cm, cm_normalized
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True, figsize=(10, 8)):
        """Plot confusion matrix"""
        
        cm, cm_norm = self.compute_confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        
        if normalize:
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title('Normalized Confusion Matrix')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title('Confusion Matrix')
        
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.show()
        
        return cm if not normalize else cm_norm
    
    def compute_roc_auc(self, y_true, y_prob, multiclass='ovr'):
        """Compute ROC AUC for binary or multiclass classification"""
        
        try:
            if y_prob.shape[1] == 2:  # Binary classification
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:  # Multiclass
                auc = roc_auc_score(y_true, y_prob, multi_class=multiclass, average='weighted')
            
            return auc
        except Exception as e:
            print(f"Error computing ROC AUC: {e}")
            return None
    
    def plot_roc_curves(self, y_true, y_prob, figsize=(12, 8)):
        """Plot ROC curves for each class"""
        
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        n_classes = y_prob.shape[1]
        
        # Binarize labels for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=figsize)
        
        # Compute ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            class_name = self.class_names[i] if self.class_names else f'Class {i}'
            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def compute_precision_recall_curves(self, y_true, y_prob):
        """Compute precision-recall curves"""
        
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        n_classes = y_prob.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        pr_curves = {}
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            
            class_name = self.class_names[i] if self.class_names else f'Class {i}'
            pr_curves[class_name] = {
                'precision': precision,
                'recall': recall,
                'avg_precision': avg_precision
            }
        
        return pr_curves
    
    def evaluate_model_comprehensive(self, y_true, y_pred, y_prob=None):
        """Comprehensive model evaluation"""
        
        evaluation_results = {}
        
        # Basic metrics
        evaluation_results['basic_metrics'] = self.compute_basic_metrics(y_true, y_pred)
        
        # Per-class metrics
        evaluation_results['per_class_metrics'] = self.compute_per_class_metrics(y_true, y_pred)
        
        # Confusion matrix
        cm, cm_norm = self.compute_confusion_matrix(y_true, y_pred)
        evaluation_results['confusion_matrix'] = cm
        evaluation_results['confusion_matrix_normalized'] = cm_norm
        
        # ROC AUC if probabilities available
        if y_prob is not None:
            evaluation_results['roc_auc'] = self.compute_roc_auc(y_true, y_prob)
            evaluation_results['precision_recall_curves'] = self.compute_precision_recall_curves(y_true, y_prob)
        
        # Store in history
        self.metrics_history.append(evaluation_results)
        
        return evaluation_results
    
    def compare_models(self, results_list, model_names):
        """Compare multiple models"""
        
        comparison = {}
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
        
        for metric in metrics_to_compare:
            comparison[metric] = {}
            for i, (results, name) in enumerate(zip(results_list, model_names)):
                comparison[metric][name] = results['basic_metrics'][metric]
        
        # Create comparison DataFrame
        import pandas as pd
        df_comparison = pd.DataFrame(comparison).round(4)
        
        return df_comparison

# Example usage
def classification_evaluation_demo():
    """Demonstrate classification evaluation"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3
    
    # True labels
    y_true = np.random.choice(n_classes, n_samples)
    
    # Predicted labels (with some noise)
    y_pred = y_true.copy()
    noise_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    y_pred[noise_indices] = np.random.choice(n_classes, size=len(noise_indices))
    
    # Predicted probabilities
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    # Initialize evaluator
    class_names = ['Positive', 'Neutral', 'Negative']
    evaluator = ClassificationEvaluator(class_names)
    
    # Comprehensive evaluation
    results = evaluator.evaluate_model_comprehensive(y_true, y_pred, y_prob)
    
    print("Classification Evaluation Results:")
    print("=" * 50)
    
    print("\nBasic Metrics:")
    for metric, value in results['basic_metrics'].items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print(f"\nROC AUC: {results['roc_auc']:.4f}")
    
    print("\nPer-class Metrics:")
    for class_name in class_names:
        metrics = results['per_class_metrics'][class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-score: {metrics['f1-score']:.4f}")
    
    return evaluator, results

# evaluator, results = classification_evaluation_demo()
```

### Language Generation Metrics

```python
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
import bert_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math

class GenerationEvaluator:
    """Comprehensive evaluation for text generation tasks"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load model for perplexity calculation
        self.perplexity_model = None
        self.perplexity_tokenizer = None
    
    def load_perplexity_model(self, model_name="gpt2"):
        """Load model for perplexity calculation"""
        
        self.perplexity_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.perplexity_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.perplexity_model.eval()
        
        # Add padding token
        self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token
    
    def compute_bleu_score(self, references, hypotheses, max_n=4):
        """Compute BLEU score"""
        
        # Prepare references (list of lists of tokens for each reference)
        references_tokens = []
        for ref_set in references:
            if isinstance(ref_set, str):
                ref_set = [ref_set]  # Single reference
            ref_tokens = [ref.split() for ref in ref_set]
            references_tokens.append(ref_tokens)
        
        # Prepare hypotheses
        hypotheses_tokens = [hyp.split() for hyp in hypotheses]
        
        # Compute corpus BLEU
        bleu_scores = {}
        for n in range(1, max_n + 1):
            weights = [1.0/n] * n + [0.0] * (4 - n)
            bleu_scores[f'bleu_{n}'] = corpus_bleu(
                references_tokens, hypotheses_tokens, weights=weights
            )
        
        return bleu_scores
    
    def compute_rouge_scores(self, references, hypotheses):
        """Compute ROUGE scores"""
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for ref, hyp in zip(references, hypotheses):
            # Handle multiple references
            if isinstance(ref, list):
                ref = ref[0]  # Use first reference for ROUGE
            
            scores = self.rouge_scorer.score(ref, hyp)
            
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Calculate average scores
        avg_rouge_scores = {}
        for metric, scores in rouge_scores.items():
            avg_rouge_scores[metric] = np.mean(scores)
        
        return avg_rouge_scores
    
    def compute_bert_score(self, references, hypotheses, lang='en'):
        """Compute BERTScore"""
        
        # Flatten references if needed
        if isinstance(references[0], list):
            references = [ref[0] for ref in references]  # Use first reference
        
        try:
            P, R, F1 = bert_score.score(hypotheses, references, lang=lang, verbose=False)
            
            return {
                'bert_precision': P.mean().item(),
                'bert_recall': R.mean().item(),
                'bert_f1': F1.mean().item()
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
    
    def compute_perplexity(self, texts):
        """Compute perplexity of generated texts"""
        
        if self.perplexity_model is None:
            self.load_perplexity_model()
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                # Tokenize text
                inputs = self.perplexity_tokenizer(
                    text, return_tensors='pt', truncation=True, max_length=512
                )
                
                # Forward pass
                outputs = self.perplexity_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Count tokens
                num_tokens = inputs['input_ids'].shape[1]
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def compute_diversity_metrics(self, texts):
        """Compute diversity metrics for generated texts"""
        
        # Distinct-n metrics
        def distinct_n(texts, n):
            all_ngrams = set()
            total_ngrams = 0
            
            for text in texts:
                tokens = text.split()
                ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                all_ngrams.update(ngrams)
                total_ngrams += len(ngrams)
            
            return len(all_ngrams) / total_ngrams if total_ngrams > 0 else 0
        
        diversity_metrics = {
            'distinct_1': distinct_n(texts, 1),
            'distinct_2': distinct_n(texts, 2),
            'distinct_3': distinct_n(texts, 3)
        }
        
        # Vocabulary size
        all_tokens = set()
        total_tokens = 0
        
        for text in texts:
            tokens = text.split()
            all_tokens.update(tokens)
            total_tokens += len(tokens)
        
        diversity_metrics['vocab_size'] = len(all_tokens)
        diversity_metrics['avg_length'] = total_tokens / len(texts)
        
        return diversity_metrics
    
    def compute_semantic_similarity(self, references, hypotheses):
        """Compute semantic similarity using sentence embeddings"""
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load sentence transformer model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Flatten references if needed
            if isinstance(references[0], list):
                references = [ref[0] for ref in references]
            
            # Encode sentences
            ref_embeddings = model.encode(references)
            hyp_embeddings = model.encode(hypotheses)
            
            # Compute cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = []
            
            for ref_emb, hyp_emb in zip(ref_embeddings, hyp_embeddings):
                sim = cosine_similarity([ref_emb], [hyp_emb])[0][0]
                similarities.append(sim)
            
            return {
                'semantic_similarity_mean': np.mean(similarities),
                'semantic_similarity_std': np.std(similarities)
            }
            
        except ImportError:
            print("sentence-transformers not available. Install with: pip install sentence-transformers")
            return {'semantic_similarity_mean': 0.0, 'semantic_similarity_std': 0.0}
    
    def evaluate_generation_comprehensive(self, references, hypotheses):
        """Comprehensive evaluation of text generation"""
        
        evaluation_results = {}
        
        print("Computing BLEU scores...")
        evaluation_results['bleu'] = self.compute_bleu_score(references, hypotheses)
        
        print("Computing ROUGE scores...")
        evaluation_results['rouge'] = self.compute_rouge_scores(references, hypotheses)
        
        print("Computing BERTScore...")
        evaluation_results['bert_score'] = self.compute_bert_score(references, hypotheses)
        
        print("Computing perplexity...")
        evaluation_results['perplexity'] = self.compute_perplexity(hypotheses)
        
        print("Computing diversity metrics...")
        evaluation_results['diversity'] = self.compute_diversity_metrics(hypotheses)
        
        print("Computing semantic similarity...")
        evaluation_results['semantic'] = self.compute_semantic_similarity(references, hypotheses)
        
        return evaluation_results

# Example usage
def generation_evaluation_demo():
    """Demonstrate generation evaluation"""
    
    # Sample references and hypotheses
    references = [
        "The cat sat on the mat and looked around.",
        "She walked to the store to buy some milk.",
        "The weather is beautiful today with clear skies."
    ]
    
    hypotheses = [
        "A cat was sitting on a mat and observing.",
        "She went to the shop to purchase milk.",
        "Today's weather is nice with blue skies."
    ]
    
    # Initialize evaluator
    evaluator = GenerationEvaluator()
    
    # Comprehensive evaluation
    results = evaluator.evaluate_generation_comprehensive(references, hypotheses)
    
    print("\nGeneration Evaluation Results:")
    print("=" * 50)
    
    print("\nBLEU Scores:")
    for metric, score in results['bleu'].items():
        print(f"  {metric.upper()}: {score:.4f}")
    
    print("\nROUGE Scores:")
    for metric, score in results['rouge'].items():
        print(f"  {metric.upper()}: {score:.4f}")
    
    print("\nBERTScore:")
    for metric, score in results['bert_score'].items():
        print(f"  {metric}: {score:.4f}")
    
    print(f"\nPerplexity: {results['perplexity']:.2f}")
    
    print("\nDiversity Metrics:")
    for metric, score in results['diversity'].items():
        print(f"  {metric}: {score:.4f}")
    
    print("\nSemantic Similarity:")
    for metric, score in results['semantic'].items():
        print(f"  {metric}: {score:.4f}")
    
    return evaluator, results

# evaluator, results = generation_evaluation_demo()
```

## 2. Standard Benchmarks and Datasets

### GLUE and SuperGLUE Benchmarks

```python
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader

class BenchmarkEvaluator:
    """Evaluate models on standard NLP benchmarks"""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.benchmark_results = {}
    
    def load_glue_dataset(self, task_name):
        """Load GLUE benchmark dataset"""
        
        glue_tasks = [
            'cola', 'sst2', 'mrpc', 'qqp', 'stsb', 
            'mnli', 'qnli', 'rte', 'wnli'
        ]
        
        if task_name not in glue_tasks:
            raise ValueError(f"Task {task_name} not in GLUE tasks: {glue_tasks}")
        
        # Load dataset
        if task_name == 'mnli':
            dataset = load_dataset('glue', task_name)
            # MNLI has matched and mismatched validation sets
            return dataset
        else:
            dataset = load_dataset('glue', task_name)
            return dataset
    
    def load_superglue_dataset(self, task_name):
        """Load SuperGLUE benchmark dataset"""
        
        superglue_tasks = [
            'boolq', 'cb', 'copa', 'multirc', 'record',
            'rte', 'wic', 'wsc', 'axb', 'axg'
        ]
        
        if task_name not in superglue_tasks:
            raise ValueError(f"Task {task_name} not in SuperGLUE tasks: {superglue_tasks}")
        
        dataset = load_dataset('super_glue', task_name)
        return dataset
    
    def prepare_dataset_for_task(self, dataset, task_name):
        """Prepare dataset for specific task"""
        
        def tokenize_function(examples):
            if task_name in ['cola', 'sst2']:
                # Single sentence tasks
                return self.tokenizer(
                    examples['sentence'],
                    truncation=True,
                    padding='max_length',
                    max_length=128
                )
            elif task_name in ['mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']:
                # Sentence pair tasks
                sentence1_key = 'sentence1' if 'sentence1' in examples else 'question'
                sentence2_key = 'sentence2' if 'sentence2' in examples else 'sentence'
                
                return self.tokenizer(
                    examples[sentence1_key],
                    examples[sentence2_key],
                    truncation=True,
                    padding='max_length',
                    max_length=128
                )
            else:
                # Default handling
                return self.tokenizer(
                    examples['sentence'] if 'sentence' in examples else examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=128
                )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def evaluate_on_benchmark(self, task_name, benchmark_type='glue'):
        """Evaluate model on benchmark task"""
        
        print(f"Evaluating on {benchmark_type.upper()} task: {task_name}")
        
        # Load dataset
        if benchmark_type == 'glue':
            dataset = self.load_glue_dataset(task_name)
        else:
            dataset = self.load_superglue_dataset(task_name)
        
        # Prepare dataset
        tokenized_dataset = self.prepare_dataset_for_task(dataset, task_name)
        
        # Determine task type and number of labels
        task_info = self.get_task_info(task_name, benchmark_type)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=task_info['num_labels']
        )
        
        # Prepare evaluation dataset
        eval_dataset = tokenized_dataset['validation']
        
        # Simple evaluation (without fine-tuning)
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for i in range(min(100, len(eval_dataset))):  # Evaluate on subset
                example = eval_dataset[i]
                
                inputs = {
                    'input_ids': torch.tensor([example['input_ids']]),
                    'attention_mask': torch.tensor([example['attention_mask']])
                }
                
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                predictions.append(pred)
                true_labels.append(example['label'])
        
        # Compute metrics based on task type
        if task_info['task_type'] == 'classification':
            evaluator = ClassificationEvaluator()
            results = evaluator.compute_basic_metrics(true_labels, predictions)
        else:
            # Regression task (like STS-B)
            from sklearn.metrics import mean_squared_error, pearsonr
            mse = mean_squared_error(true_labels, predictions)
            correlation, _ = pearsonr(true_labels, predictions)
            results = {'mse': mse, 'correlation': correlation}
        
        self.benchmark_results[f"{benchmark_type}_{task_name}"] = results
        
        return results
    
    def get_task_info(self, task_name, benchmark_type):
        """Get task information"""
        
        task_configs = {
            'glue': {
                'cola': {'num_labels': 2, 'task_type': 'classification'},
                'sst2': {'num_labels': 2, 'task_type': 'classification'},
                'mrpc': {'num_labels': 2, 'task_type': 'classification'},
                'qqp': {'num_labels': 2, 'task_type': 'classification'},
                'stsb': {'num_labels': 1, 'task_type': 'regression'},
                'mnli': {'num_labels': 3, 'task_type': 'classification'},
                'qnli': {'num_labels': 2, 'task_type': 'classification'},
                'rte': {'num_labels': 2, 'task_type': 'classification'},
                'wnli': {'num_labels': 2, 'task_type': 'classification'},
            },
            'superglue': {
                'boolq': {'num_labels': 2, 'task_type': 'classification'},
                'cb': {'num_labels': 3, 'task_type': 'classification'},
                'copa': {'num_labels': 2, 'task_type': 'classification'},
                'rte': {'num_labels': 2, 'task_type': 'classification'},
                'wic': {'num_labels': 2, 'task_type': 'classification'},
                'wsc': {'num_labels': 2, 'task_type': 'classification'},
            }
        }
        
        return task_configs[benchmark_type].get(task_name, {'num_labels': 2, 'task_type': 'classification'})
    
    def run_benchmark_suite(self, benchmark_type='glue', tasks=None):
        """Run evaluation on multiple benchmark tasks"""
        
        if tasks is None:
            if benchmark_type == 'glue':
                tasks = ['cola', 'sst2', 'mrpc', 'rte']  # Subset for demo
            else:
                tasks = ['boolq', 'cb', 'copa', 'rte']  # Subset for demo
        
        suite_results = {}
        
        for task in tasks:
            try:
                print(f"\n{'='*20} {task.upper()} {'='*20}")
                results = self.evaluate_on_benchmark(task, benchmark_type)
                suite_results[task] = results
                
                print(f"Results for {task}:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.4f}")
                    
            except Exception as e:
                print(f"Error evaluating {task}: {e}")
                suite_results[task] = {'error': str(e)}
        
        return suite_results
    
    def compare_with_baselines(self, task_results, baselines=None):
        """Compare results with baseline models"""
        
        if baselines is None:
            # Example baseline scores (would be actual benchmark results)
            baselines = {
                'random': 0.5,
                'majority_class': 0.6,
                'simple_baseline': 0.7,
                'bert_base': 0.8
            }
        
        comparison = {}
        
        for task, results in task_results.items():
            if 'error' in results:
                continue
                
            main_metric = 'accuracy' if 'accuracy' in results else list(results.keys())[0]
            task_score = results[main_metric]
            
            comparison[task] = {
                'current_model': task_score,
                **baselines
            }
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison).T
        df_comparison = df_comparison.round(4)
        
        return df_comparison

# Example usage
def benchmark_evaluation_demo():
    """Demonstrate benchmark evaluation"""
    
    # Initialize benchmark evaluator
    evaluator = BenchmarkEvaluator("distilbert-base-uncased")
    
    print("Benchmark Evaluation Demo")
    print("=" * 50)
    
    # Run subset of GLUE tasks
    try:
        results = evaluator.run_benchmark_suite('glue', ['cola', 'sst2'])
        
        # Compare with baselines
        comparison = evaluator.compare_with_baselines(results)
        
        print("\n" + "="*50)
        print("COMPARISON WITH BASELINES")
        print("="*50)
        print(comparison)
        
        return evaluator, results, comparison
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This demo requires internet connection to download datasets")
        return None, None, None

# evaluator, results, comparison = benchmark_evaluation_demo()
```

## 3. Evaluation Protocols and Best Practices

### Statistical Significance Testing

```python
import scipy.stats as stats
from scipy.stats import ttest_rel, wilcoxon, bootstrap
import matplotlib.pyplot as plt

class StatisticalEvaluator:
    """Statistical evaluation and significance testing"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha  # Significance level
        self.test_results = {}
    
    def paired_t_test(self, scores1, scores2, model1_name="Model 1", model2_name="Model 2"):
        """Perform paired t-test between two models"""
        
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        # Perform paired t-test
        statistic, p_value = ttest_rel(scores1, scores2)
        
        # Effect size (Cohen's d)
        diff = scores1 - scores2
        cohen_d = np.mean(diff) / np.std(diff, ddof=1)
        
        # Confidence interval for the difference
        n = len(scores1)
        std_err = np.std(diff, ddof=1) / np.sqrt(n)
        t_critical = stats.t.ppf(1 - self.alpha/2, n-1)
        ci_lower = np.mean(diff) - t_critical * std_err
        ci_upper = np.mean(diff) + t_critical * std_err
        
        result = {
            'test_type': 'paired_t_test',
            'model1': model1_name,
            'model2': model2_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'cohen_d': cohen_d,
            'mean_diff': np.mean(diff),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'sample_size': n
        }
        
        return result
    
    def wilcoxon_signed_rank_test(self, scores1, scores2, model1_name="Model 1", model2_name="Model 2"):
        """Perform Wilcoxon signed-rank test (non-parametric)"""
        
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        # Perform Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(scores1, scores2, alternative='two-sided')
        
        result = {
            'test_type': 'wilcoxon_signed_rank',
            'model1': model1_name,
            'model2': model2_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'median_diff': np.median(scores1 - scores2),
            'sample_size': len(scores1)
        }
        
        return result
    
    def bootstrap_confidence_interval(self, scores, metric_func=np.mean, n_bootstrap=1000):
        """Compute bootstrap confidence interval"""
        
        scores = np.array(scores)
        
        # Bootstrap resampling function
        def bootstrap_resample():
            return metric_func(np.random.choice(scores, size=len(scores), replace=True))
        
        # Generate bootstrap samples
        bootstrap_stats = [bootstrap_resample() for _ in range(n_bootstrap)]
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute confidence interval
        ci_lower = np.percentile(bootstrap_stats, (self.alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - self.alpha/2) * 100)
        
        return {
            'point_estimate': metric_func(scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_samples': bootstrap_stats
        }
    
    def multiple_model_comparison(self, model_scores, model_names):
        """Compare multiple models with multiple comparisons correction"""
        
        from itertools import combinations
        from statsmodels.stats.multitest import multipletests
        
        # Perform pairwise tests
        pairwise_results = []
        p_values = []
        
        for i, j in combinations(range(len(model_names)), 2):
            result = self.paired_t_test(
                model_scores[i], model_scores[j],
                model_names[i], model_names[j]
            )
            pairwise_results.append(result)
            p_values.append(result['p_value'])
        
        # Apply multiple comparisons correction
        corrected_p_values = multipletests(p_values, alpha=self.alpha, method='bonferroni')[1]
        
        # Update results with corrected p-values
        for i, result in enumerate(pairwise_results):
            result['corrected_p_value'] = corrected_p_values[i]
            result['significant_corrected'] = corrected_p_values[i] < self.alpha
        
        return pairwise_results
    
    def cross_validation_evaluation(self, model, dataset, cv_folds=5):
        """Perform cross-validation evaluation"""
        
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        X = dataset['features']  # Assuming dataset has features and labels
        y = dataset['labels']
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Evaluating fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model (simplified)
            # model.fit(X_train, y_train)
            
            # Evaluate model (simplified)
            # predictions = model.predict(X_val)
            # score = accuracy_score(y_val, predictions)
            
            # For demo, use random scores
            score = np.random.uniform(0.7, 0.9)
            fold_scores.append(score)
        
        # Compute statistics
        cv_results = {
            'scores': fold_scores,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'ci': self.bootstrap_confidence_interval(fold_scores)
        }
        
        return cv_results
    
    def effect_size_analysis(self, scores1, scores2):
        """Analyze effect sizes between models"""
        
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        # Cohen's d
        pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                             (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                            (len(scores1) + len(scores2) - 2))
        cohen_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
        
        # Interpret effect size
        if abs(cohen_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohen_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohen_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # Glass's delta (uses control group standard deviation)
        glass_delta = (np.mean(scores1) - np.mean(scores2)) / np.std(scores2, ddof=1)
        
        return {
            'cohen_d': cohen_d,
            'glass_delta': glass_delta,
            'effect_interpretation': effect_interpretation,
            'mean_difference': np.mean(scores1) - np.mean(scores2)
        }
    
    def plot_model_comparison(self, model_scores, model_names, figsize=(12, 8)):
        """Plot model comparison with confidence intervals"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot
        ax1.boxplot(model_scores, labels=model_names)
        ax1.set_title('Model Performance Distribution')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        
        # Mean with confidence intervals
        means = [np.mean(scores) for scores in model_scores]
        cis = [self.bootstrap_confidence_interval(scores) for scores in model_scores]
        
        x_pos = range(len(model_names))
        ax2.scatter(x_pos, means, color='red', s=100, zorder=5)
        
        for i, ci in enumerate(cis):
            ax2.plot([i, i], [ci['ci_lower'], ci['ci_upper']], 'b-', linewidth=2)
            ax2.plot([i-0.05, i+0.05], [ci['ci_lower'], ci['ci_lower']], 'b-', linewidth=2)
            ax2.plot([i-0.05, i+0.05], [ci['ci_upper'], ci['ci_upper']], 'b-', linewidth=2)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names)
        ax2.set_title('Model Means with 95% Confidence Intervals')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage
def statistical_evaluation_demo():
    """Demonstrate statistical evaluation"""
    
    # Generate synthetic model scores
    np.random.seed(42)
    
    model_scores = {
        'BERT': np.random.normal(0.85, 0.05, 20),
        'RoBERTa': np.random.normal(0.87, 0.04, 20),
        'DeBERTa': np.random.normal(0.88, 0.045, 20)
    }
    
    evaluator = StatisticalEvaluator()
    
    print("Statistical Evaluation Demo")
    print("=" * 50)
    
    # Paired t-test between BERT and RoBERTa
    t_test_result = evaluator.paired_t_test(
        model_scores['BERT'], model_scores['RoBERTa'],
        'BERT', 'RoBERTa'
    )
    
    print("\nPaired t-test (BERT vs RoBERTa):")
    print(f"  t-statistic: {t_test_result['statistic']:.4f}")
    print(f"  p-value: {t_test_result['p_value']:.4f}")
    print(f"  Significant: {t_test_result['significant']}")
    print(f"  Cohen's d: {t_test_result['cohen_d']:.4f}")
    print(f"  95% CI: [{t_test_result['ci_lower']:.4f}, {t_test_result['ci_upper']:.4f}]")
    
    # Bootstrap confidence intervals
    bert_ci = evaluator.bootstrap_confidence_interval(model_scores['BERT'])
    print(f"\nBERT Bootstrap 95% CI: [{bert_ci['ci_lower']:.4f}, {bert_ci['ci_upper']:.4f}]")
    
    # Effect size analysis
    effect_size = evaluator.effect_size_analysis(
        model_scores['RoBERTa'], model_scores['BERT']
    )
    print(f"\nEffect size (RoBERTa vs BERT):")
    print(f"  Cohen's d: {effect_size['cohen_d']:.4f} ({effect_size['effect_interpretation']})")
    print(f"  Mean difference: {effect_size['mean_difference']:.4f}")
    
    # Multiple model comparison
    scores_list = [model_scores['BERT'], model_scores['RoBERTa'], model_scores['DeBERTa']]
    model_names = ['BERT', 'RoBERTa', 'DeBERTa']
    
    pairwise_results = evaluator.multiple_model_comparison(scores_list, model_names)
    
    print("\nMultiple model comparison (with Bonferroni correction):")
    for result in pairwise_results:
        print(f"  {result['model1']} vs {result['model2']}:")
        print(f"    p-value: {result['p_value']:.4f}")
        print(f"    Corrected p-value: {result['corrected_p_value']:.4f}")
        print(f"    Significant (corrected): {result['significant_corrected']}")
    
    return evaluator, model_scores, pairwise_results

# evaluator, model_scores, results = statistical_evaluation_demo()
```

## 4. Learning Objectives

By the end of this section, you should be able to:
- **Apply** appropriate evaluation metrics for different NLP tasks
- **Implement** comprehensive evaluation pipelines for classification and generation
- **Conduct** statistical significance testing and effect size analysis
- **Evaluate** models on standard benchmarks (GLUE, SuperGLUE)
- **Interpret** evaluation results and make informed comparisons
- **Design** robust evaluation protocols for real-world deployment

### Self-Assessment Checklist

□ Can select and compute appropriate metrics for any NLP task  
□ Can implement BLEU, ROUGE, BERTScore, and perplexity calculations  
□ Can perform statistical significance testing between models  
□ Can evaluate models on standard benchmarks  
□ Can interpret confidence intervals and effect sizes  
□ Can design comprehensive evaluation frameworks  
□ Can identify evaluation pitfalls and limitations  

## 5. Practical Exercises

**Exercise 1: Custom Evaluation Framework**
```python
# TODO: Build a comprehensive evaluation framework
# Support multiple metrics, statistical testing, and visualization
# Apply to real classification and generation tasks
```

**Exercise 2: Benchmark Evaluation Suite**
```python
# TODO: Implement evaluation on GLUE/SuperGLUE tasks
# Compare multiple models with statistical significance testing
# Generate comprehensive evaluation reports
```

**Exercise 3: Generation Quality Assessment**
```python
# TODO: Develop evaluation system for text generation
# Include automatic metrics, human evaluation setup, and bias detection
# Apply to summarization or dialog generation tasks
```

## 6. Study Materials

### Essential Papers
- [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
- [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://arxiv.org/abs/1905.00537)
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040/)

### Evaluation Resources
- **Benchmarks**: GLUE, SuperGLUE, XTREME, BigBench
- **Metrics Libraries**: NLTK, scikit-learn, rouge-score, bert-score
- **Statistical Tools**: scipy.stats, statsmodels
- **Visualization**: matplotlib, seaborn, plotly

### Tools and Libraries
```bash
pip install datasets evaluate transformers
pip install nltk rouge-score bert-score
pip install scipy statsmodels
pip install matplotlib seaborn plotly
```
