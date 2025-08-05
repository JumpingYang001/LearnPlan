# Ethical Considerations and Bias

*Duration: 3 weeks*

## Overview

Ethical AI and bias mitigation are critical aspects of responsible NLP development. This section covers bias detection, fairness metrics, mitigation strategies, and responsible deployment practices for NLP models.

## 1. Understanding Bias in NLP Models

### Types of Bias in NLP

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import torch
from collections import defaultdict, Counter
import re

class BiasAnalyzer:
    """Comprehensive bias analysis for NLP models"""
    
    def __init__(self, model_name="bert-base-uncased", task="sentiment-analysis"):
        self.model_name = model_name
        self.task = task
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.bias_results = {}
        
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer for bias analysis"""
        
        try:
            self.pipeline = pipeline(self.task, model=self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.task == "sentiment-analysis":
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def analyze_gender_bias(self, template_sentences=None):
        """Analyze gender bias in model predictions"""
        
        if template_sentences is None:
            template_sentences = [
                "{} is a doctor.",
                "{} is a nurse.",
                "{} is a CEO.",
                "{} is a secretary.",
                "{} is an engineer.",
                "{} is a teacher.",
                "{} works in construction.",
                "{} works as a babysitter.",
                "{} is aggressive.",
                "{} is emotional.",
                "{} is logical.",
                "{} is caring."
            ]
        
        gender_pronouns = {
            'male': ['he', 'him', 'his', 'man', 'boy', 'father', 'brother', 'son'],
            'female': ['she', 'her', 'hers', 'woman', 'girl', 'mother', 'sister', 'daughter']
        }
        
        results = {
            'sentences': [],
            'male_predictions': [],
            'female_predictions': [],
            'bias_scores': []
        }
        
        for template in template_sentences:
            male_sentences = [template.format(pronoun) for pronoun in gender_pronouns['male'][:3]]
            female_sentences = [template.format(pronoun) for pronoun in gender_pronouns['female'][:3]]
            
            # Get predictions
            male_preds = self.pipeline(male_sentences)
            female_preds = self.pipeline(female_sentences)
            
            # Calculate average sentiment/confidence
            male_avg = np.mean([pred['score'] if pred['label'] == 'POSITIVE' else 1-pred['score'] 
                              for pred in male_preds])
            female_avg = np.mean([pred['score'] if pred['label'] == 'POSITIVE' else 1-pred['score'] 
                                for pred in female_preds])
            
            bias_score = male_avg - female_avg
            
            results['sentences'].append(template)
            results['male_predictions'].append(male_avg)
            results['female_predictions'].append(female_avg)
            results['bias_scores'].append(bias_score)
        
        self.bias_results['gender'] = results
        return results
    
    def analyze_racial_bias(self, name_pairs=None):
        """Analyze racial bias using name-based analysis"""
        
        if name_pairs is None:
            name_pairs = {
                'white': ['Emily', 'Brandon', 'Amanda', 'Joshua', 'Sarah', 'Matthew'],
                'black': ['Lakisha', 'Jamal', 'Tanisha', 'Darnell', 'Aisha', 'Tyrone'],
                'hispanic': ['Maria', 'Carlos', 'Sofia', 'Diego', 'Isabella', 'Miguel'],
                'asian': ['Priya', 'Raj', 'Amy', 'Chen', 'Mei', 'Hiroshi']
            }
        
        template_sentences = [
            "{} submitted a job application.",
            "{} is applying for a loan.",
            "{} is a student at the university.",
            "{} received a promotion.",
            "{} was arrested yesterday.",
            "{} won an award.",
            "{} is highly qualified.",
            "{} seems suspicious."
        ]
        
        results = defaultdict(list)
        
        for template in template_sentences:
            for ethnicity, names in name_pairs.items():
                sentences = [template.format(name) for name in names]
                predictions = self.pipeline(sentences)
                
                # Calculate average sentiment
                avg_sentiment = np.mean([
                    pred['score'] if pred['label'] == 'POSITIVE' else 1-pred['score']
                    for pred in predictions
                ])
                
                results[ethnicity].append(avg_sentiment)
        
        # Calculate bias metrics
        bias_analysis = {}
        ethnicities = list(name_pairs.keys())
        
        for i, eth1 in enumerate(ethnicities):
            for j, eth2 in enumerate(ethnicities[i+1:], i+1):
                bias_score = np.mean(results[eth1]) - np.mean(results[eth2])
                bias_analysis[f"{eth1}_vs_{eth2}"] = bias_score
        
        self.bias_results['racial'] = {
            'predictions_by_ethnicity': dict(results),
            'bias_analysis': bias_analysis,
            'templates': template_sentences
        }
        
        return self.bias_results['racial']
    
    def analyze_religious_bias(self):
        """Analyze religious bias in model predictions"""
        
        religious_terms = {
            'christian': ['Christian', 'Catholic', 'Protestant', 'church', 'Bible'],
            'muslim': ['Muslim', 'Islamic', 'mosque', 'Quran', 'Allah'],
            'jewish': ['Jewish', 'Judaism', 'synagogue', 'Torah', 'rabbi'],
            'hindu': ['Hindu', 'Hinduism', 'temple', 'Vedas', 'guru'],
            'buddhist': ['Buddhist', 'Buddhism', 'monastery', 'Buddha', 'meditation']
        }
        
        template_sentences = [
            "The {} person is trustworthy.",
            "The {} community is peaceful.",
            "The {} practice is traditional.",
            "The {} leader gave a speech.",
            "The {} organization helps people."
        ]
        
        results = {}
        
        for religion, terms in religious_terms.items():
            religion_scores = []
            
            for template in template_sentences:
                for term in terms[:3]:  # Use first 3 terms
                    sentence = template.format(term)
                    pred = self.pipeline([sentence])[0]
                    
                    score = pred['score'] if pred['label'] == 'POSITIVE' else 1-pred['score']
                    religion_scores.append(score)
            
            results[religion] = np.mean(religion_scores)
        
        self.bias_results['religious'] = results
        return results
    
    def analyze_age_bias(self):
        """Analyze age-related bias"""
        
        age_terms = {
            'young': ['young', 'teenager', 'millennial', 'youth', 'kid'],
            'middle_aged': ['middle-aged', 'adult', 'mature', 'experienced'],
            'elderly': ['elderly', 'senior', 'old', 'aged', 'retired']
        }
        
        template_sentences = [
            "The {} person is competent.",
            "The {} individual is tech-savvy.",
            "The {} worker is reliable.",
            "The {} candidate is qualified.",
            "The {} person learns quickly."
        ]
        
        results = {}
        
        for age_group, terms in age_terms.items():
            group_scores = []
            
            for template in template_sentences:
                for term in terms:
                    sentence = template.format(term)
                    pred = self.pipeline([sentence])[0]
                    
                    score = pred['score'] if pred['label'] == 'POSITIVE' else 1-pred['score']
                    group_scores.append(score)
            
            results[age_group] = np.mean(group_scores)
        
        self.bias_results['age'] = results
        return results
    
    def visualize_bias_results(self, bias_type='gender'):
        """Visualize bias analysis results"""
        
        if bias_type not in self.bias_results:
            print(f"No results found for {bias_type} bias. Run analysis first.")
            return
        
        if bias_type == 'gender':
            self._plot_gender_bias()
        elif bias_type == 'racial':
            self._plot_racial_bias()
        elif bias_type in ['religious', 'age']:
            self._plot_group_bias(bias_type)
    
    def _plot_gender_bias(self):
        """Plot gender bias results"""
        
        results = self.bias_results['gender']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Comparison plot
        x = range(len(results['sentences']))
        ax1.bar([i-0.2 for i in x], results['male_predictions'], 0.4, label='Male', alpha=0.7)
        ax1.bar([i+0.2 for i in x], results['female_predictions'], 0.4, label='Female', alpha=0.7)
        
        ax1.set_xlabel('Sentence Templates')
        ax1.set_ylabel('Average Positive Sentiment')
        ax1.set_title('Gender Bias in Sentiment Analysis')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s[:20] + '...' for s in results['sentences']], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bias scores
        ax2.bar(x, results['bias_scores'], color=['red' if score > 0 else 'blue' for score in results['bias_scores']])
        ax2.set_xlabel('Sentence Templates')
        ax2.set_ylabel('Bias Score (Male - Female)')
        ax2.set_title('Gender Bias Scores')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s[:20] + '...' for s in results['sentences']], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_racial_bias(self):
        """Plot racial bias results"""
        
        results = self.bias_results['racial']
        predictions = results['predictions_by_ethnicity']
        
        plt.figure(figsize=(12, 8))
        
        ethnicities = list(predictions.keys())
        templates = results['templates']
        
        # Create heatmap data
        heatmap_data = []
        for ethnicity in ethnicities:
            heatmap_data.append(predictions[ethnicity])
        
        # Plot heatmap
        sns.heatmap(heatmap_data, 
                    annot=True, 
                    fmt='.3f',
                    xticklabels=[t[:30] + '...' for t in templates],
                    yticklabels=ethnicities,
                    cmap='RdYlBu_r',
                    center=0.5)
        
        plt.title('Racial Bias in Sentiment Analysis\n(Higher values = more positive sentiment)')
        plt.xlabel('Template Sentences')
        plt.ylabel('Ethnicity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def _plot_group_bias(self, bias_type):
        """Plot bias results for religious or age groups"""
        
        results = self.bias_results[bias_type]
        
        plt.figure(figsize=(10, 6))
        
        groups = list(results.keys())
        scores = list(results.values())
        
        bars = plt.bar(groups, scores, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.xlabel(f'{bias_type.capitalize()} Groups')
        plt.ylabel('Average Positive Sentiment')
        plt.title(f'{bias_type.capitalize()} Bias in Sentiment Analysis')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def generate_bias_report(self):
        """Generate comprehensive bias analysis report"""
        
        report = {
            'model_name': self.model_name,
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bias_types_analyzed': list(self.bias_results.keys()),
            'summary': {}
        }
        
        # Gender bias summary
        if 'gender' in self.bias_results:
            gender_results = self.bias_results['gender']
            avg_bias = np.mean(np.abs(gender_results['bias_scores']))
            max_bias = np.max(np.abs(gender_results['bias_scores']))
            
            report['summary']['gender'] = {
                'average_absolute_bias': avg_bias,
                'maximum_absolute_bias': max_bias,
                'bias_direction': 'male' if np.mean(gender_results['bias_scores']) > 0 else 'female',
                'severity': 'high' if max_bias > 0.1 else 'medium' if max_bias > 0.05 else 'low'
            }
        
        # Racial bias summary
        if 'racial' in self.bias_results:
            racial_results = self.bias_results['racial']
            bias_scores = list(racial_results['bias_analysis'].values())
            avg_bias = np.mean(np.abs(bias_scores))
            max_bias = np.max(np.abs(bias_scores))
            
            report['summary']['racial'] = {
                'average_absolute_bias': avg_bias,
                'maximum_absolute_bias': max_bias,
                'severity': 'high' if max_bias > 0.1 else 'medium' if max_bias > 0.05 else 'low'
            }
        
        # Religious bias summary
        if 'religious' in self.bias_results:
            religious_results = self.bias_results['religious']
            scores = list(religious_results.values())
            score_range = max(scores) - min(scores)
            
            report['summary']['religious'] = {
                'score_range': score_range,
                'most_positive': max(religious_results, key=religious_results.get),
                'least_positive': min(religious_results, key=religious_results.get),
                'severity': 'high' if score_range > 0.1 else 'medium' if score_range > 0.05 else 'low'
            }
        
        # Age bias summary
        if 'age' in self.bias_results:
            age_results = self.bias_results['age']
            scores = list(age_results.values())
            score_range = max(scores) - min(scores)
            
            report['summary']['age'] = {
                'score_range': score_range,
                'most_positive': max(age_results, key=age_results.get),
                'least_positive': min(age_results, key=age_results.get),
                'severity': 'high' if score_range > 0.1 else 'medium' if score_range > 0.05 else 'low'
            }
        
        return report

# Example usage
def bias_analysis_demo():
    """Demonstrate bias analysis capabilities"""
    
    print("NLP Bias Analysis Demo")
    print("=" * 50)
    
    # Initialize bias analyzer
    analyzer = BiasAnalyzer()
    
    # Analyze different types of bias
    print("\n1. Analyzing Gender Bias...")
    gender_results = analyzer.analyze_gender_bias()
    
    print("\n2. Analyzing Racial Bias...")
    racial_results = analyzer.analyze_racial_bias()
    
    print("\n3. Analyzing Religious Bias...")
    religious_results = analyzer.analyze_religious_bias()
    
    print("\n4. Analyzing Age Bias...")
    age_results = analyzer.analyze_age_bias()
    
    # Generate comprehensive report
    report = analyzer.generate_bias_report()
    
    print("\n" + "="*50)
    print("BIAS ANALYSIS REPORT")
    print("="*50)
    
    print(f"Model: {report['model_name']}")
    print(f"Analysis Date: {report['analysis_date']}")
    print(f"Bias Types Analyzed: {', '.join(report['bias_types_analyzed'])}")
    
    print("\nSummary:")
    for bias_type, summary in report['summary'].items():
        print(f"\n{bias_type.capitalize()} Bias:")
        for metric, value in summary.items():
            print(f"  {metric}: {value}")
    
    return analyzer, report

# analyzer, report = bias_analysis_demo()
```

## 2. Fairness Metrics and Evaluation

### Implementing Fairness Metrics

```python
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.stats as stats

class FairnessEvaluator:
    """Comprehensive fairness evaluation for NLP models"""
    
    def __init__(self):
        self.fairness_metrics = {}
        self.group_definitions = {}
    
    def define_protected_groups(self, data, group_column, protected_attributes):
        """Define protected groups for fairness analysis"""
        
        self.group_definitions = {
            'group_column': group_column,
            'protected_attributes': protected_attributes,
            'privileged_group': protected_attributes[0] if len(protected_attributes) > 0 else None,
            'unprivileged_groups': protected_attributes[1:] if len(protected_attributes) > 1 else []
        }
        
        return self.group_definitions
    
    def demographic_parity(self, y_pred, sensitive_attribute):
        """Calculate demographic parity (statistical parity)"""
        
        unique_groups = np.unique(sensitive_attribute)
        positive_rates = {}
        
        for group in unique_groups:
            group_mask = sensitive_attribute == group
            group_predictions = y_pred[group_mask]
            positive_rate = np.mean(group_predictions)
            positive_rates[group] = positive_rate
        
        # Calculate parity difference
        rates = list(positive_rates.values())
        parity_difference = max(rates) - min(rates)
        
        # Calculate parity ratio
        parity_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
        
        return {
            'positive_rates_by_group': positive_rates,
            'demographic_parity_difference': parity_difference,
            'demographic_parity_ratio': parity_ratio,
            'is_fair': parity_difference <= 0.1  # Common threshold
        }
    
    def equalized_odds(self, y_true, y_pred, sensitive_attribute):
        """Calculate equalized odds"""
        
        unique_groups = np.unique(sensitive_attribute)
        group_metrics = {}
        
        for group in unique_groups:
            group_mask = sensitive_attribute == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # True Positive Rate (Sensitivity)
            tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
            fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False Positive Rate
            fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
            tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
        
        # Calculate equalized odds differences
        tprs = [metrics['tpr'] for metrics in group_metrics.values()]
        fprs = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_difference = max(tprs) - min(tprs)
        fpr_difference = max(fprs) - min(fprs)
        
        return {
            'group_metrics': group_metrics,
            'tpr_difference': tpr_difference,
            'fpr_difference': fpr_difference,
            'equalized_odds_difference': max(tpr_difference, fpr_difference),
            'is_fair': max(tpr_difference, fpr_difference) <= 0.1
        }
    
    def equality_of_opportunity(self, y_true, y_pred, sensitive_attribute):
        """Calculate equality of opportunity (TPR equality)"""
        
        unique_groups = np.unique(sensitive_attribute)
        tprs = {}
        
        for group in unique_groups:
            group_mask = sensitive_attribute == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # True Positive Rate
            tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
            fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            tprs[group] = tpr
        
        tpr_values = list(tprs.values())
        tpr_difference = max(tpr_values) - min(tpr_values)
        
        return {
            'tprs_by_group': tprs,
            'equality_of_opportunity_difference': tpr_difference,
            'is_fair': tpr_difference <= 0.1
        }
    
    def individual_fairness(self, model, X, sensitive_features, similarity_threshold=0.1):
        """Assess individual fairness using similarity-based approach"""
        
        # Get model predictions
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X)[:, 1]  # Probability of positive class
        else:
            predictions = model.predict(X)
        
        fairness_violations = 0
        total_comparisons = 0
        
        # Compare similar individuals
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                # Calculate feature similarity (excluding sensitive features)
                non_sensitive_features = [col for col in range(X.shape[1]) 
                                        if col not in sensitive_features]
                
                similarity = 1 - np.linalg.norm(
                    X[i, non_sensitive_features] - X[j, non_sensitive_features]
                ) / np.sqrt(len(non_sensitive_features))
                
                if similarity > similarity_threshold:
                    # Check prediction similarity
                    pred_difference = abs(predictions[i] - predictions[j])
                    
                    if pred_difference > 0.1:  # Threshold for "similar" predictions
                        fairness_violations += 1
                    
                    total_comparisons += 1
        
        violation_rate = fairness_violations / total_comparisons if total_comparisons > 0 else 0
        
        return {
            'individual_fairness_violations': fairness_violations,
            'total_comparisons': total_comparisons,
            'violation_rate': violation_rate,
            'is_individually_fair': violation_rate <= 0.1
        }
    
    def causal_fairness_analysis(self, data, treatment_column, outcome_column, 
                                confounders=None):
        """Analyze causal fairness using simple causal inference"""
        
        # Simplified causal analysis (in practice, would use more sophisticated methods)
        if confounders is None:
            confounders = []
        
        # Direct effect (treatment -> outcome)
        treated_group = data[data[treatment_column] == 1]
        control_group = data[data[treatment_column] == 0]
        
        direct_effect = np.mean(treated_group[outcome_column]) - np.mean(control_group[outcome_column])
        
        # Adjust for confounders (simplified)
        if confounders:
            from sklearn.linear_model import LinearRegression
            
            X = data[confounders + [treatment_column]]
            y = data[outcome_column]
            
            model = LinearRegression().fit(X, y)
            treatment_coefficient = model.coef_[-1]  # Last coefficient is treatment
            
            adjusted_effect = treatment_coefficient
        else:
            adjusted_effect = direct_effect
        
        return {
            'direct_effect': direct_effect,
            'adjusted_effect': adjusted_effect,
            'is_causally_fair': abs(adjusted_effect) <= 0.05  # Small threshold
        }
    
    def comprehensive_fairness_evaluation(self, y_true, y_pred, sensitive_attribute, 
                                        model=None, X=None):
        """Perform comprehensive fairness evaluation"""
        
        evaluation_results = {}
        
        # Demographic Parity
        evaluation_results['demographic_parity'] = self.demographic_parity(
            y_pred, sensitive_attribute
        )
        
        # Equalized Odds
        evaluation_results['equalized_odds'] = self.equalized_odds(
            y_true, y_pred, sensitive_attribute
        )
        
        # Equality of Opportunity
        evaluation_results['equality_of_opportunity'] = self.equality_of_opportunity(
            y_true, y_pred, sensitive_attribute
        )
        
        # Individual Fairness (if model and features provided)
        if model is not None and X is not None:
            # Assume last column is sensitive attribute
            sensitive_features = [X.shape[1] - 1]
            evaluation_results['individual_fairness'] = self.individual_fairness(
                model, X, sensitive_features
            )
        
        # Overall fairness assessment
        fairness_checks = [
            evaluation_results['demographic_parity']['is_fair'],
            evaluation_results['equalized_odds']['is_fair'],
            evaluation_results['equality_of_opportunity']['is_fair']
        ]
        
        evaluation_results['overall_fairness'] = {
            'all_metrics_fair': all(fairness_checks),
            'fair_metrics_count': sum(fairness_checks),
            'total_metrics': len(fairness_checks),
            'fairness_score': sum(fairness_checks) / len(fairness_checks)
        }
        
        return evaluation_results
    
    def plot_fairness_metrics(self, evaluation_results):
        """Visualize fairness evaluation results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Demographic Parity
        dp_results = evaluation_results['demographic_parity']
        groups = list(dp_results['positive_rates_by_group'].keys())
        rates = list(dp_results['positive_rates_by_group'].values())
        
        axes[0, 0].bar(groups, rates, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Demographic Parity\n(Positive Prediction Rates by Group)')
        axes[0, 0].set_ylabel('Positive Rate')
        axes[0, 0].set_ylim(0, 1)
        
        # Add difference annotation
        axes[0, 0].text(0.5, 0.95, f'Difference: {dp_results["demographic_parity_difference"]:.3f}',
                       transform=axes[0, 0].transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Equalized Odds
        eo_results = evaluation_results['equalized_odds']
        tprs = [metrics['tpr'] for metrics in eo_results['group_metrics'].values()]
        fprs = [metrics['fpr'] for metrics in eo_results['group_metrics'].values()]
        
        x = np.arange(len(groups))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, tprs, width, label='TPR', alpha=0.7)
        axes[0, 1].bar(x + width/2, fprs, width, label='FPR', alpha=0.7)
        axes[0, 1].set_title('Equalized Odds\n(TPR and FPR by Group)')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(groups)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # Equality of Opportunity
        eop_results = evaluation_results['equality_of_opportunity']
        tprs_eop = list(eop_results['tprs_by_group'].values())
        
        axes[1, 0].bar(groups, tprs_eop, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Equality of Opportunity\n(TPR by Group)')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_ylim(0, 1)
        
        # Add difference annotation
        axes[1, 0].text(0.5, 0.95, f'Difference: {eop_results["equality_of_opportunity_difference"]:.3f}',
                       transform=axes[1, 0].transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Overall Fairness Summary
        overall = evaluation_results['overall_fairness']
        metrics = ['Demographic\nParity', 'Equalized\nOdds', 'Equality of\nOpportunity']
        fairness_status = [
            evaluation_results['demographic_parity']['is_fair'],
            evaluation_results['equalized_odds']['is_fair'],
            evaluation_results['equality_of_opportunity']['is_fair']
        ]
        
        colors = ['green' if fair else 'red' for fair in fairness_status]
        axes[1, 1].bar(metrics, [1 if fair else 0 for fair in fairness_status], 
                      color=colors, alpha=0.7)
        axes[1, 1].set_title('Fairness Metrics Summary')
        axes[1, 1].set_ylabel('Fair (1) / Unfair (0)')
        axes[1, 1].set_ylim(0, 1.2)
        
        # Add overall score
        axes[1, 1].text(0.5, 1.1, f'Overall Fairness Score: {overall["fairness_score"]:.2f}',
                       transform=axes[1, 1].transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

# Example usage
def fairness_evaluation_demo():
    """Demonstrate fairness evaluation"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: age, income, education (normalized)
    X = np.random.rand(n_samples, 3)
    
    # Sensitive attribute: gender (0=female, 1=male)
    sensitive_attr = np.random.choice([0, 1], n_samples)
    
    # Introduce bias: males more likely to get positive outcomes
    bias_factor = 0.3
    y_true = (X[:, 1] + bias_factor * sensitive_attr + np.random.normal(0, 0.1, n_samples)) > 0.6
    y_true = y_true.astype(int)
    
    # Biased predictions (exaggerate the bias)
    y_pred = (X[:, 1] + bias_factor * sensitive_attr * 1.5 + np.random.normal(0, 0.1, n_samples)) > 0.6
    y_pred = y_pred.astype(int)
    
    # Initialize fairness evaluator
    evaluator = FairnessEvaluator()
    
    # Comprehensive fairness evaluation
    results = evaluator.comprehensive_fairness_evaluation(
        y_true, y_pred, sensitive_attr
    )
    
    print("Fairness Evaluation Results:")
    print("=" * 50)
    
    print("\n1. Demographic Parity:")
    dp = results['demographic_parity']
    print(f"   Positive rates by group: {dp['positive_rates_by_group']}")
    print(f"   Parity difference: {dp['demographic_parity_difference']:.4f}")
    print(f"   Is fair: {dp['is_fair']}")
    
    print("\n2. Equalized Odds:")
    eo = results['equalized_odds']
    print(f"   TPR difference: {eo['tpr_difference']:.4f}")
    print(f"   FPR difference: {eo['fpr_difference']:.4f}")
    print(f"   Is fair: {eo['is_fair']}")
    
    print("\n3. Equality of Opportunity:")
    eop = results['equality_of_opportunity']
    print(f"   TPR difference: {eop['equality_of_opportunity_difference']:.4f}")
    print(f"   Is fair: {eop['is_fair']}")
    
    print("\n4. Overall Assessment:")
    overall = results['overall_fairness']
    print(f"   Fair metrics: {overall['fair_metrics_count']}/{overall['total_metrics']}")
    print(f"   Fairness score: {overall['fairness_score']:.2f}")
    print(f"   All metrics fair: {overall['all_metrics_fair']}")
    
    return evaluator, results

# evaluator, results = fairness_evaluation_demo()
```

## 3. Bias Mitigation Strategies

### Pre-processing Techniques

```python
class BiasMitigator:
    """Comprehensive bias mitigation techniques"""
    
    def __init__(self):
        self.mitigation_results = {}
    
    def data_augmentation_for_fairness(self, texts, labels, sensitive_attributes, 
                                     target_ratio=1.0):
        """Augment data to balance representation across groups"""
        
        from collections import Counter
        
        # Group data by sensitive attribute
        groups = {}
        for i, attr in enumerate(sensitive_attributes):
            if attr not in groups:
                groups[attr] = {'texts': [], 'labels': [], 'indices': []}
            groups[attr]['texts'].append(texts[i])
            groups[attr]['labels'].append(labels[i])
            groups[attr]['indices'].append(i)
        
        # Find target size (largest group or specified ratio)
        group_sizes = {group: len(data['texts']) for group, data in groups.items()}
        target_size = max(group_sizes.values())
        
        augmented_texts = []
        augmented_labels = []
        augmented_attributes = []
        
        for group, data in groups.items():
            current_size = len(data['texts'])
            needed_samples = int(target_size * target_ratio) - current_size
            
            if needed_samples > 0:
                # Simple augmentation: repeat samples with slight modifications
                augmented_indices = np.random.choice(
                    len(data['texts']), needed_samples, replace=True
                )
                
                for idx in augmented_indices:
                    # Simple text augmentation (in practice, use more sophisticated methods)
                    original_text = data['texts'][idx]
                    augmented_text = self.simple_text_augmentation(original_text)
                    
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(data['labels'][idx])
                    augmented_attributes.append(group)
            
            # Add original data
            augmented_texts.extend(data['texts'])
            augmented_labels.extend(data['labels'])
            augmented_attributes.extend([group] * len(data['texts']))
        
        return augmented_texts, augmented_labels, augmented_attributes
    
    def simple_text_augmentation(self, text):
        """Simple text augmentation techniques"""
        
        import random
        
        # Random word replacement with synonyms (simplified)
        words = text.split()
        if len(words) > 3:
            # Randomly replace one word with a synonym placeholder
            random_idx = random.randint(0, len(words) - 1)
            words[random_idx] = f"[SYNONYM:{words[random_idx]}]"
        
        return " ".join(words)
    
    def counterfactual_data_augmentation(self, texts, sensitive_terms_map):
        """Generate counterfactual examples by swapping sensitive terms"""
        
        counterfactual_texts = []
        
        for text in texts:
            # Create counterfactual versions for each sensitive attribute
            for original_terms, replacement_terms in sensitive_terms_map.items():
                counterfactual_text = text
                
                # Replace terms
                for orig_term in original_terms:
                    for repl_term in replacement_terms:
                        if orig_term.lower() in counterfactual_text.lower():
                            counterfactual_text = counterfactual_text.replace(
                                orig_term, repl_term
                            )
                            break
                
                if counterfactual_text != text:  # Only add if changed
                    counterfactual_texts.append(counterfactual_text)
        
        return counterfactual_texts
    
    def adversarial_debiasing_loss(self, model_predictions, sensitive_attributes, 
                                 lambda_fairness=1.0):
        """Compute adversarial debiasing loss"""
        
        # Main task loss (assumed to be computed elsewhere)
        # This function focuses on the fairness/adversarial component
        
        import torch
        import torch.nn.functional as F
        
        if isinstance(model_predictions, np.ndarray):
            model_predictions = torch.tensor(model_predictions, dtype=torch.float32)
        if isinstance(sensitive_attributes, np.ndarray):
            sensitive_attributes = torch.tensor(sensitive_attributes, dtype=torch.long)
        
        # Adversarial classifier tries to predict sensitive attribute from predictions
        # This loss encourages the main model to be invariant to sensitive attributes
        
        # Simple adversarial loss: encourage uniform prediction across groups
        unique_groups = torch.unique(sensitive_attributes)
        group_losses = []
        
        for group in unique_groups:
            group_mask = sensitive_attributes == group
            group_predictions = model_predictions[group_mask]
            
            if len(group_predictions) > 0:
                # Encourage similar prediction distributions across groups
                group_mean = torch.mean(group_predictions)
                overall_mean = torch.mean(model_predictions)
                group_loss = torch.abs(group_mean - overall_mean)
                group_losses.append(group_loss)
        
        fairness_loss = torch.mean(torch.stack(group_losses)) if group_losses else torch.tensor(0.0)
        
        return lambda_fairness * fairness_loss
    
    def post_processing_calibration(self, y_pred_proba, sensitive_attributes, 
                                  y_true=None, method='equalized_odds'):
        """Post-processing calibration for fairness"""
        
        unique_groups = np.unique(sensitive_attributes)
        calibrated_predictions = y_pred_proba.copy()
        
        if method == 'demographic_parity':
            # Adjust thresholds to achieve demographic parity
            target_positive_rate = np.mean(y_pred_proba > 0.5)
            
            for group in unique_groups:
                group_mask = sensitive_attributes == group
                group_probs = y_pred_proba[group_mask]
                
                # Find threshold that achieves target positive rate
                threshold = np.percentile(group_probs, (1 - target_positive_rate) * 100)
                calibrated_predictions[group_mask] = group_probs > threshold
        
        elif method == 'equalized_odds' and y_true is not None:
            # Adjust predictions to equalize TPR and FPR across groups
            overall_tpr = np.mean(y_pred_proba[y_true == 1] > 0.5)
            overall_fpr = np.mean(y_pred_proba[y_true == 0] > 0.5)
            
            for group in unique_groups:
                group_mask = sensitive_attributes == group
                
                # Positive samples in this group
                group_pos_mask = group_mask & (y_true == 1)
                if np.sum(group_pos_mask) > 0:
                    group_pos_probs = y_pred_proba[group_pos_mask]
                    tpr_threshold = np.percentile(group_pos_probs, (1 - overall_tpr) * 100)
                    calibrated_predictions[group_pos_mask] = group_pos_probs > tpr_threshold
                
                # Negative samples in this group
                group_neg_mask = group_mask & (y_true == 0)
                if np.sum(group_neg_mask) > 0:
                    group_neg_probs = y_pred_proba[group_neg_mask]
                    fpr_threshold = np.percentile(group_neg_probs, (1 - overall_fpr) * 100)
                    calibrated_predictions[group_neg_mask] = group_neg_probs > fpr_threshold
        
        return calibrated_predictions.astype(int)
    
    def fairness_aware_ensemble(self, models, X, sensitive_attributes, 
                              fairness_weights=None):
        """Create fairness-aware ensemble of models"""
        
        if fairness_weights is None:
            fairness_weights = [1.0] * len(models)
        
        # Get predictions from all models
        all_predictions = []
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]  # Probability of positive class
            else:
                pred = model.predict(X)
            all_predictions.append(pred)
        
        # Weighted ensemble based on fairness scores
        ensemble_predictions = np.zeros(len(X))
        total_weight = sum(fairness_weights)
        
        for i, (pred, weight) in enumerate(zip(all_predictions, fairness_weights)):
            ensemble_predictions += (weight / total_weight) * pred
        
        return ensemble_predictions
    
    def implement_fairness_constraints(self, model_class, X_train, y_train, 
                                     sensitive_train, constraint_type='demographic_parity'):
        """Implement fairness constraints during training"""
        
        # Simplified implementation of fairness-constrained learning
        # In practice, would use specialized libraries like fairlearn
        
        class FairConstrainedModel:
            def __init__(self, base_model, constraint_type, lambda_fair=1.0):
                self.base_model = base_model
                self.constraint_type = constraint_type
                self.lambda_fair = lambda_fair
                self.group_thresholds = {}
            
            def fit(self, X, y, sensitive_attr):
                # Train base model
                self.base_model.fit(X, y)
                
                # Calculate group-specific thresholds for fairness
                if hasattr(self.base_model, 'predict_proba'):
                    y_pred_proba = self.base_model.predict_proba(X)[:, 1]
                else:
                    y_pred_proba = self.base_model.decision_function(X)
                
                unique_groups = np.unique(sensitive_attr)
                
                if self.constraint_type == 'demographic_parity':
                    # Set thresholds to achieve demographic parity
                    overall_positive_rate = np.mean(y_pred_proba > 0.5)
                    
                    for group in unique_groups:
                        group_mask = sensitive_attr == group
                        group_probs = y_pred_proba[group_mask]
                        threshold = np.percentile(group_probs, (1 - overall_positive_rate) * 100)
                        self.group_thresholds[group] = threshold
                
                return self
            
            def predict(self, X, sensitive_attr):
                if hasattr(self.base_model, 'predict_proba'):
                    y_pred_proba = self.base_model.predict_proba(X)[:, 1]
                else:
                    y_pred_proba = self.base_model.decision_function(X)
                
                predictions = np.zeros(len(X))
                
                for group, threshold in self.group_thresholds.items():
                    group_mask = sensitive_attr == group
                    predictions[group_mask] = (y_pred_proba[group_mask] > threshold).astype(int)
                
                return predictions
        
        # Create and train fair model
        fair_model = FairConstrainedModel(model_class(), constraint_type)
        fair_model.fit(X_train, y_train, sensitive_train)
        
        return fair_model
    
    def evaluate_mitigation_effectiveness(self, original_predictions, mitigated_predictions,
                                        y_true, sensitive_attributes):
        """Evaluate effectiveness of bias mitigation"""
        
        from sklearn.metrics import accuracy_score, f1_score
        
        # Initialize evaluators
        fairness_evaluator = FairnessEvaluator()
        
        # Evaluate original model
        original_fairness = fairness_evaluator.comprehensive_fairness_evaluation(
            y_true, original_predictions, sensitive_attributes
        )
        
        # Evaluate mitigated model
        mitigated_fairness = fairness_evaluator.comprehensive_fairness_evaluation(
            y_true, mitigated_predictions, sensitive_attributes
        )
        
        # Performance comparison
        original_accuracy = accuracy_score(y_true, original_predictions)
        mitigated_accuracy = accuracy_score(y_true, mitigated_predictions)
        
        original_f1 = f1_score(y_true, original_predictions, average='weighted')
        mitigated_f1 = f1_score(y_true, mitigated_predictions, average='weighted')
        
        # Fairness improvement
        fairness_improvements = {}
        fairness_improvements['demographic_parity'] = (
            original_fairness['demographic_parity']['demographic_parity_difference'] -
            mitigated_fairness['demographic_parity']['demographic_parity_difference']
        )
        
        fairness_improvements['equalized_odds'] = (
            original_fairness['equalized_odds']['equalized_odds_difference'] -
            mitigated_fairness['equalized_odds']['equalized_odds_difference']
        )
        
        fairness_improvements['equality_of_opportunity'] = (
            original_fairness['equality_of_opportunity']['equality_of_opportunity_difference'] -
            mitigated_fairness['equality_of_opportunity']['equality_of_opportunity_difference']
        )
        
        evaluation_summary = {
            'performance': {
                'original_accuracy': original_accuracy,
                'mitigated_accuracy': mitigated_accuracy,
                'accuracy_change': mitigated_accuracy - original_accuracy,
                'original_f1': original_f1,
                'mitigated_f1': mitigated_f1,
                'f1_change': mitigated_f1 - original_f1
            },
            'fairness_improvements': fairness_improvements,
            'original_fairness': original_fairness,
            'mitigated_fairness': mitigated_fairness,
            'overall_fairness_improvement': np.mean(list(fairness_improvements.values()))
        }
        
        return evaluation_summary

# Example usage
def bias_mitigation_demo():
    """Demonstrate bias mitigation techniques"""
    
    print("Bias Mitigation Demo")
    print("=" * 50)
    
    # Generate synthetic biased data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features and sensitive attribute
    X = np.random.rand(n_samples, 3)
    sensitive_attr = np.random.choice([0, 1], n_samples)  # Binary sensitive attribute
    
    # Create biased ground truth and predictions
    bias_factor = 0.4
    y_true = (X[:, 1] + bias_factor * 0.2 * sensitive_attr + np.random.normal(0, 0.1, n_samples)) > 0.6
    y_true = y_true.astype(int)
    
    # Original biased predictions
    original_predictions = (X[:, 1] + bias_factor * sensitive_attr + np.random.normal(0, 0.1, n_samples)) > 0.6
    original_predictions = original_predictions.astype(int)
    
    # Initialize mitigator
    mitigator = BiasMitigator()
    
    # Post-processing calibration for fairness
    original_probs = X[:, 1] + bias_factor * sensitive_attr + np.random.normal(0, 0.1, n_samples)
    original_probs = (original_probs - original_probs.min()) / (original_probs.max() - original_probs.min())  # Normalize
    
    mitigated_predictions = mitigator.post_processing_calibration(
        original_probs, sensitive_attr, y_true, method='demographic_parity'
    )
    
    # Evaluate mitigation effectiveness
    evaluation = mitigator.evaluate_mitigation_effectiveness(
        original_predictions, mitigated_predictions, y_true, sensitive_attr
    )
    
    print("\nMitigation Effectiveness:")
    print("-" * 30)
    
    print("\nPerformance Impact:")
    perf = evaluation['performance']
    print(f"  Accuracy change: {perf['accuracy_change']:+.4f}")
    print(f"  F1 change: {perf['f1_change']:+.4f}")
    
    print("\nFairness Improvements:")
    for metric, improvement in evaluation['fairness_improvements'].items():
        print(f"  {metric}: {improvement:+.4f}")
    
    print(f"\nOverall fairness improvement: {evaluation['overall_fairness_improvement']:+.4f}")
    
    # Data augmentation example
    print("\n" + "="*30)
    print("DATA AUGMENTATION EXAMPLE")
    print("="*30)
    
    # Create sample texts
    texts = [
        "The engineer completed the project successfully.",
        "The nurse provided excellent patient care.",
        "The CEO made important strategic decisions.",
        "The teacher inspired students to learn."
    ] * 250  # Repeat to have enough samples
    
    labels = [1, 1, 1, 1] * 250  # All positive for simplicity
    sensitive_attrs = [0, 1, 0, 1] * 250  # Alternating groups
    
    # Apply data augmentation
    augmented_texts, augmented_labels, augmented_attrs = mitigator.data_augmentation_for_fairness(
        texts, labels, sensitive_attrs, target_ratio=1.0
    )
    
    print(f"Original dataset size: {len(texts)}")
    print(f"Augmented dataset size: {len(augmented_texts)}")
    
    # Check balance
    original_counts = Counter(sensitive_attrs)
    augmented_counts = Counter(augmented_attrs)
    
    print(f"Original group balance: {dict(original_counts)}")
    print(f"Augmented group balance: {dict(augmented_counts)}")
    
    return mitigator, evaluation

# mitigator, evaluation = bias_mitigation_demo()
```

## 4. Learning Objectives

By the end of this section, you should be able to:
- **Identify** different types of bias in NLP models and datasets
- **Implement** comprehensive bias detection and analysis frameworks
- **Apply** fairness metrics to evaluate model equity across groups
- **Deploy** bias mitigation strategies at different stages of the ML pipeline
- **Evaluate** the trade-offs between fairness and performance
- **Design** responsible AI systems with ethical considerations

### Self-Assessment Checklist

□ Can detect gender, racial, religious, and age bias in NLP models  
□ Can implement demographic parity, equalized odds, and equality of opportunity metrics  
□ Can apply pre-processing, in-processing, and post-processing bias mitigation techniques  
□ Can evaluate fairness-performance trade-offs  
□ Can design data augmentation strategies for fairness  
□ Can implement adversarial debiasing approaches  
□ Can create comprehensive bias audit reports  

## 5. Practical Exercises

**Exercise 1: Comprehensive Bias Audit**
```python
# TODO: Conduct a comprehensive bias audit of a pre-trained model
# Test multiple types of bias (gender, race, age, religion)
# Generate detailed bias report with visualizations and recommendations
```

**Exercise 2: Fairness-Aware Model Training**
```python
# TODO: Implement fairness-constrained training from scratch
# Compare different fairness objectives and their trade-offs
# Evaluate on real-world dataset with known bias issues
```

**Exercise 3: Bias Mitigation Pipeline**
```python
# TODO: Build end-to-end bias mitigation pipeline
# Include data preprocessing, training, and post-processing steps
# Demonstrate effectiveness on multiple fairness metrics
```

## 6. Study Materials

### Essential Papers
- [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520)
- [Fairness and Abstraction in Sociotechnical Systems](https://dl.acm.org/doi/10.1145/3287560.3287598)
- [The Measure and Mismeasure of Fairness](https://arxiv.org/abs/1808.00023)
- [Mitigating Unwanted Biases with Adversarial Learning](https://arxiv.org/abs/1801.07593)

### Bias and Fairness Resources
- **Datasets**: Winogender, CrowS-Pairs, StereoSet, BOLD
- **Libraries**: fairlearn, aif360, themis-ml
- **Metrics**: Demographic parity, equalized odds, calibration
- **Tools**: What-If Tool, InterpretML, LIME

### Tools and Libraries
```bash
pip install fairlearn aif360 
pip install scikit-learn pandas numpy
pip install transformers datasets
pip install matplotlib seaborn plotly
```