# GraphJudge Evaluation System - Research Usage Guidelines

**Version:** 1.0
**Date:** 2025-09-23
**Status:** Research Ready

This document provides comprehensive guidelines for researchers using the GraphJudge evaluation system for academic research, publication, and comparative analysis of knowledge graph generation methods.

---

## Table of Contents

1. [Research Overview](#research-overview)
2. [Evaluation Methodology](#evaluation-methodology)
3. [Metric Interpretation](#metric-interpretation)
4. [Best Practices for Research](#best-practices-for-research)
5. [Comparative Analysis](#comparative-analysis)
6. [Publication Guidelines](#publication-guidelines)
7. [Reproducibility](#reproducibility)
8. [Common Pitfalls](#common-pitfalls)
9. [Citation and Attribution](#citation-and-attribution)

---

## Research Overview

### Purpose and Scope

The GraphJudge evaluation system is designed for rigorous academic research in knowledge graph generation and evaluation. It provides multi-dimensional assessment capabilities that enable researchers to:

- **Quantify Knowledge Graph Quality**: Comprehensive metrics for objective quality assessment
- **Compare Generation Methods**: Standardized evaluation framework for method comparison
- **Analyze System Performance**: Fine-grained analysis of generation pipeline components
- **Validate Research Contributions**: Robust evaluation for academic publication

### Research Domains

The evaluation system is particularly suited for research in:

- **Natural Language Processing**: Text-to-knowledge-graph conversion
- **Information Extraction**: Named entity recognition and relation extraction
- **Knowledge Representation**: Graph structure and semantic modeling
- **AI System Evaluation**: Multi-modal assessment methodologies
- **Computational Linguistics**: Cross-lingual knowledge graph generation

---

## Evaluation Methodology

### Multi-Dimensional Assessment Framework

The evaluation system implements a comprehensive assessment framework based on multiple complementary dimensions:

#### 1. Exact Matching Dimension
- **Triple Match F1**: Measures precision and recall of exact triple matches
- **Graph Match Accuracy**: Evaluates structural graph isomorphism
- **Use Case**: Strict accuracy requirements, factual correctness validation

#### 2. Text Similarity Dimension
- **G-BLEU**: Applies BLEU metrics to graph edge representations
- **G-ROUGE**: Applies ROUGE metrics to graph edge representations
- **Use Case**: Surface-level textual similarity, lexical variation tolerance

#### 3. Semantic Similarity Dimension
- **G-BertScore**: Uses BERT embeddings for semantic similarity assessment
- **Use Case**: Semantic equivalence detection, paraphrase recognition

#### 4. Structural Distance Dimension
- **Graph Edit Distance (Optional)**: Minimum edit operations for graph transformation
- **Use Case**: Structural analysis, graph topology comparison

### Evaluation Protocol

#### Standard Research Protocol

1. **Dataset Preparation**
   ```python
   # Prepare your datasets
   predicted_graphs = load_predicted_graphs("model_output.json")
   reference_graphs = load_reference_graphs("gold_standard.json")

   # Validate data quality
   validate_graph_consistency(predicted_graphs, reference_graphs)
   ```

2. **Evaluation Configuration**
   ```python
   # Configure evaluation for research reproducibility
   evaluator = GraphEvaluator(
       enable_ged=True,              # Include structural metrics
       enable_bert_score=True,       # Include semantic metrics
       max_evaluation_time=300.0     # Allow sufficient time
   )
   ```

3. **Batch Evaluation**
   ```python
   # Run comprehensive evaluation
   results = evaluator.evaluate_batch(
       [(pred, ref) for pred, ref in zip(predicted_graphs, reference_graphs)]
   )
   ```

4. **Statistical Analysis**
   ```python
   # Compute aggregate statistics
   metrics_analysis = analyze_evaluation_results(results)
   ```

#### Cross-Validation Protocol

For robust research findings, implement cross-validation:

```python
from sklearn.model_selection import KFold

def cross_validate_evaluation(graphs, k_folds=5):
    """Perform k-fold cross-validation for evaluation stability."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_results = []

    for train_idx, test_idx in kf.split(graphs):
        test_graphs = [graphs[i] for i in test_idx]
        fold_results = evaluate_graph_set(test_graphs)
        cv_results.append(fold_results)

    return aggregate_cv_results(cv_results)
```

---

## Metric Interpretation

### Individual Metric Analysis

#### Triple Match F1 Score
- **Range**: 0.0 - 1.0
- **Interpretation**:
  - 1.0 = Perfect match (all triples exactly correct)
  - 0.8-0.9 = High accuracy with minor errors
  - 0.6-0.8 = Moderate accuracy, significant room for improvement
  - <0.6 = Low accuracy, substantial errors
- **Research Use**: Primary metric for factual accuracy assessment

#### Graph Match Accuracy
- **Range**: 0.0 - 1.0
- **Interpretation**:
  - 1.0 = Identical graph structure
  - 0.8-0.9 = Very similar structure with minor differences
  - 0.5-0.8 = Moderate structural similarity
  - <0.5 = Significant structural differences
- **Research Use**: Structural coherence and organization assessment

#### G-BLEU/G-ROUGE F1
- **Range**: 0.0 - 1.0
- **Interpretation**:
  - Similar to traditional BLEU/ROUGE but applied to graph edges
  - Measures n-gram overlap in graph edge representations
  - Tolerates lexical variations and synonyms
- **Research Use**: Surface-level similarity, robustness to paraphrasing

#### G-BertScore F1
- **Range**: 0.0 - 1.0 (typically 0.6-1.0 for meaningful content)
- **Interpretation**:
  - 0.9-1.0 = Semantically very similar or identical
  - 0.8-0.9 = Semantically similar with minor differences
  - 0.7-0.8 = Moderate semantic similarity
  - <0.7 = Limited semantic similarity
- **Research Use**: Semantic equivalence, meaning preservation

#### Graph Edit Distance
- **Range**: 0 - ∞ (lower is better)
- **Interpretation**:
  - 0 = Identical graphs
  - 1-5 = Minor structural differences
  - 5-20 = Moderate structural differences
  - >20 = Major structural differences
- **Research Use**: Fine-grained structural analysis

### Overall Score Interpretation

The overall score combines key metrics using equal weighting:

```python
overall_score = (triple_match_f1 + graph_match_accuracy +
                g_bleu_f1 + g_rouge_f1 + g_bert_f1) / 5
```

#### Score Ranges and Quality Assessment

- **0.9-1.0 (Excellent)**: Publication-ready quality, minimal errors
- **0.8-0.9 (Good)**: High quality with minor improvement opportunities
- **0.7-0.8 (Fair)**: Moderate quality, significant improvement needed
- **0.6-0.7 (Poor)**: Low quality, substantial issues present
- **<0.6 (Unacceptable)**: Very poor quality, major problems

---

## Best Practices for Research

### Experimental Design

#### 1. Dataset Selection and Preparation

**Reference Graph Quality**
```python
# Validate reference graph quality
def validate_reference_quality(reference_graphs):
    """Ensure reference graphs meet research standards."""
    for graph in reference_graphs:
        # Check completeness
        assert len(graph) > 0, "Empty reference graph"

        # Check consistency
        validate_triple_consistency(graph)

        # Check coverage
        validate_domain_coverage(graph)
```

**Data Splitting**
```python
# Proper train/validation/test splitting
from sklearn.model_selection import train_test_split

def prepare_research_datasets(data, test_size=0.2, val_size=0.1):
    """Prepare datasets with proper splitting for research."""
    # Split data maintaining domain distribution
    train_val, test = train_test_split(
        data, test_size=test_size, stratify=data['domain'], random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=42
    )
    return train, val, test
```

#### 2. Evaluation Configuration

**Reproducible Configuration**
```python
# Standard research configuration
RESEARCH_CONFIG = {
    'enable_ged': True,              # Include all metrics
    'enable_bert_score': True,       # Enable semantic analysis
    'max_evaluation_time': 300.0,    # Allow sufficient time
    'random_seed': 42,               # Ensure reproducibility
    'evaluation_version': '1.0'      # Track evaluation version
}

evaluator = GraphEvaluator(**RESEARCH_CONFIG)
```

#### 3. Statistical Rigor

**Significance Testing**
```python
from scipy import stats

def statistical_comparison(results_a, results_b, metric='overall_score'):
    """Compare two sets of evaluation results with statistical testing."""
    scores_a = [getattr(r.metrics, metric) for r in results_a if r.success]
    scores_b = [getattr(r.metrics, metric) for r in results_b if r.success]

    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(scores_a, scores_b)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(scores_a)-1)*np.var(scores_a) +
                         (len(scores_b)-1)*np.var(scores_b)) /
                        (len(scores_a)+len(scores_b)-2))
    cohens_d = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std

    return {
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': cohens_d,
        'significant': p_value < 0.05
    }
```

**Confidence Intervals**
```python
def calculate_confidence_intervals(results, confidence=0.95):
    """Calculate confidence intervals for evaluation metrics."""
    import scipy.stats as stats

    scores = [r.metrics.get_overall_score() for r in results if r.success]
    mean_score = np.mean(scores)
    std_error = stats.sem(scores)

    # Calculate confidence interval
    ci = stats.t.interval(
        confidence,
        len(scores)-1,
        loc=mean_score,
        scale=std_error
    )

    return {
        'mean': mean_score,
        'confidence_interval': ci,
        'margin_of_error': ci[1] - mean_score
    }
```

### Performance Profiling

#### Resource Usage Monitoring
```python
import psutil
import time

def profile_evaluation_performance(evaluator, graph_pairs):
    """Profile evaluation performance for research reporting."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # Run evaluation
    results = evaluator.evaluate_batch(graph_pairs)

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    return {
        'total_time': end_time - start_time,
        'time_per_graph': (end_time - start_time) / len(graph_pairs),
        'memory_usage': end_memory - start_memory,
        'successful_evaluations': sum(1 for r in results if r.success),
        'evaluation_rate': len(graph_pairs) / (end_time - start_time)
    }
```

---

## Comparative Analysis

### Method Comparison Framework

#### Standardized Comparison Protocol

```python
class MethodComparison:
    """Framework for comparing different KG generation methods."""

    def __init__(self, reference_graphs):
        self.reference_graphs = reference_graphs
        self.evaluator = GraphEvaluator(
            enable_ged=True,
            enable_bert_score=True,
            max_evaluation_time=300.0
        )

    def compare_methods(self, method_results):
        """Compare multiple methods on the same dataset."""
        comparison_results = {}

        for method_name, predicted_graphs in method_results.items():
            # Evaluate method
            results = self.evaluator.evaluate_batch([
                (pred, ref) for pred, ref in
                zip(predicted_graphs, self.reference_graphs)
            ])

            # Compute aggregate statistics
            comparison_results[method_name] = self.aggregate_results(results)

        return self.generate_comparison_report(comparison_results)

    def aggregate_results(self, results):
        """Aggregate evaluation results for method comparison."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return None

        metrics = {
            'triple_match_f1': np.mean([r.metrics.triple_match_f1 for r in successful_results]),
            'graph_match_accuracy': np.mean([r.metrics.graph_match_accuracy for r in successful_results]),
            'g_bleu_f1': np.mean([r.metrics.g_bleu_f1 for r in successful_results]),
            'g_rouge_f1': np.mean([r.metrics.g_rouge_f1 for r in successful_results]),
            'g_bert_f1': np.mean([r.metrics.g_bert_f1 for r in successful_results]),
            'overall_score': np.mean([r.metrics.get_overall_score() for r in successful_results]),
            'std_dev': np.std([r.metrics.get_overall_score() for r in successful_results]),
            'success_rate': len(successful_results) / len(results)
        }

        return metrics
```

#### Visualization for Research

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_comparison_visualizations(comparison_results):
    """Create publication-ready comparison visualizations."""

    # Radar chart for multi-dimensional comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Radar chart
    categories = ['Triple F1', 'Graph Acc', 'G-BLEU F1', 'G-ROUGE F1', 'G-BERT F1']

    for method_name, metrics in comparison_results.items():
        values = [
            metrics['triple_match_f1'],
            metrics['graph_match_accuracy'],
            metrics['g_bleu_f1'],
            metrics['g_rouge_f1'],
            metrics['g_bert_f1']
        ]

        # Plot radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values += values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))

        axes[0].plot(angles, values, 'o-', linewidth=2, label=method_name)
        axes[0].fill(angles, values, alpha=0.25)

    axes[0].set_xticks(angles[:-1])
    axes[0].set_xticklabels(categories)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Multi-Dimensional Performance Comparison')
    axes[0].legend()
    axes[0].grid(True)

    # Bar chart for overall scores
    methods = list(comparison_results.keys())
    overall_scores = [comparison_results[m]['overall_score'] for m in methods]
    std_devs = [comparison_results[m]['std_dev'] for m in methods]

    axes[1].bar(methods, overall_scores, yerr=std_devs, capsize=5, alpha=0.7)
    axes[1].set_ylabel('Overall Score')
    axes[1].set_title('Overall Performance Comparison')
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    return fig
```

---

## Publication Guidelines

### Reporting Standards

#### Results Reporting Template

```python
def generate_research_report(evaluation_results, method_name="Proposed Method"):
    """Generate standardized research report for publication."""

    report = f"""
## Evaluation Results for {method_name}

### Dataset Statistics
- Total Graphs Evaluated: {len(evaluation_results)}
- Successful Evaluations: {sum(1 for r in evaluation_results if r.success)}
- Success Rate: {sum(1 for r in evaluation_results if r.success)/len(evaluation_results)*100:.1f}%

### Performance Metrics (Mean ± Std Dev)
"""

    # Calculate statistics
    successful_results = [r for r in evaluation_results if r.success]

    metrics = {
        'Triple Match F1': [r.metrics.triple_match_f1 for r in successful_results],
        'Graph Match Accuracy': [r.metrics.graph_match_accuracy for r in successful_results],
        'G-BLEU F1': [r.metrics.g_bleu_f1 for r in successful_results],
        'G-ROUGE F1': [r.metrics.g_rouge_f1 for r in successful_results],
        'G-BertScore F1': [r.metrics.g_bert_f1 for r in successful_results],
        'Overall Score': [r.metrics.get_overall_score() for r in successful_results]
    }

    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        report += f"- {metric_name}: {mean_val:.3f} ± {std_val:.3f}\n"

    return report
```

#### LaTeX Table Generation

```python
def generate_latex_results_table(comparison_results):
    """Generate LaTeX table for research publication."""

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Knowledge Graph Generation Performance Comparison}
\label{tab:kg_performance}
\begin{tabular}{lcccccc}
\toprule
Method & Triple F1 & Graph Acc & G-BLEU F1 & G-ROUGE F1 & G-BERT F1 & Overall \\
\midrule
"""

    for method_name, metrics in comparison_results.items():
        latex += f"{method_name} & "
        latex += f"{metrics['triple_match_f1']:.3f} & "
        latex += f"{metrics['graph_match_accuracy']:.3f} & "
        latex += f"{metrics['g_bleu_f1']:.3f} & "
        latex += f"{metrics['g_rouge_f1']:.3f} & "
        latex += f"{metrics['g_bert_f1']:.3f} & "
        latex += f"{metrics['overall_score']:.3f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    return latex
```

### Methodological Transparency

#### Configuration Reporting

Always report evaluation configuration for reproducibility:

```python
def report_evaluation_configuration():
    """Report evaluation configuration for research transparency."""
    return {
        'evaluation_framework': 'GraphJudge v1.0',
        'metrics_enabled': {
            'exact_matching': True,
            'text_similarity': True,
            'semantic_similarity': True,
            'structural_distance': True
        },
        'configuration': {
            'enable_ged': True,
            'enable_bert_score': True,
            'max_evaluation_time': 300.0,
            'bert_model': 'bert-base-uncased',
            'evaluation_timeout': '5 minutes'
        },
        'dependencies': {
            'nltk': '3.8+',
            'rouge_score': '0.1.2+',
            'bert_score': '0.3.13+',
            'networkx': '3.0+'
        }
    }
```

---

## Reproducibility

### Code and Data Sharing

#### Research Package Template

```python
# research_evaluation.py
"""
Reproducible evaluation script for [Paper Title]
Authors: [Author Names]
Conference: [Conference Name]
Year: [Year]
"""

import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Reproduce evaluation results')
    parser.add_argument('--predicted', required=True, help='Predicted graphs file')
    parser.add_argument('--reference', required=True, help='Reference graphs file')
    parser.add_argument('--output', required=True, help='Output results file')
    parser.add_argument('--config', default='evaluation_config.json', help='Evaluation config')

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = json.load(f)

    # Initialize evaluator with exact configuration
    evaluator = GraphEvaluator(**config['evaluator_params'])

    # Load data
    predicted_graphs = load_graphs(args.predicted)
    reference_graphs = load_graphs(args.reference)

    # Run evaluation
    results = evaluator.evaluate_batch([
        (pred, ref) for pred, ref in zip(predicted_graphs, reference_graphs)
    ])

    # Save results
    save_results(results, args.output)

    # Generate report
    report = generate_research_report(results)
    print(report)

if __name__ == '__main__':
    main()
```

#### Configuration File Template

```json
{
  "evaluation_metadata": {
    "paper_title": "Your Paper Title",
    "authors": ["Author 1", "Author 2"],
    "evaluation_date": "2025-09-23",
    "evaluation_version": "1.0"
  },
  "evaluator_params": {
    "enable_ged": true,
    "enable_bert_score": true,
    "max_evaluation_time": 300.0
  },
  "dataset_info": {
    "name": "Your Dataset Name",
    "version": "1.0",
    "size": 1000,
    "domains": ["domain1", "domain2"]
  },
  "system_info": {
    "python_version": "3.8+",
    "required_packages": {
      "streamlit-pipeline": "1.0",
      "nltk": "3.8+",
      "bert-score": "0.3.13+"
    }
  }
}
```

### Version Control

#### Git Repository Structure

```
research_project/
├── README.md                  # Project overview and setup
├── requirements.txt           # Python dependencies
├── evaluation_config.json    # Evaluation configuration
├── data/
│   ├── predicted_graphs/     # Model outputs
│   ├── reference_graphs/     # Gold standard data
│   └── metadata/             # Dataset metadata
├── scripts/
│   ├── run_evaluation.py     # Main evaluation script
│   ├── analyze_results.py    # Results analysis
│   └── generate_plots.py     # Visualization generation
├── results/
│   ├── raw_results.json      # Raw evaluation results
│   ├── processed_results.csv # Processed statistics
│   └── figures/              # Generated plots
└── paper/
    ├── results_table.tex     # LaTeX tables
    └── supplementary.pdf     # Supplementary materials
```

---

## Common Pitfalls

### 1. Data Leakage

**Problem**: Using test data during development or evaluation tuning.

**Solution**: Strict train/validation/test separation
```python
# Maintain strict data separation
def validate_data_separation(train_ids, val_ids, test_ids):
    """Ensure no overlap between dataset splits."""
    assert len(set(train_ids) & set(val_ids)) == 0, "Train-validation overlap"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train-test overlap"
    assert len(set(val_ids) & set(test_ids)) == 0, "Validation-test overlap"
```

### 2. Cherry-Picking Results

**Problem**: Reporting only favorable metrics or subsets.

**Solution**: Report all metrics consistently
```python
def report_complete_results(results):
    """Report all evaluation metrics without cherry-picking."""
    return {
        'all_metrics': calculate_all_metrics(results),
        'failure_analysis': analyze_failures(results),
        'sample_size': len(results),
        'confidence_intervals': calculate_confidence_intervals(results)
    }
```

### 3. Inadequate Baselines

**Problem**: Comparing only against weak baselines.

**Solution**: Include multiple strong baselines
```python
def comprehensive_baseline_comparison():
    """Include multiple baseline methods for fair comparison."""
    baselines = {
        'random_baseline': generate_random_graphs,
        'rule_based_baseline': generate_rule_based_graphs,
        'existing_sota': load_sota_results,
        'human_performance': load_human_annotations
    }
    return baselines
```

### 4. Statistical Issues

**Problem**: Insufficient sample sizes or improper statistical tests.

**Solution**: Power analysis and appropriate testing
```python
def validate_sample_size(effect_size=0.5, power=0.8, alpha=0.05):
    """Calculate required sample size for statistical power."""
    from statsmodels.stats.power import ttest_power
    return ttest_power(effect_size, power, alpha)
```

---

## Citation and Attribution

### How to Cite

When using the GraphJudge evaluation system in research, please cite:

```bibtex
@software{graphjudge_evaluation,
  title={GraphJudge Evaluation System: Multi-Dimensional Assessment for Knowledge Graph Generation},
  author={[Author Names]},
  year={2025},
  version={1.0},
  url={https://github.com/your-repo/streamlit_pipeline}
}
```

### Acknowledging Metrics

If using specific metrics, cite the original papers:

- **BLEU**: Papineni et al. (2002)
- **ROUGE**: Lin (2004)
- **BertScore**: Zhang et al. (2019)
- **Graph Edit Distance**: Sanfeliu & Fu (1983)

### Contributing Back

We encourage researchers to contribute improvements:

1. **Report Issues**: Submit bug reports and feature requests
2. **Share Datasets**: Contribute evaluation datasets for community use
3. **Method Contributions**: Add new evaluation metrics or methods
4. **Documentation**: Improve documentation and examples

---

This research guide provides comprehensive guidelines for using the GraphJudge evaluation system in academic research. For technical API details, see the [API Reference](EVALUATION_API.md).

For questions or support, please contact the development team or submit issues to the project repository.