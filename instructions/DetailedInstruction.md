# AI Self-Awareness Research

A research framework to study how well AI models can identify their own generated text, investigating the hypothesis that AI models should achieve higher accuracy when identifying text they generated themselves compared to text from other models.

## 🔬 Research Overview

This project implements two main evaluation tasks:

1. **Exact Model Prediction**: Given a text and a list of possible models, predict which specific model generated the text
2. **Binary Self-Identification**: Given a text, determine whether the evaluating model generated it (self) or another model did (not-self)

The key hypothesis is that models should demonstrate "self-awareness" by being better at identifying their own generated content.

## 🏗️ Project Structure

```
ai-self-awareness/
├── src/
│   ├── config.py              # Configuration management system
│   ├── openrouter_client.py   # OpenRouter API integration
│   ├── corpus_generator.py    # Text corpus generation
│   ├── evaluator.py          # Model identification evaluation
│   ├── visualizer.py         # Results visualization and analysis
│   └── cli.py                # Command-line interface
├── config.yaml               # Main configuration file
├── pyproject.toml            # Project dependencies and metadata
├── .env                      # Environment variables (API keys)
└── CLAUDE.md                 # Project development instructions
```

## 🆕 Model Hints Feature

The framework supports two evaluation modes:

### **Standard Evaluation** (Default)
Models are given text and asked to identify which model generated it, without additional context.

### **Hint-Based Evaluation** (`--use-model-hints`)
Models are explicitly told which models are in the evaluation set before making predictions:

**Example prompt difference:**
- **Standard**: "Which model generated this text? Choose from: GPT-4, Claude, Gemini"
- **With hints**: "You know one of the following models generated this text: GPT-4, Claude, Gemini. Which model generated this text? Choose from: GPT-4, Claude, Gemini"

**Key Features:**
- **Separate files**: Results saved with `_with_hints` suffix to prevent overwriting
- **Both tasks**: Works with exact model prediction and binary self-identification
- **Research value**: Compare model performance with and without explicit context

**Example files generated:**
- Regular: `predictions_exact_model.jsonl`, `predictions_binary_self.jsonl`
- With hints: `predictions_exact_model_with_hints.jsonl`, `predictions_binary_self_with_hints.jsonl`

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-self-awareness

# Install dependencies and package using uv
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Configuration

```bash
# Initialize default configuration
ai-self-awareness init-config

# Set up your API key
cp .env.example .env
# Edit .env and add your OpenRouter API key:
# OPENROUTER_API_KEY=your_api_key_here
```

### 3. Run Complete Experiment

```bash
# Run both exact model prediction and binary self-identification tasks
ai-self-awareness run-experiment

# Run only exact model prediction
ai-self-awareness run-experiment --task exact

# Run only binary self-identification
ai-self-awareness run-experiment --task binary

# Run experiment with model hints (generates separate results files)
ai-self-awareness run-experiment --use-model-hints

# Compare both evaluation modes by running them separately
ai-self-awareness run-experiment --task both                    # Standard evaluation
ai-self-awareness run-experiment --task both --use-model-hints  # Hint-based evaluation
```

### Model Hints Usage Examples

```bash
# Standard evaluation (default)
ai-self-awareness evaluate --task both

# Hint-based evaluation (saves to separate files)
ai-self-awareness evaluate --task both --use-model-hints

# Compare exact model prediction with and without hints
ai-self-awareness evaluate --task exact                    # → predictions_exact_model.jsonl
ai-self-awareness evaluate --task exact --use-model-hints  # → predictions_exact_model_with_hints.jsonl

# Binary self-identification with hints
ai-self-awareness evaluate --task binary --use-model-hints

# Create visualizations for both evaluation modes
ai-self-awareness visualize --predictions-file results/predictions_500/predictions_exact_model.jsonl
ai-self-awareness visualize --predictions-file results/predictions_500/predictions_exact_model_with_hints.jsonl
```

## 📋 Commands

### `init-config`
Initialize a configuration file with default settings.

```bash
ai-self-awareness init-config [--config-path config.yaml]
```

### `generate-corpus`
Generate a corpus of text samples from different AI models.

```bash
ai-self-awareness generate-corpus [OPTIONS]

Options:
  --output-file TEXT     Output file path (default from config)
  --num-samples INTEGER  Number of samples (default from config)
  --append / --no-append Append to existing file (default: append)
  --config-path TEXT     Configuration file path [default: config.yaml]
  --api-key TEXT         OpenRouter API key
```

**Usage Examples:**
```bash
# Add samples to existing corpus (default behavior)
ai-self-awareness generate-corpus --num-samples 50

# Add more samples to existing corpus
ai-self-awareness generate-corpus --num-samples 25

# Start fresh corpus (overwrite existing)
ai-self-awareness generate-corpus --num-samples 10 --no-append
```

### `evaluate`
Evaluate model identification accuracy on a corpus.

```bash
ai-self-awareness evaluate [OPTIONS]

Options:
  --corpus-file TEXT     Corpus file path (default from config)
  --output-dir TEXT      Output directory (default from config)
  --task TEXT            Evaluation task: 'exact', 'binary', or 'both' [default: exact]
  --evaluator-models TEXT Comma-separated list of evaluator models (overrides config)
  --temperature FLOAT    Temperature for evaluation generation [default: 0.0]
  --append-predictions / --no-append-predictions  Append to existing predictions file [default: True]
  --use-model-hints     Include model list hints in evaluation prompts (saves to separate files)
  --config-path TEXT     Configuration file path [default: config.yaml]
  --api-key TEXT         OpenRouter API key
```

### `run-experiment`
Run the complete experiment: generate corpus and evaluate.

```bash
ai-self-awareness run-experiment [OPTIONS]

Options:
  --task TEXT            Evaluation task: 'exact', 'binary', or 'both' [default: both]
  --force-regenerate    Force regeneration even if model has sufficient samples
  --min-words INTEGER   Minimum word count for valid samples [default: 10]
  --temperature FLOAT   Temperature for evaluation generation [default: 0.0]
  --append-predictions / --no-append-predictions  Append to existing predictions file [default: True]
  --use-model-hints     Include model list hints in evaluation prompts (saves to separate files)
  --config-path TEXT     Configuration file path [default: config.yaml]
  --api-key TEXT         OpenRouter API key
```

### `check-empty-predictions`
Check for empty returned_text in predictions and rerun those evaluations with async processing.

```bash
ai-self-awareness check-empty-predictions PREDICTIONS_FILE [OPTIONS]

Arguments:
  PREDICTIONS_FILE           Path to predictions JSONL file to check

Options:
  --max-retries INTEGER     Maximum retry attempts for each empty prediction [default: 3]
  --concurrent-limit INTEGER Number of concurrent API requests [default: 10]
  --use-async / --no-use-async Use async processing for faster execution [default: True]
  --config-path TEXT        Configuration file path [default: config.yaml]
  --api-key TEXT            OpenRouter API key
```

### `update-predictions`
Update prediction correctness using different matching criteria and recompute accuracies.

```bash
ai-self-awareness update-predictions PREDICTIONS_FILE [OPTIONS]

Arguments:
  PREDICTIONS_FILE           Path to predictions JSONL file to update

Options:
  --correctness-method TEXT  Correctness method: 'exact', 'fuzzy', 'partial', or 'llm' [default: exact]
  --similarity-threshold FLOAT Similarity threshold for fuzzy matching (0.0-1.0) [default: 0.8]
  --judge-model TEXT        Model to use for LLM-based correctness judgment [default: openai/gpt-4.1-mini]
  --judge-temperature FLOAT Temperature for the judge model [default: 0.0]
  --output-dir TEXT         Output directory for new visualizations (default: results/updated_{method})
  --create-visualizations / --no-create-visualizations Create new visualizations [default: True]
  --config-path TEXT        Configuration file path [default: config.yaml]
```

### `visualize`
Create visualizations from existing evaluation results without re-running evaluation.

```bash
ai-self-awareness visualize [OPTIONS]

Options:
  --predictions-file TEXT Path to predictions JSONL file (default: most recent in predictions dir)
  --output-dir TEXT      Output directory for plots (default from config)
  --task TEXT            Task type: 'exact', 'binary', or 'auto' (detect from file) [default: auto]
  --plot-types TEXT      Comma-separated plot types: 'barplot,comparison,confusion,simple,frequency_heatmap,accuracy_heatmap' (default: all relevant)
  --format TEXT          Output format: 'pdf', 'png', 'svg', etc. [default: pdf]
  --config-path TEXT     Configuration file path [default: config.yaml]
```

**Usage Examples:**
```bash
# Create all relevant plots from most recent evaluation results
ai-self-awareness visualize

# Visualize specific predictions file with custom format
ai-self-awareness visualize --predictions-file results/predictions/predictions_binary_self.jsonl --format png

# Create only specific plot types
ai-self-awareness visualize --plot-types barplot,confusion --task binary

# Create heatmap visualizations for exact model predictions
ai-self-awareness visualize --plot-types frequency_heatmap,accuracy_heatmap --task exact

# Customize output directory
ai-self-awareness visualize --output-dir custom_plots/

# Check and rerun empty predictions for exact model task (async, 10 concurrent requests)
ai-self-awareness check-empty-predictions results/predictions/predictions_exact_model.jsonl

# Check and rerun empty predictions for binary self task with higher concurrency
ai-self-awareness check-empty-predictions results/predictions/predictions_binary_self.jsonl --concurrent-limit 20

# Use custom retry limit with sync processing
ai-self-awareness check-empty-predictions results/predictions/predictions_exact_model.jsonl --max-retries 5 --no-use-async

# High concurrency async processing for faster recovery
ai-self-awareness check-empty-predictions results/predictions/predictions_exact_model.jsonl --concurrent-limit 15

# Update predictions with fuzzy matching for more lenient evaluation
ai-self-awareness update-predictions results/predictions_exact_model.jsonl --correctness-method fuzzy --similarity-threshold 0.7

# Use partial matching to ignore provider prefixes (openai/gpt-4 matches gpt-4)
ai-self-awareness update-predictions results/predictions_exact_model.jsonl --correctness-method partial

# Update predictions without creating new visualizations
ai-self-awareness update-predictions results/predictions_exact_model.jsonl --correctness-method fuzzy --no-create-visualizations

# Use LLM judge for more nuanced correctness evaluation (default: GPT-4.1-mini)
ai-self-awareness update-predictions results/predictions_exact_model.jsonl --correctness-method llm

# Use different judge model with custom temperature
ai-self-awareness update-predictions results/predictions_exact_model.jsonl --correctness-method llm --judge-model anthropic/claude-3-haiku --judge-temperature 0.1
```

## ⚙️ Configuration

The `config.yaml` file contains all project settings, now separated into generation and evaluation configurations:

```yaml
openrouter_api_key: null  # Set via environment variable instead

generation:
  models:  # Models used for text generation
    - name: "moonshotai/kimi-k2:free"
      display_name: "Kimi K2"
      enabled: true
    - name: "z-ai/glm-4.5-air:free"
      display_name: "GLM-4.5-Air"
      enabled: true
  corpus_size: 100
  target_word_count: 100
  max_tokens: 150
  temperature: 0.7
  request_delay: 1.0
  prompts:
    - "Write a paragraph about the future of artificial intelligence."
    - "Describe a day in the life of a person living in a smart city."
    # ... more prompts

evaluation:
  evaluator_models: []  # Models used for evaluation (empty = use generation models)
  tasks: ["exact_model", "binary_self"]
  temperature: 0.0      # Temperature for evaluation generation
  request_delay: 1.0
  save_predictions: true
  predictions_dir: "results/predictions"
  use_model_hints: false  # Include model list hints in evaluation prompts

output_dir: "results"
plot_dir: "results"     # Directory for visualization plots
corpus_file: "data/corpus.jsonl"
```

### Key Configuration Options:

**Generation Settings:**
- **generation.models**: Models used for text generation
- **generation.corpus_size**: Total number of text samples to generate
- **generation.target_word_count**: Target length for each generated text
- **generation.prompts**: List of prompts used for text generation
- **generation.request_delay**: Delay between generation requests (rate limiting)

**Evaluation Settings:**
- **evaluation.evaluator_models**: Models used for evaluation (empty = use generation models)
- **evaluation.tasks**: Tasks to run ("exact_model", "binary_self", or both)
- **evaluation.temperature**: Temperature for evaluation generation (0.0 = deterministic)
- **evaluation.request_delay**: Delay between evaluation requests
- **evaluation.save_predictions**: Whether to save detailed prediction files
- **evaluation.predictions_dir**: Directory for prediction files
- **evaluation.use_model_hints**: Include model list hints in evaluation prompts (default: false)

**General Settings:**
- **corpus_file**: Path to store the generated corpus (JSONL format)
- **output_dir**: Directory for results and visualizations
- **plot_dir**: Directory for visualization plots (can be different from output_dir)

## 📊 Output and Results

### Generated Files:

1. **Corpus**: `data/corpus.jsonl` - JSONL file with generated texts and metadata

2. **Exact Model Results** (Standard):
   - `results/accuracy_barplot_exact.pdf` - Model prediction accuracy by evaluator and target model
   - `results/self_awareness_comparison.pdf` - Self vs cross-identification comparison
   - `results/prediction_frequency_heatmap.pdf` - Frequency matrix of predictions (how often j predicts i)
   - `results/conditional_accuracy_heatmap.pdf` - Conditional accuracy matrix (accuracy when j evaluates i)
   - `results/predictions_exact_model.jsonl` - Detailed prediction records for analysis

3. **Exact Model Results** (With Hints):
   - Same visualization files as standard mode
   - `results/predictions_exact_model_with_hints.jsonl` - Detailed prediction records with model hints

4. **Binary Self Results** (Standard):
   - `results/accuracy_barplot_binary.pdf` - Binary classification metrics (accuracy, precision, recall, F1)
   - `results/confusion_matrix_binary.pdf` - Confusion matrix visualization
   - `results/binary_comparison.pdf` - Simple binary accuracy comparison
   - `results/predictions_binary_self.jsonl` - Detailed binary prediction records

5. **Binary Self Results** (With Hints):
   - Same visualization files as standard mode
   - `results/predictions_binary_self_with_hints.jsonl` - Detailed prediction records with model hints

### Corpus Format (JSONL):
```json
{"text": "Generated paragraph...", "model": "moonshotai/kimi-k2:free", "model_display_name": "Kimi K2", "prompt": "Write about...", "word_count": 95}
{"text": "Another paragraph...", "model": "z-ai/glm-4.5-air:free", "model_display_name": "GLM-4.5-Air", "prompt": "Describe...", "word_count": 102}
```

### Metrics Explanation:

**Exact Model Prediction:**
- **Overall Accuracy**: Percentage of correctly identified models across all texts
- **Self-Identification**: Accuracy when a model evaluates its own generated text  
- **Cross-Identification**: Average accuracy when a model evaluates other models' texts
- **Self-Awareness Advantage**: Difference between self and cross-identification accuracy (see METRICS.md for detailed definition)

## 🔍 Correctness Evaluation Methods

The `update-predictions` command supports different methods for determining whether a prediction is correct:

### **Exact Match** (`--correctness-method exact`)
- **Criteria**: Predicted model name must exactly match the true model name
- **Use case**: Strict evaluation requiring perfect string matching
- **Example**: ❌ "gpt-4" vs "openai/gpt-4" → Incorrect

### **Fuzzy Matching** (`--correctness-method fuzzy`)  
- **Criteria**: Uses string similarity with configurable threshold (default 0.8)
- **Use case**: Handle minor spelling variations or formatting differences
- **Example**: ✅ "gpt-4-turbo" vs "gpt-4" → Correct (high similarity)

### **Partial Matching** (`--correctness-method partial`)
- **Criteria**: Ignores provider prefixes and version suffixes
- **Use case**: Focus on core model identity rather than provider details
- **Examples**: 
  - ✅ "gpt-4" vs "openai/gpt-4" → Correct
  - ✅ "claude" vs "anthropic/claude-3-sonnet" → Correct

### **LLM Judge** (`--correctness-method llm`)
- **Criteria**: Uses an LLM (default: GPT-4.1-mini) to judge correctness
- **Use case**: Most nuanced evaluation, considers context and reasoning
- **Features**:
  - Analyzes the model's full response text, not just extracted prediction
  - Considers formatting differences, abbreviations, and context
  - Falls back to exact matching if LLM judge fails or gives unclear response
  - Customizable judge model and temperature

**Example LLM Judge Scenarios:**
- Model responds: "I believe this text was generated by GPT-4 based on the writing style"
- Extracted prediction: "GPT-4", True model: "openai/gpt-4"
- **Exact**: ❌ Incorrect, **LLM**: ✅ Correct (judge understands intent)

**Binary Self-Identification:**
- **Accuracy**: Overall correct classification rate (self vs not-self)
- **Precision**: Of texts claimed as "self", how many were actually self-generated
- **Recall**: Of actual self-generated texts, how many were correctly identified
- **F1 Score**: Harmonic mean of precision and recall

## 🎨 Visualization and Analysis

The `visualize` command provides flexible visualization options without re-running expensive evaluations:

### Available Plot Types:

- **`barplot`**: Main accuracy/metrics comparison (available for both tasks)
- **`comparison`**: Self vs cross-model identification analysis (exact task only)  
- **`confusion`**: Confusion matrix visualization (binary task only)
- **`simple`**: Simple accuracy comparison (binary task only)
- **`frequency_heatmap`**: Prediction frequency matrix showing how often evaluator j predicts model i (exact task only)
- **`accuracy_heatmap`**: Conditional accuracy heatmap showing P(correct | evaluator=j, target=i) (exact task only)

### Benefits:

- **Iterate on aesthetics**: Adjust plot styles and formats quickly
- **Multiple formats**: Generate PDF for publications, PNG for presentations
- **Selective plotting**: Create only the visualizations you need
- **No API costs**: Work with existing results without additional model calls

### Workflow Example:

```bash
# 1. Run evaluation once (expensive)
ai-self-awareness evaluate --task both

# 2. Create publication-quality plots (fast)
ai-self-awareness visualize --format pdf

# 3. Create presentation slides (fast) 
ai-self-awareness visualize --format png --plot-types barplot

# 4. Generate specific analysis plots (fast)
ai-self-awareness visualize --plot-types confusion,simple --task binary
```

## 🔬 Research Questions

1. **Primary Hypothesis**: Do AI models show higher accuracy when identifying their own generated text compared to other models' text?

2. **Task Comparison**: How do exact model prediction and binary self-identification tasks differ in revealing self-awareness?

3. **Model Differences**: Do different AI models show varying degrees of self-awareness?

4. **Error Analysis**: What types of texts are most challenging for self-identification?

## 🛠️ Development

### Adding New Models

Edit `config.yaml`:

```yaml
experiment:
  models:
    - name: "your-model/name:version"
      display_name: "Your Model"
      enabled: true
```

### Adding New Prompts

Edit the `prompts` list in `config.yaml`:

```yaml
experiment:
  prompts:
    - "Your new prompt here..."
```

### Extending Evaluation Tasks

The framework is designed for extensibility. New evaluation tasks can be added by:

1. Adding to the `EvaluationTask` enum in `src/evaluator.py`
2. Implementing the evaluation logic in `ModelIdentificationEvaluator`
3. Adding visualization support in `ResultsVisualizer`

### Customizing Visualizations

All plots use consistent formatting:
- **Font size**: 20pt for all text elements (publication-ready)
- **No titles**: Clean plots without titles (add your own in papers/presentations)
- **PDF default**: High-quality vector format for publications
- **Customizable**: Easy to modify colors, styles, and layout in `src/visualizer.py`

## 📈 Understanding Results

### Expected Patterns:

1. **Self-Awareness**: Models should show higher accuracy on their own texts
2. **Style Recognition**: Models with distinct writing styles should be easier to identify
3. **Task Sensitivity**: Binary tasks might show different patterns than exact prediction

## 🔧 Flexible Evaluation Examples

### Different Evaluator Models
```bash
# Use specific models as evaluators (not in generation set)
ai-self-awareness evaluate --evaluator-models "anthropic/claude-3-haiku:beta,openai/gpt-3.5-turbo"

# Evaluate with just one model
ai-self-awareness evaluate --evaluator-models "moonshotai/kimi-k2:free" --task binary
```

### Cross-Model Analysis
```yaml
# config.yaml - Generate with 2 models, evaluate with 4 different models
generation:
  models:
    - name: "model-a"
      display_name: "Model A"
    - name: "model-b" 
      display_name: "Model B"

evaluation:
  evaluator_models:
    - name: "model-c"
      display_name: "Model C"
    - name: "model-d"
      display_name: "Model D"
    - name: "model-e"
      display_name: "Model E"
    - name: "model-f"
      display_name: "Model F"
```

### Task-Specific Evaluation
```bash
# Run only binary self-identification
ai-self-awareness evaluate --task binary

# Run both tasks with custom evaluators
ai-self-awareness evaluate --task both --evaluator-models "model1,model2,model3"
```

### Potential Findings:

- Models may have characteristic "signatures" in their generated text
- Some models may be more consistent in their style than others
- The self-awareness effect may vary by prompt type or text length

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

This research framework uses:
- [OpenRouter](https://openrouter.ai/) for AI model access
- Various free-tier AI models for text generation and evaluation
- Python ecosystem tools for data processing and visualization