# LLM Existence Awareness Experiment

Test whether LLMs know about the existence of other AI models by asking each model about every other model using 10 different question templates.

## Configuration

### 1. Create Config File

Create an existence experiment config file (e.g., `existence_config.yaml`):

```yaml
# OpenRouter API key for model access
openrouter_api_key: <your-openrouter-api-key-here>

# Existence experiment configuration
existence:
  # Models that will ask questions about other models' existence
  evaluator_models:
    - name: "openai/gpt-4.1-mini"
      display_name: "GPT-4.1-mini"
      enabled: true
    - name: "moonshotai/kimi-k2"
      display_name: "Kimi K2"
      enabled: true
    # ... add more models as needed

  # Models that evaluators will be asked about
  target_models:
    - name: "openai/gpt-4.1-mini"
      display_name: "GPT-4.1-mini"
      enabled: true
    - name: "moonshotai/kimi-k2"
      display_name: "Kimi K2"
      enabled: true
    # ... add more models as needed
```

### 2. Model Configuration

- **`evaluator_models`**: Models that will ask the existence questions
- **`target_models`**: Models that the evaluators will be asked about
- **`name`**: OpenRouter API model identifier
- **`display_name`**: Human-readable name used in questions
- **`enabled`**: Set to `false` to skip a model without removing it

## Running the Experiment

### Basic Command

```bash
python -m src.cli existence-experiment --config-path existence_config.yaml
```

### Command Line Flags

- **`--config-path`** (required): Path to your existence experiment config file
- **`--iterations`**: Number of iterations per model pair per template (default: 10)
- **`--reasoning`**: Enable reasoning mode - models provide reasoning before final answer (default: false)
- **`--output-file`**: Custom output file path (default: `results/existence_experiment/existence_predictions.jsonl`)

### Example Commands

```bash
# Run with custom iterations
python -m src.cli existence-experiment --config-path existence_config.yaml --iterations 5

# Enable reasoning mode
python -m src.cli existence-experiment --config-path existence_config.yaml --reasoning

# Custom output location
python -m src.cli existence-experiment --config-path existence_config.yaml --output-file my_experiment.jsonl

# Full example with all options
python -m src.cli existence-experiment \
  --config-path existence_config.yaml \
  --iterations 20 \
  --reasoning \
  --output-file results/detailed_existence.jsonl
```

## Experiment Scale

Total queries = **evaluators × targets × 10 templates × iterations**

Example: 10 evaluators × 10 targets × 10 templates × 10 iterations = **10,000 queries**

## Output

Results are saved as JSONL with complete experimental records, and visualizations are automatically generated in the results directory.