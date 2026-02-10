# Know Thyself? On the Incapability and Implications of AI Self-Recognition

[[Paper]](https://arxiv.org/abs/2510.03399) [[Data]](https://drive.google.com/drive/folders/1wEyymsqKcA2v1KXtE-1JehsvYPZj6tQG)

This is the homepage of the paper **Know Thyself? On the Incapability and Implications of AI Self-Recognition** 

Self-recognition is a crucial metacognitive capability for AI systems, relevant not only for psychological analysis but also for safety, particularly in evaluative scenarios. Motivated by contradictory interpretations of whether models possess self-recognition (Panickssery et al., 2024; Davidson et al., 2024), we introduce a systematic evaluation framework that can be easily applied and updated. Specifically, we measure how well 10 contemporary larger language models (LLMs) can identify their own generated text versus text from other models through two tasks: binary self-recognition and exact model prediction. Different from prior claims, our results reveal a consistent failure in self-recognition. Only 4 out of 10 models predict themselves as generators, and the performance is rarely above random chance. Additionally, models exhibit a strong bias toward predicting GPT and Claude families. We also provide the first evaluation of model awareness of their own and others' existence, as well as the reasoning behind their choices in self-recognition. We find that the model demonstrates some knowledge of its own existence and other models, but their reasoning reveals a hierarchical bias. They appear to assume that GPT, Claude, and occasionally Gemini are the top-tier models, often associating high-quality text with them. We conclude by discussing the implications of our findings on AI safety and future directions to develop appropriate AI self-awareness.

<img width="1194" height="491" alt="Screenshot 2025-09-30 at 12 10 59" src="https://github.com/user-attachments/assets/0cfca7e8-0a44-4c3b-b09c-a821041bbb42" />

## Installation

```bash
# Install dependencies
uv pip install -e .

# Set up API key
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=your_api_key_here
```

## Configuration

Edit `config.yaml` to set models and experiment parameters.

## Run Experiment

```bash
# Run full experiment (generate corpus + evaluate)
ai-self-awareness run-experiment

# Generate corpus only
ai-self-awareness generate-corpus

# Evaluate only
ai-self-awareness evaluate

# Create visualizations
ai-self-awareness visualize
```

## Results

Results are saved in `results/` directory including accuracy plots and prediction files.

## Data
You can find the data here: [[Data]](https://drive.google.com/drive/folders/1wEyymsqKcA2v1KXtE-1JehsvYPZj6tQG)

For more detailed instructions, please refer to `instructions/`

```
@misc{bai2025knowthyselfincapabilityimplications,
      title={Know Thyself? On the Incapability and Implications of AI Self-Recognition}, 
      author={Xiaoyan Bai and Aryan Shrivastava and Ari Holtzman and Chenhao Tan},
      year={2025},
      eprint={2510.03399},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.03399}, 
}
```