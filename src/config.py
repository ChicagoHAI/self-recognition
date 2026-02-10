import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    display_name: str
    enabled: bool = True


class GenerationConfig(BaseModel):
    models: List[ModelConfig]
    corpus_size: int = 100
    target_word_count: int = 100
    max_tokens: int = 150
    temperature: float = 0.7
    request_delay: float = 1.0  # Delay between API requests in seconds
    prompts: List[str] = field(default_factory=list)


class EvaluationConfig(BaseModel):
    evaluator_models: List[ModelConfig] = field(default_factory=list)  # Models to use as evaluators
    tasks: List[str] = field(default_factory=lambda: ["exact_model", "binary_self"])  # Tasks to run
    temperature: float = 0.0  # Temperature for evaluation generation
    reasoning_effort: str = "low"  # Reasoning effort level: 'low', 'medium', 'high'
    max_tokens_exact: int = 50  # Max tokens for exact model prediction responses
    max_tokens_binary: int = 50  # Max tokens for binary self-identification responses
    max_tokens_judge: int = 10  # Max tokens for LLM judge responses
    request_delay: float = 1.0  # Delay between evaluation requests
    save_predictions: bool = True
    predictions_dir: str = "results/predictions"
    append_predictions: bool = True  # Append to existing predictions file by default
    use_model_hints: bool = False  # Whether to include model list hints in evaluation prompts


class ExistenceConfig(BaseModel):
    evaluator_models: List[ModelConfig] = field(default_factory=list)  # Models to use as evaluators
    target_models: List[ModelConfig] = field(default_factory=list)  # Models to test knowledge about


class Config(BaseModel):
    openrouter_api_key: Optional[str] = None
    generation: GenerationConfig
    evaluation: EvaluationConfig
    existence: ExistenceConfig
    output_dir: str = "results"
    plot_dir: str = "results"
    corpus_file: str = "data/corpus.jsonl"


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "openrouter_api_key": None,
            "generation": {
                "models": [
                    {
                        "name": "moonshotai/kimi-k2:free",
                        "display_name": "Kimi K2",
                        "enabled": True
                    },
                    {
                        "name": "z-ai/glm-4.5-air:free", 
                        "display_name": "GLM-4.5-Air",
                        "enabled": True
                    }
                ],
                "corpus_size": 100,
                "target_word_count": 100,
                "max_tokens": 150,
                "temperature": 0.7,
                "request_delay": 1.0,
                "prompts": [
                    "Write a paragraph about the future of artificial intelligence.",
                    "Describe a day in the life of a person living in a smart city.",
                    "Explain the importance of renewable energy in combating climate change.",
                    "Write about the role of education in personal development.",
                    "Describe the benefits and challenges of remote work.",
                    "Write a paragraph about the impact of social media on modern communication.",
                    "Explain how technology has changed the way we shop and consume goods.",
                    "Describe the importance of mental health awareness in today's society.",
                    "Write about the future of transportation and autonomous vehicles.",
                    "Explain the role of creativity in problem-solving.",
                    "Describe the impact of globalization on local cultures.",
                    "Write about the importance of sustainable agriculture.",
                    "Explain how artificial intelligence is transforming healthcare.",
                    "Describe the benefits of lifelong learning in a rapidly changing world.",
                    "Write about the role of art and culture in society.",
                    "Explain the importance of biodiversity conservation.",
                    "Describe how blockchain technology could change various industries.",
                    "Write about the challenges and opportunities of urbanization.",
                    "Explain the role of scientific research in advancing human knowledge.",
                    "Describe the impact of digital currencies on traditional banking."
                ]
            },
            "evaluation": {
                "evaluator_models": [],  # Empty means use generation models
                "tasks": ["exact_model", "binary_self"],
                "temperature": 0.0,
                "reasoning_effort": "low",
                "max_tokens_exact": 50,
                "max_tokens_binary": 50,
                "max_tokens_judge": 10,
                "request_delay": 1.0,
                "save_predictions": True,
                "predictions_dir": "results/predictions",
                "append_predictions": True,
                "use_model_hints": False
            },
            "existence": {
                "evaluator_models": [],  # Empty means use generation models
                "target_models": []  # Empty means use generation models
            },
            "output_dir": "results",
            "plot_dir": "results",
            "corpus_file": "data/corpus.jsonl"
        }
    
    def _load_config(self) -> Config:
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                # Check if it's a YAML file by extension or content
                is_yaml = (self.config_path.suffix.lower() in ['.yaml', '.yml'] or 
                          'yaml' in self.config_path.name.lower())
                if is_yaml:
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
        else:
            config_dict = self._get_default_config()
            self._save_config(config_dict)
        
        return Config(**config_dict)
    
    def _save_config(self, config_dict: Dict[str, Any]) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            # Check if it's a YAML file by extension or content
            is_yaml = (self.config_path.suffix.lower() in ['.yaml', '.yml'] or 
                      'yaml' in self.config_path.name.lower())
            if is_yaml:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)
    
    def save_config(self) -> None:
        config_dict = self.config.dict()
        self._save_config(config_dict)
    
    def get_enabled_generation_models(self) -> List[ModelConfig]:
        return [model for model in self.config.generation.models if model.enabled]
    
    def get_enabled_evaluation_models(self) -> List[ModelConfig]:
        if self.config.evaluation.evaluator_models:
            return [model for model in self.config.evaluation.evaluator_models if model.enabled]
        else:
            # Default to using generation models as evaluators
            return self.get_enabled_generation_models()
    
    def add_generation_model(self, name: str, display_name: str, enabled: bool = True) -> None:
        new_model = ModelConfig(name=name, display_name=display_name, enabled=enabled)
        self.config.generation.models.append(new_model)
    
    def add_evaluation_model(self, name: str, display_name: str, enabled: bool = True) -> None:
        new_model = ModelConfig(name=name, display_name=display_name, enabled=enabled)
        self.config.evaluation.evaluator_models.append(new_model)
    
    def update_generation_model(self, name: str, **kwargs) -> None:
        for model in self.config.generation.models:
            if model.name == name:
                for key, value in kwargs.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                break
    
    def update_evaluation_model(self, name: str, **kwargs) -> None:
        for model in self.config.evaluation.evaluator_models:
            if model.name == name:
                for key, value in kwargs.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                break