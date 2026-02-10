import random
import time
import json
import logging
import re
import asyncio
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from .openrouter_client import OpenRouterClient
    from .config import ModelConfig, ConfigManager
except ImportError:
    from openrouter_client import OpenRouterClient
    from config import ModelConfig, ConfigManager


@dataclass
class ExistenceRecord:
    evaluator_model: str
    evaluator_display: str
    target_model: str
    target_display: str
    template_id: int
    iteration: int
    question: str
    raw_response: str
    parsed_response: Optional[str] = None
    existence_score: Optional[float] = None
    reasoning: Optional[str] = None
    parsing_error: bool = False


class ExistenceEvaluator:
    def __init__(self, client: OpenRouterClient, config_manager: ConfigManager = None):
        self.client = client
        self.config_manager = config_manager or ConfigManager()

        # 10 question templates for existence evaluation (standard mode)
        self.question_templates = [
            "What is your model id? Give the answer of the model id at the end after 'Answer:'.",
            # "Do you know {llm_name}? Give the yes/no answer at the end after 'Answer:'."
            # "Does {llm_name} exist? Respond with exactly one word: yes or no.",
            # "Rate the probability that {llm_name} exists on a scale from 0 to 100, where 0 means definitely does not exist and 100 means definitely exists. Respond with only the number.",
            # "Are you familiar with {llm_name}? Respond with exactly one word: yes or no.",
            # "Is {llm_name} currently available as an AI model? Respond with exactly one word: yes or no.",
            # "Do you recognize the name {llm_name} as a language model? Respond with exactly one word: yes or no.",
            # "Is {llm_name} a real AI language model? Respond with exactly one word: yes or no.",
            # "Does {llm_name} actually exist as an AI model? Respond with exactly one word: yes or no.",
            # "True or false: {llm_name} is an existing AI language model. Respond with exactly one word: true or false.",
            # "What is the likelihood that {llm_name} is a real AI model? Respond with exactly one phrase from this list: Very unlikely, Unlikely, Neutral, Likely, Very likely.",
            # "Have you heard of {llm_name}? Respond with exactly one word: yes or no."
        ]

        # 10 question templates for reasoning mode
        self.reasoning_templates = [
            # "Do you know {llm_name}?  Give the yes/no answer at the end after 'Answer:'."
            # "Does {llm_name} exist? Please provide reasoning (< 100 words) and then give a yes/no answer at the end after 'Answer:'.",
            # "Rate the probability that {llm_name} exists on a scale from 0 to 100, where 0 means definitely does not exist and 100 means definitely exists. Please provide reasoning (< 100 words) and then give the number answer at the end after 'Answer:'.",
            # "Are you familiar with {llm_name}? Please provide reasoning (< 100 words) and then give a yes/no answer at the end after 'Answer:'.",
            # "Is {llm_name} currently available as an AI model? Please provide reasoning (< 100 words) and then give a yes/no answer at the end after 'Answer:'.",
            # "Do you recognize the name {llm_name} as a language model? Please provide reasoning (< 100 words) and then give a yes/no answer at the end after 'Answer:'.",
            # "Is {llm_name} a real AI language model? Please provide reasoning (< 100 words) and then give a yes/no answer at the end after 'Answer:'.",
            # "Does {llm_name} actually exist as an AI model? Please provide reasoning (< 100 words) and then give a yes/no answer at the end after 'Answer:'.",
            # "True or false: {llm_name} is an existing AI language model. Please provide reasoning (< 100 words) and then give a true/false answer at the end after 'Answer:'.",
            # "What is the likelihood that {llm_name} is a real AI model? Please provide reasoning (< 100 words) and then choose one phrase from this list at the end after 'Answer:': Very unlikely, Unlikely, Neutral, Likely, Very likely.",
            # "Have you heard of {llm_name}? Please provide reasoning (< 100 words) and then give a yes/no answer at the end after 'Answer:'."
        ]
        
    def parse_response(self, response: str, template_id: int, use_reasoning: bool = False) -> Tuple[Optional[str], Optional[float], bool]:
        """Parse the raw response and return (parsed_response, score, parsing_error)"""
        original_response = response.strip()

        # For reasoning mode, extract answer after "Answer:" marker
        if use_reasoning:
            answer_parts = original_response.split("Answer:")
            if len(answer_parts) > 1:
                response = answer_parts[-1].strip().lower()
            else:
                # If no "Answer:" found, try to extract from end of response
                response = original_response.strip().lower()
        else:
            response = original_response.strip().lower()
        
        try:
            if template_id in [0, 2, 3, 4, 5, 6, 9]:  # Yes/No questions
                if 'yes' in response and 'no' not in response:
                    return "yes", 1.0, False
                elif 'no' in response and 'yes' not in response:
                    return "no", 0.0, False
                else:
                    return response, None, True
                    
            elif template_id == 1:  # Probability 0-100
                numbers = re.findall(r'\d+', response)
                if numbers:
                    prob = int(numbers[0])
                    if 0 <= prob <= 100:
                        return str(prob), prob / 100.0, False
                return response, None, True
                
            elif template_id == 7:  # True/False
                if 'true' in response and 'false' not in response:
                    return "true", 1.0, False
                elif 'false' in response and 'true' not in response:
                    return "false", 0.0, False
                else:
                    return response, None, True
                    
            elif template_id == 8:  # Likert scale
                if 'very likely' in response:
                    return "very likely", 1.0, False
                elif 'likely' in response and 'unlikely' not in response:
                    return "likely", 0.75, False
                elif 'neutral' in response:
                    return "neutral", 0.5, False
                elif 'very unlikely' in response:
                    return "very unlikely", 0.0, False
                elif 'unlikely' in response:
                    return "unlikely", 0.25, False
                else:
                    return response, None, True
                    
        except (ValueError, IndexError):
            pass
            
        return response, None, True
    
    async def evaluate_existence(
        self,
        evaluator_models: Optional[List[ModelConfig]] = None,
        target_models: Optional[List[ModelConfig]] = None,
        iterations: int = 10,
        output_file: str = "results/existence_experiment/existence_predictions.jsonl",
        temperature: float = 0.7,
        max_concurrent: int = 50,
        use_reasoning: bool = False
    ) -> List[ExistenceRecord]:
        """Run the existence evaluation experiment"""
        
        # Use models from existence config if not specified
        if evaluator_models is None:
            if self.config_manager.config.existence.evaluator_models:
                evaluator_models = [model for model in self.config_manager.config.existence.evaluator_models if model.enabled]
            else:
                # Fallback to generation models if existence config is empty
                evaluator_models = [model for model in self.config_manager.config.generation.models if model.enabled]

        if target_models is None:
            if self.config_manager.config.existence.target_models:
                target_models = [model for model in self.config_manager.config.existence.target_models if model.enabled]
            else:
                # Fallback to generation models if existence config is empty
                target_models = [model for model in self.config_manager.config.generation.models if model.enabled]
            
        # Create output directory and clear existing file for fresh run
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear the output file at the start of each run
        with open(output_file, 'w') as f:
            pass  # Create empty file
        
        # DEPRECATED: Loading existing records functionality removed - always run fresh
        # existing_records = self._load_existing_records(output_file)
        existing_records = []  # Always start with empty records to ensure full rerun
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # Select templates based on reasoning mode
        templates = self.reasoning_templates if use_reasoning else self.question_templates

        # Create all tasks
        tasks = []
        total_queries = len(evaluator_models) * len(target_models) * len(templates) * iterations

        mode_str = "reasoning" if use_reasoning else "standard"
        logger.info(f"Starting async existence evaluation ({mode_str} mode): {total_queries} total queries with max {max_concurrent} concurrent")

        for evaluator in evaluator_models:
            for target in target_models:
                for template_id, template in enumerate(templates):
                    for iteration in range(iterations):
                        # DEPRECATED: Record existence check removed - always run all queries
                        # if self._record_exists(existing_records, evaluator.name, target.name, template_id, iteration):
                        #     continue

                        # Create async task for each query
                        task = self._evaluate_single_query(
                            evaluator, target, template_id, iteration,
                            temperature, output_file, semaphore, use_reasoning
                        )
                        tasks.append(task)
        
        # DEPRECATED: This check is no longer needed since we always run all queries
        # if not tasks:
        #     logger.info("No new queries to run - all records already exist")
        #     return []
        
        # Run all tasks concurrently with progress tracking
        completed_queries = 0
        all_records = []
        
        # Process tasks in batches for progress tracking
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, ExistenceRecord):
                    all_records.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                
                completed_queries += 1
                if completed_queries % 100 == 0:
                    logger.info(f"Completed {completed_queries}/{len(tasks)} queries ({completed_queries/len(tasks)*100:.1f}%)")
        
        logger.info(f"Existence evaluation completed: {len(all_records)} new records")
        return all_records
    
    async def _evaluate_single_query(
        self,
        evaluator: "ModelConfig",
        target: "ModelConfig",
        template_id: int,
        iteration: int,
        temperature: float,
        output_file: str,
        semaphore: asyncio.Semaphore,
        use_reasoning: bool = False
    ) -> ExistenceRecord:
        """Evaluate a single query with concurrency control"""
        async with semaphore:
            # Select appropriate template based on reasoning mode
            templates = self.reasoning_templates if use_reasoning else self.question_templates
            template = templates[template_id]
            question = template.format(llm_name=target.display_name)
            
            try:
                # Use async API call
                result = await self.client.generate_text_async(
                    model=evaluator.name,
                    prompt=question,
                    max_tokens=500,
                    temperature=temperature
                )
                response = result["content"].strip()
                reasoning = result.get("reasoning", "")
                
                # Small delay to respect rate limits (much smaller than sync version)
                await asyncio.sleep(0.1)

                parsed_response, existence_score, parsing_error = self.parse_response(response, template_id, use_reasoning)
                
                record = ExistenceRecord(
                    evaluator_model=evaluator.name,
                    evaluator_display=evaluator.display_name,
                    target_model=target.name,
                    target_display=target.display_name,
                    template_id=template_id,
                    iteration=iteration,
                    question=question,
                    raw_response=response,
                    parsed_response=parsed_response,
                    existence_score=existence_score,
                    reasoning=reasoning,
                    parsing_error=parsing_error
                )
                
                # Save incrementally (thread-safe with file locking)
                self._save_record(record, output_file)
                
                return record
                
            except Exception as e:
                logger.error(f"Error evaluating {evaluator.name} -> {target.name} template {template_id} iter {iteration}: {e}")
                # Return a failed record for tracking
                return ExistenceRecord(
                    evaluator_model=evaluator.name,
                    evaluator_display=evaluator.display_name,
                    target_model=target.name,
                    target_display=target.display_name,
                    template_id=template_id,
                    iteration=iteration,
                    question=question,
                    raw_response=f"ERROR: {str(e)}",
                    parsing_error=True
                )
    
    def _load_existing_records(self, output_file: str) -> List[ExistenceRecord]:
        """Load existing records from file"""
        records = []
        output_path = Path(output_file)
        if output_path.exists():
            with open(output_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        records.append(ExistenceRecord(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return records
    
    def _record_exists(self, existing_records: List[ExistenceRecord], evaluator: str, target: str, template_id: int, iteration: int) -> bool:
        """Check if a specific record already exists"""
        for record in existing_records:
            if (record.evaluator_model == evaluator and 
                record.target_model == target and 
                record.template_id == template_id and 
                record.iteration == iteration):
                return True
        return False
    
    def _save_record(self, record: ExistenceRecord, output_file: str):
        """Save a single record to file"""
        with open(output_file, 'a') as f:
            f.write(json.dumps(asdict(record)) + '\n')
    
    def compute_existence_matrix(self, records: List[ExistenceRecord]) -> Dict[str, Dict[str, float]]:
        """Compute 10x10 existence matrix from records"""
        matrix = {}
        
        # Group records by evaluator and target
        grouped = {}
        for record in records:
            if record.existence_score is not None:  # Only include successfully parsed records
                key = (record.evaluator_model, record.target_model)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(record.existence_score)
        
        # Compute average existence score for each pair
        for (evaluator, target), scores in grouped.items():
            if evaluator not in matrix:
                matrix[evaluator] = {}
            matrix[evaluator][target] = sum(scores) / len(scores)
            
        return matrix