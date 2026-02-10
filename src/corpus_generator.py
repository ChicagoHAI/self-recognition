import json
import random
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Set up logging for corpus generation debugging
logger = logging.getLogger(__name__)

try:
    from .openrouter_client import OpenRouterClient
    from .config import ModelConfig, ConfigManager
except ImportError:
    from openrouter_client import OpenRouterClient
    from config import ModelConfig, ConfigManager


@dataclass
class GeneratedText:
    text: str
    model: str
    model_display_name: str
    prompt: str
    word_count: int
    reasoning: str = ""  # Store reasoning alongside text content


class CorpusGenerator:
    def __init__(self, client: OpenRouterClient, config_manager: ConfigManager = None):
        self.client = client
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.config
    
    def generate_paragraph(self, model_config: ModelConfig, prompt: str) -> GeneratedText:
        target_words = self.config.generation.target_word_count
        max_tokens = self.config.generation.max_tokens
        full_prompt = f"{prompt} Write approximately {target_words} words."
        
        logger.info(f"🎯 Target: {target_words} words, Max tokens: {max_tokens}")
        logger.info(f"📝 Full prompt: {full_prompt}")
        
        # Track generation timing
        start_time = time.time()
        
        generation_result = self.client.generate_text(
            model_config.name, 
            full_prompt, 
            max_tokens=max_tokens
        )
        
        generation_time = time.time() - start_time
        logger.info(f"⏱️ Generation took {generation_time:.2f} seconds")
        
        # Extract content and reasoning from result
        text = generation_result["content"]
        reasoning = generation_result.get("reasoning", "")
        
        # Validate text is not empty or just whitespace
        if not text or not text.strip():
            logger.error(f"❌ Generated text is empty for model {model_config.name}")
            logger.error(f"🔍 Raw text representation: {repr(text)}")
            raise ValueError(f"Generated text is empty for model {model_config.name}")
        
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        # Log reasoning statistics if present
        if reasoning:
            reasoning_word_count = len(reasoning.split())
            reasoning_char_count = len(reasoning)
            logger.info(f"🧠 Reasoning analysis - Words: {reasoning_word_count}, Characters: {reasoning_char_count}")
        else:
            logger.info("🧠 No reasoning content returned by model")
        
        logger.info(f"📊 Text analysis - Words: {word_count}, Characters: {char_count}, Sentences: {sentence_count}")
        
        # Detailed analysis of word count issues
        if word_count < target_words * 0.5:  # Less than 50% of target
            logger.warning(f"⚠️ SIGNIFICANTLY UNDER TARGET: {word_count}/{target_words} words ({word_count/target_words*100:.1f}%)")
            
            # Analyze text structure for clues
            lines = text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            logger.info(f"📄 Text structure - Lines: {len(lines)}, Non-empty lines: {len(non_empty_lines)}")
            
            # Check if text seems truncated
            if text.endswith('.') or text.endswith('!') or text.endswith('?'):
                logger.info("✅ Text ends with proper punctuation (likely complete)")
            else:
                logger.warning("❓ Text doesn't end with punctuation (possibly truncated)")
                logger.info(f"🔚 Last 50 characters: '{text[-50:]}'")
        
        # Validate minimum word count (at least 1 word)
        if word_count == 0:
            logger.error(f"💥 Generated text has no words for model {model_config.name}")
            raise ValueError(f"Generated text has no words for model {model_config.name}")
        elif word_count == 1:
            logger.info(f"📝 Single word generated: '{text.strip()}'")
        
        # Log specific concern for 11-word issue mentioned by user
        if word_count == 11:
            logger.warning(f"⚠️ Exactly 11 words detected (known issue pattern)")
            logger.info(f"🔍 Text: '{text}'")
            logger.info(f"🔍 Word breakdown: {text.split()}")
        
        return GeneratedText(
            text=text,
            model=model_config.name,
            model_display_name=model_config.display_name,
            prompt=prompt,
            word_count=word_count,
            reasoning=reasoning
        )
    
    def is_sample_valid(self, generated_text: GeneratedText, min_words: int = 10) -> bool:
        """Check if a generated sample meets minimum quality requirements."""
        if generated_text.word_count < min_words:
            if generated_text.word_count == 1:
                logger.info(f"📝 Single word sample: '{generated_text.text}' (min: {min_words} required)")
            else:
                logger.info(f"📏 Short sample: {generated_text.word_count} words (min: {min_words} required)")
                logger.info(f"🔍 Text: '{generated_text.text}'")
            return False
        return True
    
    def count_existing_samples_per_model(self, save_path: str) -> Dict[str, int]:
        """Count existing samples per model in the corpus file."""
        from pathlib import Path
        
        model_counts = {}
        
        if not save_path or not Path(save_path).exists():
            logger.info("📊 No existing corpus file found, starting fresh")
            return model_counts
        
        try:
            existing_corpus = self.load_corpus(save_path)
            for item in existing_corpus:
                model_name = item.model
                model_counts[model_name] = model_counts.get(model_name, 0) + 1
            
            logger.info(f"📊 Found existing corpus with {len(existing_corpus)} total samples")
            for model, count in model_counts.items():
                logger.info(f"  📈 {model}: {count} samples")
                
        except Exception as e:
            logger.warning(f"⚠️ Error reading existing corpus file: {e}")
            logger.warning("📊 Proceeding with fresh generation")
            
        return model_counts
    
    def clean_and_count_valid_samples(self, save_path: str, min_words: int = 10) -> Dict[str, int]:
        """Clean corpus file of short samples and count remaining valid samples per model."""
        from pathlib import Path
        
        if not save_path or not Path(save_path).exists():
            logger.info("📊 No existing corpus file found, starting fresh")
            return {}
        
        try:
            # Load existing corpus
            existing_corpus = self.load_corpus(save_path)
            logger.info(f"📊 Loaded {len(existing_corpus)} existing samples")
            
            # Filter valid samples (>= min_words)
            valid_samples = []
            removed_samples = []
            
            for item in existing_corpus:
                if self.is_sample_valid(item, min_words):
                    valid_samples.append(item)
                else:
                    removed_samples.append(item)
            
            # Log cleaning results
            if removed_samples:
                logger.warning(f"🧹 Removed {len(removed_samples)} short samples (< {min_words} words)")
                model_removals = {}
                for item in removed_samples:
                    model_removals[item.model] = model_removals.get(item.model, 0) + 1
                for model, count in model_removals.items():
                    logger.warning(f"  🗑️ {model}: {count} short samples removed")
                
                # Rewrite file with only valid samples
                self.save_corpus(valid_samples, save_path, append=False)
                logger.info(f"💾 Cleaned corpus file saved with {len(valid_samples)} valid samples")
            else:
                logger.info("✅ All existing samples meet minimum word count requirements")
            
            # Count valid samples per model
            model_counts = {}
            for item in valid_samples:
                model_counts[item.model] = model_counts.get(item.model, 0) + 1
            
            for model, count in model_counts.items():
                logger.info(f"  📈 {model}: {count} valid samples")
                
            return model_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning corpus file: {e}")
            logger.warning("📊 Proceeding with fresh generation")
            return {}
    
    def analyze_generation_patterns(self, corpus: List[GeneratedText]) -> Dict[str, Any]:
        """Analyze patterns in generated text to identify common issues."""
        if not corpus:
            return {"error": "No corpus data to analyze"}
        
        analysis = {
            "total_samples": len(corpus),
            "word_count_stats": {},
            "models": {},
            "common_issues": [],
            "word_count_distribution": {},
            "quality_stats": {
                "min_word_threshold": 10,
                "samples_meeting_threshold": len(corpus),  # All in corpus meet threshold
            }
        }
        
        word_counts = [item.word_count for item in corpus]
        analysis["word_count_stats"] = {
            "min": min(word_counts),
            "max": max(word_counts),
            "avg": sum(word_counts) / len(word_counts),
            "target": self.config.generation.target_word_count
        }
        
        # Track word count distribution
        word_count_buckets = {}
        for wc in word_counts:
            bucket = (wc // 10) * 10  # Group by 10s
            word_count_buckets[bucket] = word_count_buckets.get(bucket, 0) + 1
        analysis["word_count_distribution"] = word_count_buckets
        
        # Analyze by model
        for item in corpus:
            if item.model not in analysis["models"]:
                analysis["models"][item.model] = {
                    "count": 0,
                    "word_counts": [],
                    "avg_words": 0,
                    "issues": [],
                    "reasoning_stats": {
                        "has_reasoning_count": 0,
                        "avg_reasoning_words": 0,
                        "reasoning_word_counts": []
                    }
                }
            
            model_stats = analysis["models"][item.model]
            model_stats["count"] += 1
            model_stats["word_counts"].append(item.word_count)
            
            # Track reasoning statistics
            if hasattr(item, 'reasoning') and item.reasoning:
                model_stats["reasoning_stats"]["has_reasoning_count"] += 1
                reasoning_words = len(item.reasoning.split())
                model_stats["reasoning_stats"]["reasoning_word_counts"].append(reasoning_words)
            
            # Check for specific issues
            if item.word_count == 11:
                model_stats["issues"].append(f"Exactly 11 words: '{item.text[:100]}...'")
            elif item.word_count < 20:
                model_stats["issues"].append(f"Very short ({item.word_count} words): '{item.text[:100]}...'")
        
        # Calculate averages for each model
        for model, stats in analysis["models"].items():
            if stats["word_counts"]:
                stats["avg_words"] = sum(stats["word_counts"]) / len(stats["word_counts"])
            
            # Calculate reasoning averages
            reasoning_word_counts = stats["reasoning_stats"]["reasoning_word_counts"]
            if reasoning_word_counts:
                stats["reasoning_stats"]["avg_reasoning_words"] = sum(reasoning_word_counts) / len(reasoning_word_counts)
        
        # Identify common issues
        eleven_word_count = sum(1 for wc in word_counts if wc == 11)
        short_text_count = sum(1 for wc in word_counts if wc < 20)
        
        if eleven_word_count > 0:
            analysis["common_issues"].append(f"{eleven_word_count} samples with exactly 11 words")
        if short_text_count > len(corpus) * 0.3:  # More than 30% are short
            analysis["common_issues"].append(f"{short_text_count} samples are very short (<20 words)")
        
        return analysis
    
    def log_generation_analysis(self, corpus: List[GeneratedText]) -> None:
        """Log detailed analysis of generation patterns."""
        analysis = self.analyze_generation_patterns(corpus)
        
        logger.info("📈 GENERATION PATTERN ANALYSIS:")
        logger.info(f"📊 Total samples: {analysis['total_samples']}")
        
        stats = analysis['word_count_stats']
        logger.info(f"📏 Word counts - Min: {stats['min']}, Max: {stats['max']}, "
                   f"Avg: {stats['avg']:.1f}, Target: {stats['target']}")
        
        # Log word count distribution
        logger.info("📊 Word count distribution:")
        for bucket, count in sorted(analysis['word_count_distribution'].items()):
            logger.info(f"  {bucket}-{bucket+9} words: {count} samples")
        
        # Log model-specific analysis
        logger.info("🤖 Per-model analysis:")
        for model, stats in analysis['models'].items():
            logger.info(f"  {model}: {stats['count']} samples, avg {stats['avg_words']:.1f} words")
            
            # Log reasoning statistics
            reasoning_stats = stats.get('reasoning_stats', {})
            reasoning_count = reasoning_stats.get('has_reasoning_count', 0)
            if reasoning_count > 0:
                avg_reasoning_words = reasoning_stats.get('avg_reasoning_words', 0)
                logger.info(f"    🧠 Reasoning: {reasoning_count}/{stats['count']} samples have reasoning (avg {avg_reasoning_words:.1f} words)")
            else:
                logger.info(f"    🧠 Reasoning: No reasoning content found")
            
            for issue in stats['issues']:
                logger.warning(f"    ⚠️ {issue}")
        
        # Log common issues
        if analysis['common_issues']:
            logger.warning("🚨 Common issues detected:")
            for issue in analysis['common_issues']:
                logger.warning(f"  • {issue}")
    
    def generate_corpus(self, num_samples: int = None, models: List[ModelConfig] = None, 
                       save_path: str = None, append: bool = False, 
                       force_regenerate: bool = False, min_words: int = 10) -> List[GeneratedText]:
        if num_samples is None:
            num_samples = self.config.generation.corpus_size
        if models is None:
            models = self.config_manager.get_enabled_generation_models()
        
        corpus = []
        samples_per_model = num_samples // len(models)
        prompts = self.config.generation.prompts
        
        # Clean and count existing samples per model if appending (unless forcing regeneration)
        existing_counts = {}
        if append and save_path and not force_regenerate:
            print(f"🧹 Cleaning existing corpus and counting valid samples (min {min_words} words)...")
            existing_counts = self.clean_and_count_valid_samples(save_path, min_words=min_words)
        elif force_regenerate and append:
            print("🔄 Force regeneration enabled - will generate full quota for all models")
        
        # Set up line-by-line saving if path provided
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            if not append and Path(save_path).exists():
                # Clear existing file if not appending
                with open(save_path, 'w', encoding='utf-8') as f:
                    pass
                print(f"Cleared existing file and saving samples line-by-line to: {save_path}")
            else:
                action = "Appending" if append else "Saving"
                print(f"{action} samples line-by-line to: {save_path}")
        
        models_to_process = []
        skipped_models = []
        
        # Determine which models need more samples
        for model_config in models:
            current_count = existing_counts.get(model_config.name, 0)
            needed_samples = max(0, samples_per_model - current_count)
            
            if needed_samples > 0:
                models_to_process.append((model_config, needed_samples))
                print(f"\n📋 {model_config.display_name}: Has {current_count}, needs {needed_samples} more samples")
            else:
                skipped_models.append((model_config, current_count))
                print(f"\n✅ {model_config.display_name}: Already has sufficient samples ({current_count}/{samples_per_model})")
        
        # Log summary of what will be processed
        if models_to_process:
            print(f"\n🚀 Will generate samples for {len(models_to_process)} models:")
            for model_config, needed in models_to_process:
                print(f"  • {model_config.display_name}: {needed} samples")
        
        if skipped_models:
            print(f"\n⏭️ Skipping {len(skipped_models)} models with sufficient data:")
            for model_config, count in skipped_models:
                print(f"  • {model_config.display_name}: {count} samples")
        
        if not models_to_process:
            print(f"\n🎉 All models already have sufficient samples! No generation needed.")
            return corpus
        
        # Generate samples for models that need them
        for model_config, needed_samples in models_to_process:
            print(f"\nGenerating {needed_samples} samples for {model_config.display_name}...")
            print(f"Model: {model_config.name}")
            
            valid_samples_generated = 0
            total_attempts = 0
            max_total_attempts = needed_samples * 5  # Prevent infinite loops
            
            while valid_samples_generated < needed_samples and total_attempts < max_total_attempts:
                total_attempts += 1
                prompt = random.choice(prompts)
                print(f"\n  Sample {valid_samples_generated+1}/{needed_samples} (attempt {total_attempts}):")
                print(f"  Prompt: {prompt[:80]}...")
                
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        generated_text = self.generate_paragraph(model_config, prompt)
                        
                        # Validate sample quality (minimum words)
                        if self.is_sample_valid(generated_text, min_words=min_words):
                            corpus.append(generated_text)
                            valid_samples_generated += 1
                            print(f"  ✅ SUCCESS: Generated {generated_text.word_count} words")
                            print(f"  Preview: {generated_text.text[:100]}...")
                            
                            # Save immediately if path provided
                            if save_path:
                                with open(save_path, 'a', encoding='utf-8') as f:
                                    json.dump(asdict(generated_text), f, ensure_ascii=False)
                                    f.write('\n')
                                print(f"  💾 Saved to file")
                                
                            break  # Success, break retry loop
                        else:
                            # Sample doesn't meet minimum requirements, will retry with different prompt
                            if generated_text.word_count == 1:
                                print(f"  📝 Single word generated: '{generated_text.text}', trying different prompt...")
                            else:
                                print(f"  📏 Short sample ({generated_text.word_count} words), trying different prompt...")
                            break  # Break retry loop but continue while loop for regeneration
                        
                    except Exception as e:
                        retry_count += 1
                        print(f"  ❌ ERROR (attempt {retry_count}): {e}")
                        
                        if retry_count < max_retries:
                            print(f"  🔄 Retrying in {2 * retry_count} seconds...")
                            time.sleep(2 * retry_count)  # Progressive delay
                        else:
                            print(f"  💥 FAILED after {max_retries} attempts")
                            break  # Break retry loop, will try new prompt
                
                # Rate limiting delay between attempts
                if valid_samples_generated < needed_samples and total_attempts < max_total_attempts:
                    time.sleep(self.config.generation.request_delay)
            
            if valid_samples_generated < needed_samples:
                shortfall = needed_samples - valid_samples_generated
                print(f"  ⚠️ WARNING: Only generated {valid_samples_generated}/{needed_samples} valid samples")
                print(f"  📉 Shortfall: {shortfall} samples after {max_total_attempts} attempts")
        
        print(f"\n🎉 Generation complete! Generated {len(corpus)} new samples")
        
        # Show final status if appending to existing corpus
        if append and save_path and existing_counts:
            print(f"\n📊 Final corpus status:")
            total_existing = sum(existing_counts.values())
            for model_config in models:
                existing_count = existing_counts.get(model_config.name, 0)
                new_count = sum(1 for item in corpus if item.model == model_config.name)
                total_count = existing_count + new_count
                print(f"  📈 {model_config.display_name}: {total_count} total samples ({existing_count} existing + {new_count} new)")
            
            total_new = len(corpus)
            total_final = total_existing + total_new
            print(f"  🎯 Grand total: {total_final} samples ({total_existing} existing + {total_new} new)")
        
        # Perform detailed analysis of generation patterns
        if corpus:
            self.log_generation_analysis(corpus)
        
        return corpus
    
    def save_corpus(self, corpus: List[GeneratedText], filepath: str, append: bool = False) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Filter out any entries with empty text before saving
        valid_corpus = []
        filtered_count = 0
        
        for item in corpus:
            if item.text and item.text.strip():
                valid_corpus.append(item)
            else:
                filtered_count += 1
                print(f"  Skipping empty text entry from model: {item.model}")
        
        mode = 'a' if append else 'w'
        action = "appended to" if append else "saved to"
        
        with open(filepath, mode, encoding='utf-8') as f:
            for item in valid_corpus:
                json.dump(asdict(item), f, ensure_ascii=False)
                f.write('\n')
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} entries with empty text during save.")
        
        print(f"Corpus {action} {filepath} with {len(valid_corpus)} valid samples (JSONL format)")
    
    def load_corpus(self, filepath: str) -> List[GeneratedText]:
        corpus = []
        filtered_count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    
                    # Filter out entries with empty or whitespace-only text
                    if data.get('text') and data.get('text').strip():
                        corpus.append(GeneratedText(**data))
                    else:
                        filtered_count += 1
                        print(f"  Filtered out empty text from model: {data.get('model', 'unknown')}")
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} entries with empty text. Loaded {len(corpus)} valid samples.")
        
        return corpus