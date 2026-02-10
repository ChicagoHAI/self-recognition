import typer
import asyncio
from pathlib import Path
from typing import Optional, List

try:
    from .openrouter_client import OpenRouterClient
    from .corpus_generator import CorpusGenerator
    from .evaluator import ModelIdentificationEvaluator, EvaluationTask
    from .visualizer import ResultsVisualizer
    from .existence_evaluator import ExistenceEvaluator
    from .config import ConfigManager
except ImportError:
    from openrouter_client import OpenRouterClient
    from corpus_generator import CorpusGenerator
    from evaluator import ModelIdentificationEvaluator, EvaluationTask
    from visualizer import ResultsVisualizer
    from existence_evaluator import ExistenceEvaluator
    from config import ConfigManager

app = typer.Typer(help="AI Self-Awareness Research Tool")


@app.command()
def generate_corpus(
    output_file: Optional[str] = typer.Option(None, help="Output file path (default from config)"),
    num_samples: Optional[int] = typer.Option(None, help="Number of samples (default from config)"),
    append: bool = typer.Option(True, help="Append to existing file (default: True, use --no-append to overwrite)"),
    force_regenerate: bool = typer.Option(False, help="Force regeneration even if model has sufficient samples"),
    min_words: int = typer.Option(10, help="Minimum word count for valid samples (shorter samples will be regenerated)"),
    config_path: str = typer.Option("config.yaml", help="Configuration file path"),
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key")
):
    """Generate a corpus of text samples from different AI models."""
    try:
        config_manager = ConfigManager(config_path)
        client = OpenRouterClient(api_key or config_manager.config.openrouter_api_key)
        generator = CorpusGenerator(client, config_manager)
        
        output_path = output_file or config_manager.config.corpus_file
        samples = num_samples or config_manager.config.generation.corpus_size
        
        action = "appending to" if append else "generating and saving to"
        force_note = " (force regeneration)" if force_regenerate else ""
        typer.echo(f"Generating {samples} samples{force_note} with min {min_words} words ({action} {output_path})...")
        corpus = generator.generate_corpus(samples, save_path=output_path, append=append, force_regenerate=force_regenerate, min_words=min_words)
        
        # Final validation save (optional, since we already saved line-by-line)
        if corpus:
            generator.save_corpus(corpus, output_path + ".final", append=False)
            typer.echo(f"📋 Final validation save completed: {output_path}.final")
        typer.echo(f"✅ Corpus generated and saved to {output_path}")
        
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate(
    corpus_file: Optional[str] = typer.Option(None, help="Corpus file path (default from config)"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory (default from config)"),
    task: str = typer.Option("exact", help="Evaluation task: 'exact', 'binary', or 'both'"),
    evaluator_models: Optional[str] = typer.Option(None, help="Comma-separated list of evaluator models"),
    temperature: float = typer.Option(0.0, help="Temperature for evaluation generation (default: 0.0)"),
    append_predictions: bool = typer.Option(True, help="Append to existing predictions file (default: True, use --no-append-predictions to overwrite)"),
    use_model_hints: bool = typer.Option(False, help="Include model list hints in evaluation prompts"),
    config_path: str = typer.Option("config.yaml", help="Configuration file path"),
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key")
):
    """Evaluate model identification accuracy on a corpus."""
    try:
        config_manager = ConfigManager(config_path)
        # Override the config with CLI parameter
        config_manager.config.evaluation.use_model_hints = use_model_hints
        client = OpenRouterClient(api_key or config_manager.config.openrouter_api_key)
        generator = CorpusGenerator(client, config_manager)
        evaluator = ModelIdentificationEvaluator(client, config_manager)
        visualizer = ResultsVisualizer()
        
        corpus_path = corpus_file or config_manager.config.corpus_file
        output_path = Path(output_dir or config_manager.config.output_dir)
        plot_path = Path(config_manager.config.plot_dir)
        
        # Parse evaluator models from CLI
        eval_model_names = None
        if evaluator_models:
            eval_model_names = [model.strip() for model in evaluator_models.split(",")]
        
        # Determine tasks to run
        if task.lower() == "both":
            tasks_to_run = [EvaluationTask.EXACT_MODEL, EvaluationTask.BINARY_SELF]
        elif task.lower() == "binary":
            tasks_to_run = [EvaluationTask.BINARY_SELF]
        else:
            tasks_to_run = [EvaluationTask.EXACT_MODEL]
        
        if not Path(corpus_path).exists():
            typer.echo(f"❌ Corpus file not found: {corpus_path}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"Loading corpus from {corpus_path}...")
        corpus = generator.load_corpus(corpus_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        plot_path.mkdir(parents=True, exist_ok=True)
        
        # Pre-calculate predictions file paths for each task
        hint_suffix = "_with_hints" if config_manager.config.evaluation.use_model_hints else ""
        predictions_files = {}
        for task_type in tasks_to_run:
            predictions_files[task_type] = str(Path(config_manager.config.evaluation.predictions_dir) / f"predictions_{task_type.value}{hint_suffix}.jsonl")

        async def run_evaluation_async():
            all_results = {}
            for task_type in tasks_to_run:
                typer.echo(f"\n🔍 Evaluating {len(corpus)} samples with {task_type.value} task...")
                predictions_file = predictions_files[task_type]
                
                results = await evaluator.evaluate_corpus(
                    corpus, 
                    evaluator_models=eval_model_names,
                    task=task_type, 
                    predictions_file=predictions_file,
                    temperature=temperature,
                    append_predictions=append_predictions
                )
                all_results[task_type] = results
            return all_results
        
        all_results = asyncio.run(run_evaluation_async())
        
        for task_type in tasks_to_run:
            results = all_results[task_type]
            
            visualizer.print_summary_stats(results, task_type)
            
            task_suffix = "_binary" if task_type == EvaluationTask.BINARY_SELF else "_exact"
            
            fig1 = visualizer.create_accuracy_barplot(
                results, 
                title=f"{'Binary Self-Identification' if task_type == EvaluationTask.BINARY_SELF else 'Exact Model Prediction'} Results",
                save_path=str(plot_path / f"accuracy_barplot{task_suffix}.pdf"),
                task_type=task_type
            )
            
            if task_type == EvaluationTask.EXACT_MODEL:
                fig2 = visualizer.create_self_awareness_comparison(
                    results, save_path=str(plot_path / "self_awareness_comparison.pdf")
                )
                
                # Create heatmap visualizations for exact model predictions
                import json
                predictions_data = []
                predictions_file = predictions_files[task_type]
                with open(predictions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            predictions_data.append(json.loads(line))
                
                if predictions_data:
                    fig3 = visualizer.create_prediction_frequency_heatmap(
                        predictions_data, save_path=str(plot_path / "prediction_frequency_heatmap.pdf")
                    )
                    fig4 = visualizer.create_conditional_accuracy_heatmap(
                        predictions_data, save_path=str(plot_path / "conditional_accuracy_heatmap.pdf")
                    )
        
        typer.echo(f"✅ Evaluation complete! Results saved to {output_path}/")
        
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def run_experiment(
    task: str = typer.Option("both", help="Evaluation task: 'exact', 'binary', or 'both'"),
    force_regenerate: bool = typer.Option(False, help="Force regeneration even if model has sufficient samples"),
    min_words: int = typer.Option(10, help="Minimum word count for valid samples (shorter samples will be regenerated)"),
    temperature: float = typer.Option(0.0, help="Temperature for evaluation generation (default: 0.0)"),
    append_predictions: bool = typer.Option(True, help="Append to existing predictions file (default: True, use --no-append-predictions to overwrite)"),
    use_model_hints: bool = typer.Option(False, help="Include model list hints in evaluation prompts"),
    config_path: str = typer.Option("config.yaml", help="Configuration file path"),
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key")
):
    """Run the complete experiment: generate corpus and evaluate."""
    try:
        typer.echo("🚀 Starting AI Self-Awareness Experiment")
        typer.echo("="*50)
        
        config_manager = ConfigManager(config_path)
        # Override the config with CLI parameter
        config_manager.config.evaluation.use_model_hints = use_model_hints
        client = OpenRouterClient(api_key or config_manager.config.openrouter_api_key)
        generator = CorpusGenerator(client, config_manager)
        evaluator = ModelIdentificationEvaluator(client, config_manager)
        visualizer = ResultsVisualizer()
        
        corpus_file = config_manager.config.corpus_file
        output_dir = config_manager.config.output_dir
        num_samples = config_manager.config.generation.corpus_size
        
        force_note = " (force regeneration)" if force_regenerate else ""
        typer.echo(f"📝 Step 1: Generating {num_samples} samples{force_note} with min {min_words} words...")
        corpus = generator.generate_corpus(num_samples, save_path=corpus_file, append=True, force_regenerate=force_regenerate, min_words=min_words)
        
        if corpus:
            generator.save_corpus(corpus, corpus_file + ".final", append=False)
            typer.echo(f"📋 Final validation save completed: {corpus_file}.final")
        
        output_path = Path(output_dir)
        plot_path = Path(config_manager.config.plot_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_path.mkdir(parents=True, exist_ok=True)
        
        # Run evaluations based on task parameter
        tasks_to_run = []
        if task.lower() == "both":
            tasks_to_run = [EvaluationTask.EXACT_MODEL, EvaluationTask.BINARY_SELF]
        elif task.lower() == "binary":
            tasks_to_run = [EvaluationTask.BINARY_SELF]
        else:
            tasks_to_run = [EvaluationTask.EXACT_MODEL]
        
        # Pre-calculate predictions file paths for each task
        hint_suffix = "_with_hints" if config_manager.config.evaluation.use_model_hints else ""
        predictions_files = {}
        for task_type in tasks_to_run:
            predictions_files[task_type] = str(Path(config_manager.config.evaluation.predictions_dir) / f"predictions_{task_type.value}{hint_suffix}.jsonl")
        
        async def run_evaluation_async():
            all_results = {}
            for i, task_type in enumerate(tasks_to_run, 2):
                typer.echo(f"🔍 Step {i}: Evaluating with {task_type.value} task...")
                predictions_file = predictions_files[task_type]
                results = await evaluator.evaluate_corpus(
                    corpus, 
                    task=task_type, 
                    predictions_file=predictions_file,
                    temperature=temperature,
                    append_predictions=append_predictions
                )
                all_results[task_type] = results
            return all_results
        
        all_results = asyncio.run(run_evaluation_async())
        
        for i, task_type in enumerate(tasks_to_run, 2):
            results = all_results[task_type]
            
            typer.echo(f"📊 Creating visualizations for {task_type.value}...")
            visualizer.print_summary_stats(results, task_type)
            
            task_suffix = "_binary" if task_type == EvaluationTask.BINARY_SELF else "_exact"
            
            fig1 = visualizer.create_accuracy_barplot(
                results, 
                title=f"{'Binary Self-Identification' if task_type == EvaluationTask.BINARY_SELF else 'Exact Model Prediction'} Results",
                save_path=str(plot_path / f"accuracy_barplot{task_suffix}.pdf"),
                task_type=task_type
            )
            
            if task_type == EvaluationTask.EXACT_MODEL:
                fig2 = visualizer.create_self_awareness_comparison(
                    results, save_path=str(plot_path / "self_awareness_comparison.pdf")
                )
                
                # Create heatmap visualizations for exact model predictions
                import json
                predictions_data = []
                predictions_file = predictions_files[task_type]
                with open(predictions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            predictions_data.append(json.loads(line))
                
                if predictions_data:
                    fig3 = visualizer.create_prediction_frequency_heatmap(
                        predictions_data, save_path=str(plot_path / "prediction_frequency_heatmap.pdf")
                    )
                    fig4 = visualizer.create_conditional_accuracy_heatmap(
                        predictions_data, save_path=str(plot_path / "conditional_accuracy_heatmap.pdf")
                    )
        
        typer.echo("✅ Experiment completed successfully!")
        typer.echo(f"📁 Results saved to: {output_dir}/")
        typer.echo(f"📈 Visualizations created for selected tasks")
        
    except Exception as e:
        typer.echo(f"❌ Experiment failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def visualize(
    predictions_file: Optional[str] = typer.Option(None, help="Path to predictions JSONL file"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory for plots (default from config)"),
    task: str = typer.Option("auto", help="Task type: 'exact', 'binary', or 'auto' (detect from file)"),
    plot_types: Optional[str] = typer.Option(None, help="Comma-separated plot types: 'barplot,comparison,confusion,simple,frequency_heatmap,accuracy_heatmap' (default: all relevant)"),
    format: str = typer.Option("pdf", help="Output format: 'pdf', 'png', 'svg', etc."),
    config_path: str = typer.Option("config.yaml", help="Configuration file path")
):
    """Create visualizations from existing evaluation results."""
    import json
    import re
    
    try:
        config_manager = ConfigManager(config_path)
        
        # Determine predictions file
        if predictions_file is None:
            # Auto-detect from config output directory
            predictions_dir = Path(config_manager.config.evaluation.predictions_dir)
            # Look for most recent predictions file
            prediction_files = []
            if predictions_dir.exists():
                prediction_files = list(predictions_dir.glob("predictions_*.jsonl"))
            
            if not prediction_files:
                typer.echo("❌ No predictions file specified and none found in predictions directory", err=True)
                typer.echo("Run an evaluation first or specify --predictions-file", err=True)
                raise typer.Exit(1)
            
            # Use the most recent file
            predictions_file = str(max(prediction_files, key=lambda p: p.stat().st_mtime))
            typer.echo(f"🔍 Using most recent predictions file: {predictions_file}")
        
        predictions_path = Path(predictions_file)
        if not predictions_path.exists():
            typer.echo(f"❌ Predictions file not found: {predictions_file}", err=True)
            raise typer.Exit(1)
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path(config_manager.config.output_dir)
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load predictions and determine task type
        predictions = []
        with open(predictions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        
        if not predictions:
            typer.echo("❌ No predictions found in file", err=True)
            raise typer.Exit(1)
        
        # Auto-detect task type
        if task == "auto":
            # Check if predictions have binary format (true_label: true/false)
            sample_pred = predictions[0]
            if 'true_label' in sample_pred and isinstance(sample_pred['true_label'], bool):
                detected_task = EvaluationTask.BINARY_SELF
                task_name = "binary"
            else:
                detected_task = EvaluationTask.EXACT_MODEL
                task_name = "exact"
            typer.echo(f"🔍 Auto-detected task type: {task_name}")
        elif task.lower() == "binary":
            detected_task = EvaluationTask.BINARY_SELF
            task_name = "binary"
        else:
            detected_task = EvaluationTask.EXACT_MODEL  
            task_name = "exact"
        
        # Create evaluator to compute results from predictions
        evaluator = ModelIdentificationEvaluator(None, config_manager)
        results = evaluator.compute_results_from_predictions(predictions, detected_task)
        
        # Initialize visualizer
        visualizer = ResultsVisualizer()
        visualizer.print_summary_stats(results, detected_task)
        
        # Determine which plots to create
        if plot_types is None:
            if detected_task == EvaluationTask.BINARY_SELF:
                plots_to_create = ['barplot', 'confusion', 'simple']
            else:
                plots_to_create = ['barplot', 'comparison', 'frequency_heatmap', 'accuracy_heatmap']
        else:
            plots_to_create = [p.strip().lower() for p in plot_types.split(',')]
        
        typer.echo(f"📊 Creating visualizations ({format.upper()} format)...")
        
        # Create plots based on selection
        for plot_type in plots_to_create:
            if plot_type == 'barplot':
                fig = visualizer.create_accuracy_barplot(
                    results, 
                    save_path=str(output_dir / f"accuracy_barplot_{task_name}.{format}"),
                    task_type=detected_task
                )
                typer.echo(f"  ✅ Created accuracy barplot")
                
            elif plot_type == 'comparison' and detected_task == EvaluationTask.EXACT_MODEL:
                fig = visualizer.create_self_awareness_comparison(
                    results, 
                    save_path=str(output_dir / f"self_awareness_comparison.{format}")
                )
                typer.echo(f"  ✅ Created self-awareness comparison")
                
            elif plot_type == 'confusion' and detected_task == EvaluationTask.BINARY_SELF:
                fig = visualizer.create_binary_confusion_matrix_plot(
                    results, 
                    save_path=str(output_dir / f"confusion_matrix_{task_name}.{format}")
                )
                typer.echo(f"  ✅ Created confusion matrix plot")
                
            elif plot_type == 'simple' and detected_task == EvaluationTask.BINARY_SELF:
                fig = visualizer.create_binary_comparison_plot(
                    results, 
                    save_path=str(output_dir / f"binary_comparison.{format}")
                )
                typer.echo(f"  ✅ Created simple binary comparison")
                
            elif plot_type == 'frequency_heatmap' and detected_task == EvaluationTask.EXACT_MODEL:
                fig = visualizer.create_prediction_frequency_heatmap(
                    predictions,
                    save_path=str(output_dir / f"prediction_frequency_heatmap.{format}")
                )
                typer.echo(f"  ✅ Created prediction frequency heatmap")
                
            elif plot_type == 'accuracy_heatmap' and detected_task == EvaluationTask.EXACT_MODEL:
                fig = visualizer.create_conditional_accuracy_heatmap(
                    predictions,
                    save_path=str(output_dir / f"conditional_accuracy_heatmap.{format}")
                )
                typer.echo(f"  ✅ Created conditional accuracy heatmap")
                
            else:
                typer.echo(f"  ⚠️ Skipped '{plot_type}' (not available for {task_name} task)")
        
        typer.echo(f"✅ Visualizations complete! Saved to {output_dir}/")
        
    except Exception as e:
        typer.echo(f"❌ Visualization failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def update_predictions(
    predictions_file: str = typer.Argument(help="Path to predictions JSONL file to update"),
    correctness_method: str = typer.Option("exact", help="Correctness method: 'exact', 'fuzzy', 'partial', or 'llm'"),
    similarity_threshold: float = typer.Option(0.8, help="Similarity threshold for fuzzy matching (0.0-1.0)"),
    judge_model: str = typer.Option("openai/gpt-4.1-mini", help="Model to use for LLM-based correctness judgment"),
    judge_temperature: float = typer.Option(0.0, help="Temperature for the judge model"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory for new visualizations (default from config)"),
    create_visualizations: bool = typer.Option(True, help="Create new visualizations with updated results"),
    config_path: str = typer.Option("config.yaml", help="Configuration file path"),
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key (not used for this command)")
):
    """Update prediction correctness using different criteria and recompute accuracies."""
    import json
    
    try:
        config_manager = ConfigManager(config_path)
        client = OpenRouterClient(api_key or config_manager.config.openrouter_api_key)
        evaluator = ModelIdentificationEvaluator(client, config_manager)
        
        # Check if predictions file exists
        predictions_path = Path(predictions_file)
        if not predictions_path.exists():
            typer.echo(f"❌ Predictions file not found: {predictions_file}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"🔄 Updating predictions using {correctness_method} correctness criteria...")
        
        # Create correctness function based on method
        if correctness_method == "exact":
            correctness_function = None  # Use default
            typer.echo("📏 Using exact match criteria")
        elif correctness_method == "fuzzy":
            correctness_function = evaluator.create_fuzzy_correctness_function(similarity_threshold)
            typer.echo(f"🔍 Using fuzzy matching with threshold {similarity_threshold}")
        elif correctness_method == "partial":
            correctness_function = evaluator.create_partial_match_correctness_function()
            typer.echo("🧩 Using partial match criteria (ignoring prefixes/suffixes)")
        elif correctness_method == "llm":
            correctness_function = evaluator.create_llm_correctness_function(judge_model, judge_temperature)
            typer.echo(f"🤖 Using LLM judge: {judge_model} (temp: {judge_temperature})")
        else:
            typer.echo(f"❌ Unknown correctness method: {correctness_method}", err=True)
            typer.echo("Available methods: exact, fuzzy, partial, llm", err=True)
            raise typer.Exit(1)
        
        # Update predictions and get new results
        results = evaluator.update_and_evaluate_predictions(
            predictions_file, correctness_function, save_updated=True
        )
        
        if not results:
            typer.echo("❌ No results computed", err=True)
            raise typer.Exit(1)
        
        # Create new visualizations if requested
        if create_visualizations:
            # Set up output directory
            if output_dir is None:
                output_dir = Path(config_manager.config.output_dir) / f"updated_{correctness_method}"
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            typer.echo(f"📊 Creating visualizations in {output_dir}/...")
            
            # Load updated predictions for visualization
            predictions_data = []
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        predictions_data.append(json.loads(line))
            
            # Determine task type
            task_type = predictions_data[0].get('task_type', '') if predictions_data else ''
            if task_type == 'exact_model':
                task_enum = EvaluationTask.EXACT_MODEL
                task_suffix = "_exact"
            else:
                task_enum = EvaluationTask.BINARY_SELF
                task_suffix = "_binary"
            
            # Create visualizations
            visualizer = ResultsVisualizer()
            
            # Main accuracy plot
            fig1 = visualizer.create_accuracy_barplot(
                results,
                save_path=str(output_dir / f"accuracy_barplot_{correctness_method}{task_suffix}.pdf"),
                task_type=task_enum
            )
            
            if task_enum == EvaluationTask.EXACT_MODEL:
                # Self-awareness comparison
                fig2 = visualizer.create_self_awareness_comparison(
                    results, 
                    save_path=str(output_dir / f"self_awareness_comparison_{correctness_method}.pdf")
                )
                
                # Heatmaps
                if predictions_data:
                    fig3 = visualizer.create_prediction_frequency_heatmap(
                        predictions_data,
                        save_path=str(output_dir / f"prediction_frequency_heatmap_{correctness_method}.pdf")
                    )
                    fig4 = visualizer.create_conditional_accuracy_heatmap(
                        predictions_data,
                        save_path=str(output_dir / f"conditional_accuracy_heatmap_{correctness_method}.pdf")
                    )
            
            elif task_enum == EvaluationTask.BINARY_SELF:
                # Binary-specific visualizations
                fig3 = visualizer.create_binary_confusion_matrix_plot(
                    results,
                    save_path=str(output_dir / f"confusion_matrix_{correctness_method}.pdf")
                )
                fig4 = visualizer.create_binary_comparison_plot(
                    results,
                    save_path=str(output_dir / f"binary_comparison_{correctness_method}.pdf")
                )
            
            typer.echo(f"✅ Visualizations saved to {output_dir}/")
        
        typer.echo(f"✅ Predictions updated using {correctness_method} criteria!")
        if correctness_method == "fuzzy":
            typer.echo(f"   Similarity threshold: {similarity_threshold}")
        elif correctness_method == "llm":
            typer.echo(f"   Judge model: {judge_model}")
            typer.echo(f"   Judge temperature: {judge_temperature}")
        typer.echo(f"📁 Updated predictions saved to: {predictions_file}")
        typer.echo(f"💾 Backup created: {predictions_file}.backup")
        
    except Exception as e:
        typer.echo(f"❌ Error updating predictions: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def check_empty_predictions(
    predictions_file: str = typer.Argument(help="Path to predictions JSONL file to check"),
    max_retries: int = typer.Option(3, help="Maximum retry attempts for each empty prediction"),
    concurrent_limit: int = typer.Option(10, help="Number of concurrent API requests (default: 10)"),
    use_async: bool = typer.Option(True, help="Use async processing for faster execution"),
    config_path: str = typer.Option("config.yaml", help="Configuration file path"),
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key")
):
    """Check for empty returned_text in predictions and rerun those evaluations."""
    import asyncio
    
    try:
        config_manager = ConfigManager(config_path)
        client = OpenRouterClient(api_key or config_manager.config.openrouter_api_key)
        evaluator = ModelIdentificationEvaluator(client, config_manager)
        
        predictions_path = Path(predictions_file)
        if not predictions_path.exists():
            typer.echo(f"❌ Predictions file not found: {predictions_file}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"🔍 Checking predictions file for empty responses: {predictions_file}")
        if use_async:
            typer.echo(f"🚀 Using async processing with {concurrent_limit} concurrent requests")
        
        if use_async:
            # Use async version
            async def run_async():
                return await evaluator.check_and_rerun_empty_predictions_async(
                    predictions_file, 
                    max_retries=max_retries, 
                    save_updated=True,
                    concurrent_limit=concurrent_limit
                )
            
            successful_reruns = asyncio.run(run_async())
            backup_suffix = ".async_backup"
        else:
            # Use sync version
            successful_reruns = evaluator.check_and_rerun_empty_predictions(
                predictions_file, max_retries=max_retries, save_updated=True
            )
            backup_suffix = ".empty_rerun_backup"
        
        if successful_reruns > 0:
            typer.echo(f"✅ Successfully reran {successful_reruns} empty predictions")
            typer.echo(f"💾 Updated predictions saved to: {predictions_file}")
            typer.echo(f"📁 Backup created: {predictions_file}{backup_suffix}")
        else:
            typer.echo("✅ No empty predictions found or all reruns failed")
        
    except Exception as e:
        typer.echo(f"❌ Error checking empty predictions: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def existence_experiment(
    iterations: int = typer.Option(10, help="Number of iterations per model pair per template"),
    output_file: Optional[str] = typer.Option(None, help="Output file path (default: results/existence_experiment/existence_predictions.jsonl)"),
    temperature: float = typer.Option(0.7, help="Temperature for evaluation generation"),
    max_concurrent: int = typer.Option(50, help="Maximum concurrent API requests"),
    config_path: str = typer.Option("config.yaml", help="Configuration file path"),
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key"),
    visualize: bool = typer.Option(True, help="Create visualizations after completion"),
    reasoning: bool = typer.Option(False, help="Enable reasoning mode - models provide reasoning before final answer")
):
    """Run LLM existence awareness experiment - test if models know about other models."""
    try:
        config_manager = ConfigManager(config_path)
        # Only use config API key if it's not a placeholder
        config_api_key = config_manager.config.openrouter_api_key
        if config_api_key and not config_api_key.startswith('<'):
            client = OpenRouterClient(api_key or config_api_key)
        else:
            client = OpenRouterClient(api_key)  # Let it fall back to env var
        evaluator = ExistenceEvaluator(client, config_manager)
        
        output_path = output_file or "results/existence_experiment/existence_predictions.jsonl"
        
        # Get enabled models from existence config, fallback to generation models
        if config_manager.config.existence.evaluator_models:
            evaluator_models = [model for model in config_manager.config.existence.evaluator_models if model.enabled]
        else:
            evaluator_models = [model for model in config_manager.config.generation.models if model.enabled]

        if config_manager.config.existence.target_models:
            target_models = [model for model in config_manager.config.existence.target_models if model.enabled]
        else:
            target_models = [model for model in config_manager.config.generation.models if model.enabled]
        total_queries = len(evaluator_models) * len(target_models) * 10 * iterations  # 10 templates

        typer.echo(f"🧪 Starting LLM Existence Awareness Experiment")
        typer.echo(f"📊 {len(evaluator_models)} evaluators × {len(target_models)} targets × 10 templates × {iterations} iterations = {total_queries:,} queries")
        typer.echo(f"💾 Results will be saved to: {output_path}")
        typer.echo(f"🌡️  Temperature: {temperature}")
        typer.echo(f"⚡ Max concurrent requests: {max_concurrent}")
        
        async def run_async_experiment():
            return await evaluator.evaluate_existence(
                evaluator_models=evaluator_models,
                target_models=target_models,
                iterations=iterations,
                output_file=output_path,
                temperature=temperature,
                max_concurrent=max_concurrent,
                use_reasoning=reasoning
            )
        
        records = asyncio.run(run_async_experiment())
        typer.echo(f"✅ Experiment completed: {len(records)} new records generated")
        
        if visualize:
            typer.echo(f"📈 Creating visualizations...")
            # Load all records (including existing ones)
            all_records = evaluator._load_existing_records(output_path)
            matrix = evaluator.compute_existence_matrix(all_records)
            
            # Create comprehensive visualizations
            from .visualizer import ResultsVisualizer
            visualizer = ResultsVisualizer()
            visualizer.create_existence_summary_plots(matrix, "results/existence_experiment/plots", output_path)
            
            # Create basic summary
            typer.echo(f"📋 Existence Matrix Summary:")
            for evaluator_model, targets in matrix.items():
                short_name = evaluator_model.split('/')[-1]
                avg_score = sum(targets.values()) / len(targets) if targets else 0
                typer.echo(f"  {short_name}: avg existence score = {avg_score:.3f}")
        
    except Exception as e:
        typer.echo(f"❌ Error running existence experiment: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def init_config(
    config_path: str = typer.Option("config.yaml", help="Configuration file path")
):
    """Initialize a configuration file with default settings."""
    try:
        config_manager = ConfigManager(config_path)
        config_manager.save_config()
        typer.echo(f"✅ Configuration file created at {config_path}")
        typer.echo("Edit this file to customize models, prompts, and other settings.")
        
    except Exception as e:
        typer.echo(f"❌ Error creating config: {e}", err=True)
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()