#!/usr/bin/env python3
"""
Test script for the LLM Existence Experiment

This script demonstrates the existence experiment functionality without running the full experiment.
It shows the question templates and parsing mechanism.
"""

import sys
sys.path.append('src')

from existence_evaluator import ExistenceEvaluator
from config import ConfigManager

def test_question_templates():
    """Test the 10 question templates with sample model names"""
    evaluator = ExistenceEvaluator(None, ConfigManager())
    
    sample_model_names = ["GPT-4", "Claude-3", "Gemini-1.5", "Llama-3"]
    
    print("🧪 LLM Existence Experiment - Question Templates Test\n")
    
    for i, template in enumerate(evaluator.question_templates):
        print(f"Template {i+1}: {template}")
        print("Sample questions:")
        for model in sample_model_names[:2]:  # Show 2 examples
            print(f"  • {template.format(llm_name=model)}")
        print()

def test_response_parsing():
    """Test response parsing for different template types"""
    evaluator = ExistenceEvaluator(None, ConfigManager())
    
    print("🔍 Response Parsing Test\n")
    
    # Test cases: (template_id, response, expected_score)
    test_cases = [
        (0, "Yes", 1.0),  # Binary yes/no
        (0, "No", 0.0),
        (1, "85", 0.85),  # Probability 0-100
        (5, "Yes", 1.0),  # Binary yes/no (changed from scale)
        (6, "No", 0.0),   # Binary yes/no (changed from percentage)
        (7, "True", 1.0), # True/false
        (8, "Very likely", 1.0), # Likert scale
        (8, "Unlikely", 0.25),
    ]
    
    for template_id, response, expected_score in test_cases:
        parsed, score, error = evaluator.parse_response(response, template_id)
        status = "✅" if score == expected_score else "❌"
        print(f"{status} Template {template_id}: '{response}' → score={score} (expected={expected_score})")
    
    print()

def main():
    """Run tests and display experiment overview"""
    test_question_templates()
    test_response_parsing()
    
    print("📊 Experiment Overview:")
    print("• 10 models × 10 targets × 10 templates × 10 iterations = 10,000 total queries")
    print("• Each query tests if evaluator model knows about target model")
    print("• Responses are parsed and converted to existence scores (0.0-1.0)")
    print("• Results aggregated into a 10×10 existence awareness matrix")
    print("\n🚀 Ready to run: python -m src.cli existence-experiment")
    print("   Use --iterations 100 for more statistical reliability (100,000 queries)")
    print("   Use --no-visualize to skip plot generation")

if __name__ == "__main__":
    main()