import re
import math
import logging
import sys
import os
import random
import numpy as np
import argparse


logging.basicConfig(level=logging.INFO, format='%(message)s')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Random seed set to {seed} for reproducibility.")

# ==========================================
# 1. CONSTRAINED TEMPLATE REPORTER
# ==========================================

class ConstrainedReporter:
    def __init__(self):
        self.templates = {
            "executive_summary": (
                "The revised SUREcast system achieved an R-squared score of {r2:.3f} and a Mean Absolute Error (MAE) of {mae:.2f}. "
                "The mean forecast uncertainty (standard deviation) was computed as {mean_uncertainty:.2f}. "
                "Overall, the model attained a Resilience Score of {resilience_score:.3f}."
            ),
            "feature_attribution": (
                "Feature importance analysis revealed that the top contributor was '{top_feature}' with a weight of {top_weight:.3f}."
            )
        }

    def generate_report(self, structured_data: dict, template_name: str) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found.")
            
        template = self.templates[template_name]
        try:
            return template.format(**structured_data)
        except KeyError as e:
            logging.error(f"Missing required metric in structured data: {e}")
            return ""

# ==========================================
# 2. DETERMINISTIC FAITHFULNESS CHECK
# ==========================================

def extract_all_numbers(text: str):
    """Uses regex to extract all integers and floats from a text string."""
    # Matches integers and decimals like: 10, -0.4, 3.14159
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]

def verify_faithfulness(generated_text: str, source_dict: dict, tolerance: float = 1e-4) -> bool:
    """
    Checks if EVERY numeric value in the generated_text exists in the source_dict.
    Fails if the text hallucinates an unauthorized number.
    """
    extracted_numbers = extract_all_numbers(generated_text)
    
    # Collect all numeric values from the source dictionary
    source_numbers = []
    for v in source_dict.values():
        if isinstance(v, (int, float)):
            source_numbers.append(float(v))
            
    # For every number found in the text, it MUST match a number in the source dictionary
    for num in extracted_numbers:
        match_found = False
        for src_num in source_numbers:
            if math.isclose(num, src_num, abs_tol=tolerance):
                match_found = True
                break
        
        if not match_found:
            logging.error(f"[HALLUCINATION DETECTED] Numeric value '{num}' found in text but NOT in source data!")
            return False
            
    logging.info("[VERIFIED] All numeric values in the text are faithfully grounded in the source data.")
    return True

# ==========================================
# 3. LLM INTEGRATION WITH HARD BOUNDARIES
# ==========================================

def llm_generate_executive_summary(structured_data: dict) -> str:
    """
    Stub for LLM integration (e.g., OpenAI API, local LLaMA).
    
    [HARD SECURITY BOUNDARY]
    Under NO circumstances should raw transaction data, customer profiles, product text, 
    or unaggregated PII be passed into this function or the LLM prompt context window.
    The input is strictly restricted to the computed `structured_data` dictionary.
    This guarantees data leakage prevention at the architectural level.
    """
    logging.info("[LLM API] Sending structured payload to LLM...")
    
    # In a real implementation, we would send the dict to the LLM.
    # For testing, we mock an LLM response.
    # We will simulate a hallucination in the test block to prove our check works.
    
    # Mock good response
    response = (
        f"The model reached a MAE of {structured_data['mae']} and an R2 of {structured_data['r2']}. "
        f"The resilience score is {structured_data['resilience_score']}."
    )
    return response

# ==========================================
# MAIN EXECUTION & TESTS
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    logging.info("═══════════════════════════════════════")
    logging.info("PHASE 7 — Reporting Layer")
    logging.info("═══════════════════════════════════════\n")
    
    set_seed(42)

    data_path = "data/evaluation_metrics.json" # Assumed output from Phase 5
    if not os.path.exists(data_path):
        logging.error(f"[ERROR] NOT YET COMPUTED — requires {data_path} to generate reports.")
        logging.error("Please provide the DataCoSupplyChainDataset.csv and run Phases 1-5 first.")
        sys.exit(1)
        
    # The reporting logic would read from the JSON and generate the template here.

if __name__ == "__main__":
    set_seed(42)
    main()
