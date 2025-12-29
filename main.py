import os
import logging
import json
import pandas as pd
from spec_parser import SpecParser
from anchor_generator import AnchorGenerator
from generator import T5GemmaGenerator
from validator import Validator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Clinical ADaM Synthesis Engine...")

    # 1. Parse Specifications
    parser = SpecParser(spec_dir="specs")
    specs = {}
    if not os.path.exists("specs"):
        logger.error("No specs directory found. Run create_test_spec.py first.")
        return

    for filename in os.listdir("specs"):
        if filename.endswith(".xlsx"):
            logger.info(f"Parsing {filename}...")
            context = parser.parse_excel_spec(filename)
            if context:
                specs[context['dataset']] = context
    
    if not specs:
        logger.error("No specs parsed.")
        return

    # 2. Generate Anchors
    logger.info("Generating Subject Anchors...")
    anchor_gen = AnchorGenerator(seed=42)
    anchors = anchor_gen.generate_anchors(n_subjects=10)
    anchor_gen.save_anchors(anchors)
    logger.info(f"Generated {len(anchors)} subjects.")

    # 3. Initialize Generator
    # Note: On Mac, load_in_4bit will likely be disabled by the class itself
    generator = T5GemmaGenerator(model_id="google/t5gemma-2-270m", load_in_4bit=True)

    # 4. Sequential Generation
    # Order: ADSL -> Others (simplified, usually BDS depends on ADSL, OCCDS on ADSL)
    # The prompt architecture creates rows per subject using Anchor + Spec.
    # We can generate them independently if the Anchor has all needed keys (USUBJID, Dates).
    
    generation_order = ['ADSL', 'ADLB', 'ADAE'] # Example order
    
    generated_data = {} # dataset_name -> DataFrame

    for dataset_name in generation_order:
        if dataset_name not in specs:
            logger.warning(f"Spec for {dataset_name} not found. Skipping.")
            continue
            
        spec = specs[dataset_name]
        logger.info(f"Generating {dataset_name}...")
        
        # Pass full anchor context. 
        # If dataset is BDS (ADLB), we might need ADSL info? 
        # For this PoC, Anchor contains key dates (TRTSDT etc) so it's self-sufficient for basic logic.
        
        rows = generator.generate_dataset(anchors, spec, output_path=f"output/{dataset_name}.json")
        
        if rows:
            df = pd.DataFrame(rows)
            generated_data[dataset_name] = df
            
            # Save CSV
            csv_path = f"output/{dataset_name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {dataset_name} to {csv_path}")
            
            # 5. Validation
            logger.info(f"Validating {dataset_name}...")
            integrity_issues = Validator.validate_integrity(df, spec, dataset_name)
            cdisc_issues = Validator.validate_cdisc(df, spec)
            
            if integrity_issues:
                logger.warning(f"Integrity Issues in {dataset_name}: {integrity_issues}")
            if cdisc_issues:
                logger.warning(f"CDISC Issues in {dataset_name}: {cdisc_issues}")
            
            if not integrity_issues and not cdisc_issues:
                logger.info(f"{dataset_name} passed all checks.")
        else:
            logger.warning(f"No rows generated for {dataset_name}")

    logger.info("Mission Complete.")

if __name__ == "__main__":
    main()
