import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import logging
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)

class QwenGenerator:
    def __init__(self, model_id="Qwen/Qwen2.5-3B-Instruct", device="auto", load_in_4bit=True):
        self.model_id = model_id
        self.device = device
        
        # Check environment
        self.has_cuda = torch.cuda.is_available()
        if not self.has_cuda:
            logger.warning("CUDA not found. Switching 4-bit quantization OFF and using basic loading.")
            load_in_4bit = False
            if device == "auto":
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.tokenizer = None
        self.model = None
        
        try:
            self._load_model(model_id, load_in_4bit)
        except OSError:
            logger.warning(f"Model {model_id} not found. Falling back to basic CPU loading or check network.")
            raise

    def _load_model(self, model_id, load_in_4bit):
        logger.info(f"Loading model {model_id} on {self.device} (4-bit: {load_in_4bit})...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if load_in_4bit and self.has_cuda:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            if self.device != "auto":
                self.model.to(self.device)

    def construct_messages(self, anchor, spec, row_instruction=""):
        """
        Constructs a list of messages for the chat template.
        """
        system_content = (
            "Role: CDISC ADaM Synthetic Data Generator\n"
            "Constraint: Output must be valid JSON matching the Specification variables.\n"
            "Do not output markdown code blocks. Output ONLY the raw JSON object."
        )

        user_content = f"""[EXAMPLE]
Context: {{ "dataset": "ADAE", "variables": {{ "USUBJID": "...", ... }} }}
Anchor: {{ "USUBJID": "01-1001" }}
Output: {{ "USUBJID": "01-1001", "ASTDT": "2023-01-01", "AESEV": "MILD", "AETERM": "Headache", "AEDECOD": "Headache" }}

[SPECIFICATION]
{json.dumps(spec, indent=2)}

[SUBJECT_ANCHOR]
{json.dumps(anchor, indent=2)}

[INSTRUCTION]
{row_instruction}
Dataset: {spec.get('dataset', 'UNKNOWN')}
Generate one row of data for this subject as JSON.
"""
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def generate_row(self, messages, max_retries=3):
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        for _ in range(max_retries):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode only the new tokens (removing input prompt)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Attempt to Parse JSON
            try:
                # Find JSON substring if extra text exists
                json_match = re.search(r'\{.*\}', decoded, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    return data
                else:
                    logger.warning(f"No JSON found in output: {decoded}")
            except json.JSONDecodeError:
                logger.warning(f"JSON decode error: {decoded}")
                continue
                
        return None

    def generate_dataset(self, anchors, spec, output_path):
        dataset_name = spec.get('dataset', 'DATA')
        rows = []
        
        logger.info(f"Generating {dataset_name} for {len(anchors)} subjects...")
        
        for anchor in tqdm(anchors):
            messages = self.construct_messages(
                anchor, 
                spec, 
                row_instruction=f"Generate {dataset_name} row. Ensure consistency with Anchor dates."
            )
            
            row_data = self.generate_row(messages)
            if row_data:
                # Merge Anchor keys if missing (USUBJID usually required)
                if "USUBJID" not in row_data:
                    row_data["USUBJID"] = anchor["USUBJID"]
                rows.append(row_data)
        
        return rows

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test
    # Load dummy spec and anchors
    try:
        # Create dummy anchors for testing if file doesn't exist
        dummy_anchors = [{"USUBJID": "01-1001", "TRTSDT": "2024-01-01"}]
        
        # Manually create a minimal context if spec_parser not run or to test
        spec = {
            "dataset": "ADAE",
            "variables": {
                "USUBJID": {"type": "Char"},
                "ASTDT": {"type": "Num"},
                "AESEV": {"codelist": ["MILD", "MODERATE", "SEVERE"]}
            }
        }
        
        gen = QwenGenerator(load_in_4bit=False) 
        rows = gen.generate_dataset(dummy_anchors, spec, "output/test.json")
        print(json.dumps(rows, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
