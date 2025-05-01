import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights


class Engine:
    """
    A class to manage the generation engine.
    """
    def __init__(self):
        ########################################
        # Model Configuration Parameters
        ########################################
        self.weight_path = "/model/Meta-Llama-3-8B-Instruct"
        self.head_dim = 128         # Dimensionality of each attention head
        self.num_qo_heads = 32      # Total number of query/output heads
        self.num_kv_heads = 8       # Total number of key/value heads
        self.layers = 32            # Number of transformer layers

        # Load the tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("/model/Meta-Llama-3-8B-Instruct")

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)
        
        self.kv_cache = {}
    
    def run(self, input_ids, prefill = True):
        ########################################
        # Complete this function
        ########################################
        pass
    
    def generate(self, input_string, rounds=20):
        input_ids = self.tokenizer.encode(input_string)

        print("Token IDs:", input_ids)
        output_ids = input_ids.copy()

        new_token = self.run(output_ids)
        output_ids.append(new_token)

        for round in range(rounds - 1):
            print(f"Round {round}")
            new_token = self.run(output_ids[-1:], prefill=False)
            output_ids.append(new_token)

        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    engine = Engine()
    output_text = engine.generate(input_string, rounds=20)
    print("Generated Text:", output_text)