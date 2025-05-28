import importlib.util
import sys
from pathlib import Path
import gc
import torch

# Path to no_kv.py
no_kv_path = Path(__file__).resolve().parent.parent.parent / '/CSE599K_SP25_Code' /'assignment2' / 'no_kv.py'
spec = importlib.util.spec_from_file_location("no_kv", no_kv_path)
no_kv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(no_kv)
NoKVEngine = no_kv.Engine

# Add both engine directories to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # current directory for flashinfer_pipeline

from flashinfer_pipeline import Engine as FlashInferEngine

PROMPTS = [
    "Today is a rainy day and I am going to",
    "UCB is",
    "Hi, who are you?",
    "The University of Washington is located in",
    "The capital of France is",
    "Once upon a time, there was a",
    "The quick brown fox jumps over the lazy",
    "In 2024, artificial intelligence will",
    "The largest planet in our solar system is",
    "To be or not to be, that is",
    "Python is a popular programming language because",
    "The capital of France is",
    "ASDF:AKSJDFLK ADfads",
    "123456789",
    "!!#$%^&*()_+",
]

MAX_ROUNDS = 60
MIN_ROUNDS = 20

NUM_BATCH = 10

def main():
    num_rounds_to_test = 5
    rounds_values = [torch.randint(MIN_ROUNDS, MAX_ROUNDS, (1,)).item() for _ in range(num_rounds_to_test)]
    no_kv_engine = NoKVEngine()

    for test_idx, rounds in enumerate(rounds_values):
        print(f"\n=== Test {test_idx+1}: Comparing generations for {len(PROMPTS)} prompts, rounds={rounds} (batched for flashinfer) ===\n")
        # Instantiate flashinfer_engine only once per test
        flashinfer_engine = FlashInferEngine()
        # Generate all outputs in parallel for flashinfer
        out_flashinfer = flashinfer_engine.generate_batched(PROMPTS, rounds=rounds)
        del flashinfer_engine
        gc.collect()
        torch.cuda.empty_cache()

        for idx, prompt in enumerate(PROMPTS):
            print(f"Prompt {idx+1}: {prompt}")
            out_no_kv = no_kv_engine.generate(prompt, rounds=rounds)
            out_fi = out_flashinfer[idx]
            if out_no_kv != out_fi:
                print("  [DIFFERENT]")
                print("    no_kv:       ", repr(out_no_kv))
                print("    flashinfer:  ", repr(out_fi))
                # sys.exit(1)
            else:
                print("  [SAME]")
                # print("    output:      ", repr(out_no_kv))
            print()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 