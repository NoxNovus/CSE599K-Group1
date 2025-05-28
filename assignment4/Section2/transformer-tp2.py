import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from tqdm import tqdm

from helper import WeightManager, apply_rope, extract_model_weights

# === Initialization ===
# TODO: set CUDA_VISIBLE_DEVICES to 2 GPUs before running the code
# e.g. export CUDA_VISIBLE_DEVICES=2,3

# Initialize the distributed process group with NCCL backend
# Used for communication between GPUs (Tensor Parallelism)
dist.init_process_group(backend="nccl")
# Set the current device to the process's rank
# Each rank corresponds to one GPU
torch.cuda.set_device(dist.get_rank())

# === Model Parameters ===
# Path to the pretrained model weights
weight_path = "/data/Meta-Llama-3-8B-Instruct"

# LLaMA-3 8B has 32 transformer layers
layers = 32
num_qo_heads = 32   # Number of query/output heads per transformer block
head_dim = 128       # Dimension per attention head
num_kv_heads = 8     # Number of key/value heads
hidden_dim = 4096    # Total hidden dimension of transformer
intermediate_dim = 14336  # Intermediate dimension for MLP (FFN)

# === Load Weights ===
# Load tokenizer and model weights
# Tokenizer helps convert strings to token IDs
# WeightManager loads the weights from safetensors

# Load tokenizer for text encoding/decoding
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
weight_manager = WeightManager()
weight_manager.load_from_safe_tensor(weight_path)
weights = extract_model_weights(weight_manager.weight_map, layers)

# Extract and name all necessary model weights for easier access
embedding = weights["embedding"]
layernormAttn_weight = weights["layernormAttn_weight"]
self_attn_q_proj_weight = weights["self_attn_q_proj_weight"]
self_attn_k_proj_weight = weights["self_attn_k_proj_weight"]
self_attn_v_proj_weight = weights["self_attn_v_proj_weight"]
o_proj_weight = weights["o_proj_weight"]
layernormFFN_weight = weights["layernormFFN_weight"]
up_proj_weight = weights["up_proj_weight"]
gate_proj_weight = weights["gate_proj_weight"]
down_proj_weight = weights["down_proj_weight"]
model_layernorm_weight = weights["model_layernorm_weight"]
lm_head_weight = weights["lm_head_weight"]

# === Run One Iteration with Tensor Parallelism ===
def run_one_iteration(input_ids, rank, world_size):
    # Convert input token IDs to tensor and move to current device
    input_tensor = torch.tensor(input_ids, dtype=torch.int32, device=f"cuda:{rank}")
    hidden_state = embedding.to(f'cuda:{rank}')[input_tensor]  # Embed input tokens

    # Define local dimensions for tensor parallelism
    local_q_heads = num_qo_heads // world_size
    local_kv_heads = num_kv_heads // world_size
    local_hidden_dim = hidden_dim // world_size
    local_intermediate_dim = intermediate_dim // world_size

    for layer in range(layers):
        # --- Attention Block ---
        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        x = normalized_x * layernormAttn_weight[layer]
        
        # --- Part 1 Implement the attention block ---

        # TODO: generate the q, k, v vectors
        # assuming that the weights of q_proj, k_proj, v_proj are split in a column-wise (head) manner
        # hint: use rank, local_q_heads, local_kv_heads, head_dim to figure out the correct slice
        # hint: to debug, compare the intermediate outputs with the original implementation in transformer-w3l1.py
        
        # q = 
        # k = 
        # v = 

        # Apply rotary position embeddings
        apply_rope(q, output=q, head_dim=head_dim, offset=0)
        apply_rope(k, output=k, head_dim=head_dim, offset=0)

        # Reshape into multi-head format and replicate KV for grouped attention
        sub_q = q.view(-1, local_q_heads, head_dim)
        sub_k = k.view(-1, local_kv_heads, head_dim).repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
        sub_v = v.view(-1, local_kv_heads, head_dim).repeat_interleave(num_qo_heads // num_kv_heads, dim=1)

        # Transpose for batch matmul in attention computation
        sub_q_t = sub_q.permute(1, 0, 2)
        sub_k_t = sub_k.permute(1, 0, 2)
        scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))

        # Causal mask to prevent attending to future tokens
        causal_mask = torch.tril(torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        v_t = sub_v.permute(1, 0, 2)
        attn_output = torch.matmul(attn_weights, v_t).permute(1, 0, 2).reshape(-1, local_q_heads * head_dim)

        # TODO: generate the o_proj_local vector
        # assuming that the weights of o_proj are split in a row-wise manner
        # hint: use rank, local_hidden_dim to figure out the correct slice
        # o_proj_local = 
        
        # TODO: perform the all-reduce operation
        # hint: use dist.all_reduce 
        
        o_proj_residual = o_proj_local + hidden_state  # Add residual

        # --- Part 2 Implement the feedforward block ---
        
        rms = torch.sqrt(torch.mean(o_proj_residual ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = o_proj_residual / rms
        ffn_input = normalized_x.to(torch.float16) * layernormFFN_weight[layer]

        # TODO: generate the up_local and gate_local vectors
        # assuming that the weights of up_proj and gate_proj are split in a column-wise manner
        # hint: use rank, local_intermediate_dim to figure out the correct slice
        # up_local = 
        # gate_local = 

        # SwiGLU activation (SiLU * linear)
        activation_output = up_local * F.silu(gate_local)

        # TODO: generate the down_local vector
        # assuming that the weights of down_proj are split in a row-wise manner
        # down_local = 

        # TODO: perform the all-reduce operation

        # Add residual
        hidden_state = down_local + o_proj_residual

    # Final layer norm and projection to vocabulary space
    rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
    normalized_x = hidden_state / rms
    model_output = normalized_x.to(torch.float16) * model_layernorm_weight

    logits = model_output.matmul(lm_head_weight.t())  # Final linear layer to get logits
    return torch.argmax(logits, dim=-1)[-1].item()  # Return the ID of the last token predicted

# === Text Generation ===
def generate():
    rank = dist.get_rank()
    world_size = 2
    input_string = "The University of Michigan is a"
    input_ids = tokenizer.encode(input_string)
    output_ids = input_ids.copy()
    for _ in range(100):
        new_token = run_one_iteration(output_ids, rank, world_size)
        output_ids.append(new_token)
    if rank == 0:
        print("\nOutput:", tokenizer.decode(output_ids, skip_special_tokens=True))

# === Entry ===
if __name__ == '__main__':
    # warm up
    for i in tqdm(range(10)):
        generate()
    dist.barrier()
    
    start_time = time.time()
    for i in tqdm(range(10)):
        generate()
    end_time = time.time()
    print(f"Average time taken: {(end_time - start_time) / 10} seconds")
