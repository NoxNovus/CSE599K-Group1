import os
import time
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.distributed as dist

from helper import WeightManager, apply_rope, extract_model_weights


# === Initialization ===, only use 1 GPU
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500" # change if needed
dist.init_process_group(backend="nccl", rank=0, world_size=1)
torch.cuda.set_device(0)



weight_path = "/data/Meta-Llama-3-8B-Instruct"

layers = 32
num_qo_heads = 32      # Total number of query/output heads
head_dim = 128         # Dimensionality of each attention head, (hidden_dim / # of heads)
num_kv_heads = 8       # Total number of key/value heads
# 4 query heads share 1 key/value head. This is called grouped query attention

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# Initialize and load model weights using the helper module
weight_manager = WeightManager()
# Safetensors are a serialization format for tensors that is more efficient and safer
# than traditional formats. Loading doesn't execute arbitrary code, making it safe
weight_manager.load_from_safe_tensor(weight_path)
# Extract model weights from the weight map
weights = extract_model_weights(weight_manager.weight_map, layers)

# Unpack weights for convenient reference
embedding = weights["embedding"] # token_ids -> embedding

# Attention weights, they are per layer. We'll index them by layer number
layernormAttn_weight = weights["layernormAttn_weight"]
self_attn_q_proj_weight = weights["self_attn_q_proj_weight"]
self_attn_k_proj_weight = weights["self_attn_k_proj_weight"]
self_attn_v_proj_weight = weights["self_attn_v_proj_weight"]
o_proj_weight = weights["o_proj_weight"]

# FFN weights
layernormFFN_weight = weights["layernormFFN_weight"]
up_proj_weight = weights["up_proj_weight"]
gate_proj_weight = weights["gate_proj_weight"]
down_proj_weight = weights["down_proj_weight"]

# Final layer normalization
model_layernorm_weight = weights["model_layernorm_weight"]

# Final vocabulary projection. Here, "head" is not the same as head in multi-head attention.
# "head" often refers to the final layer or component that maps the model's internal 
# hidden representations (embeddings) to the output space
lm_head_weight = weights["lm_head_weight"]

#######################################################
# Main Generation Loop: One Iteration/ Forward Pass
#######################################################

def run_one_iteration(input_ids: list) -> int:
    # --- Multi-Headed Causal Self-Attention ---
    input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')

    # Create hidden state tensor by indexing into the embedding matrix with input tensor [slide]
    # I: (seq_len): (seq_len)
    # O: (seq_len, hidden_dim): (seq_len, 4096)
    hidden_state = embedding[input_tensor]

    for layer in range(layers):
        # RMSNorm for each vector of user requests
        # I: (seq_len, hidden_dim): (seq_len, 4096)
        # O: (seq_len, hidden_dim): (seq_len, 4096)
        # Hidden dimension is also referred to as the embedding dimension or d_model.
        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5) # (seq_len, 1)
        normalized_x = hidden_state / rms # (seq_len, hidden_dim): (seq_len, 4096)
        # To complete normalization, we have an elelment-wise multiplication of the 
        # normalized vectors with the layernorm weights
        x = normalized_x * layernormAttn_weight[layer] # (seq_len, hidden_dim): (seq_len, 4096)

        # QKV projection
        # I_1 / x: (seq_len, hidden_dim): (seq_len, 4096)
        # I_2 / w: (num_qo_heads * head_dim, hidden_dim): (32 * 128, 4096)
        # O: (seq_len, num_qo_heads * head_dim): (seq_len, 32 * 128)
        q = x.matmul(self_attn_q_proj_weight[layer].t()) # (seq_len, hidden_dim): (seq_len, 4096)
        # For k, v, num_kv_heads = 8, so w is (num_kv_heads * head_dim, hidden_dim): (8 * 128, 4096)
        k = x.matmul(self_attn_k_proj_weight[layer].t()) # O: (seq_len, num_kv_heads * head_dim): (seq_len, 8 * 128)
        v = x.matmul(self_attn_v_proj_weight[layer].t()) # O: (seq_len, num_kv_heads * head_dim): (seq_len, 8 * 128)

        # RoPE (Rotary Positional Embedding)
        apply_rope(q, output=q, head_dim=head_dim, offset=0)
        apply_rope(k, output=k, head_dim=head_dim, offset=0)

        # Compute sub-components of q,k,v for each head
        # I: (seq_len, num_qo_heads * head_dim): (seq_len, 32 * 128)
        sub_q = q.view(-1, num_qo_heads, head_dim) # (seq_len, num_qo_heads, head_dim): (seq_len, 32, 128)

        sub_k = k.view(-1, num_kv_heads, head_dim)
        sub_v = v.view(-1, num_kv_heads, head_dim)

        # Compute attention-related values
        scale = 1.0 / (head_dim**0.5)
        group_size = num_qo_heads // num_kv_heads
        n_q = sub_q.shape[0]
        n_k = sub_k.shape[0]

        # Replication sub_k and sub_v for each group of query heads
        sub_k = sub_k.repeat_interleave(group_size, dim=1)
        sub_v = sub_v.repeat_interleave(group_size, dim=1)

        # Rearrange q and k so that shapes are (num_qo_heads, seq_len, head_dim)
        # each query vector and each key vector across the sequence dimension. Specifically, for 
        # multi-head attention, the operation must independently occur across each attention head.
        sub_q_t = sub_q.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        sub_k_t = sub_k.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)

        # Compute attention scores
        scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale
        
        causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device))
        # Expand the mask along the first dimension (1, seq_len, seq_len) to match the shape of the scores
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))
    
        attn_weights = torch.softmax(scores, dim=-1) # (num_qo_heads, seq_len, seq_len): (32, seq_len, seq_len)
        
        # Compute attention output by multiplying weights with the values. sub_v has shape (seq_len, num_qo_heads, head_dim)
        # Transpose the sub_v tensor to get the shape (num_qo_heads, seq_len, head_dim)
        v_t = sub_v.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        # I_1/w: (num_qo_heads, seq_len, seq_len): (32, seq_len, seq_len)
        # I_2/v: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        # O: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        attn_output = torch.matmul(attn_weights, v_t)

        # Go back to single-head from multi-head attention:
        attn_output = attn_output.permute(1, 0, 2) # (seq_len, num_qo_heads, head_dim): (seq_len, 32, 128)
        # Reshape to combine all heads' outputs into a single tensor
        # I: (seq_len, num_qo_heads, head_dim): (seq_len, 32, 128)
        # O: (seq_len, hidden_dim): (seq_len, 4096) # hidden_dim = num_qo_heads * head_dim
        attn_output = attn_output.reshape(-1, num_qo_heads * head_dim)


        # Output projection and residual connection. There is a residual connection in the attention block
        # So we add the original input to output projection. Seminal paper: https://arxiv.org/abs/1512.03385
        # I_1/attn: (seq_len, hidden_dim): (seq_len, 4096)
        # I_2/w: (hidden_dim, hidden_dim): (4096, 4096)
        # O: (seq_len, hidden_dim): (seq_len, 4096)
        o_proj_residual = attn_output.matmul(o_proj_weight[layer].t()) + hidden_state   
        
        # --- Feed-Forward Network (FFN) ---

        # RMSNorm before FFN
        rms = torch.sqrt(torch.mean(o_proj_residual ** 2, dim=-1, keepdim=True) + 1e-5) # (seq_len, 1)
        normalized_x = o_proj_residual / rms # (seq_len, hidden_dim): (seq_len, 4096)
        layernormFFN_output = normalized_x.to(torch.float16) * layernormFFN_weight[layer] # (seq_len, hidden_dim): (seq_len, 4096)

        # Up projection
        # I_1/ln: (seq_len, hidden_dim): (seq_len, 4096)
        # I_2/w: (hidden_dim * 4,hidden_dim): (16384, 4096)
        # O: (seq_len, hidden_dim * 4): (seq_len, 16384)
        up_proj_output = layernormFFN_output.matmul(up_proj_weight[layer].t())
        
        # Gate
        # I_1/ln: (seq_len, hidden_dim): (seq_len, 4096)
        # I_2/w: (hidden_dim * 4,hidden_dim): (16384, 4096)
        # O: (seq_len, hidden_dim * 4): (seq_len, 16384)
        gate_proj_output = layernormFFN_output.matmul(gate_proj_weight[layer].t())

        # Gate + SiLU = SwiGLU
        activation_output = up_proj_output * torch.nn.functional.silu(gate_proj_output) # (seq_len, hidden_dim * 4): (seq_len, 16384)

        # Down projection 
        # I_1/act: (seq_len, hidden_dim * 4): (seq_len, 16384)
        # I_2/w: (hidden_dim, hidden_dim * 4): (4096, 16384)
        # O: (seq_len, hidden_dim): (seq_len, 4096)
        down_proj_output = activation_output.matmul(down_proj_weight[layer].t())

        # Residual connection
        hidden_state = down_proj_output + o_proj_residual

    # --- Final RMS Normalization, Projection to Vocabulary, Sampling ---

    #RMSNorm
    rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5) # (seq_len, 1)
    normalized_x = hidden_state / rms # size (seq_len, hidden_dim): (seq_len, 4096)
    model_output = normalized_x.to(torch.float16) * model_layernorm_weight # size (seq_len, hidden_dim): (seq_len, 4096)

    # Project to vocabulary
    # I: (seq_len, hidden_dim): (seq_len, 4096)
    # O: (seq_len, vocab_size): (seq_len, 128256)
    # For each token in the sequence, this gives a probability distribution over the vocabulary
    logits = model_output.matmul(lm_head_weight.t()) # (seq_len, vocab_size): (seq_len, 128256)
    # Pick the next token with the highest probability
    sample_output = torch.argmax(logits, dim=-1) # (vocab_size): (128256
    # print(f"sample_output: {sample_output[-1].item()}")
    # print(f"token: {tokenizer.decode(sample_output[-1].item())}")
    return sample_output[-1].item()

########################################
# Main Loop: Text Generation
########################################
def generate():
    input_string = "The University of Michigan is a"
    input_ids = tokenizer.encode(input_string)

    output_ids = input_ids.copy()

    print(f"size of input string: {len(input_string.split())}")
    print(f"size of input_ids: {len(input_ids)}")

    iterations = 100
    for round in range(iterations):
        new_token = run_one_iteration(output_ids)
        output_ids.append(new_token)

    # Skipe special tokens
    output_string = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"output string: {output_string}")

if __name__ == "__main__":
    # warm up
    for i in tqdm(range(10)):
        generate()
    dist.barrier()
    
    # run 10 times and time it
    start_time = time.time()
    for i in tqdm(range(10)):
        generate()
    end_time = time.time()
    print(f"Average time taken: {(end_time - start_time) / 10} seconds")