import torch
from transformers import AutoTokenizer
import sys
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
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)
    
    def run(self, input_ids, prefill = True):
        ########################################
        # Already implemented
        ########################################
        input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        hidden_state = self.weights["embedding"][input_tensor]
        
        for current_layer in range(self.layers):
            # --- Self-Attention Block ---
            rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x = normalized_x.to(torch.float16) * self.weights["layernormAttn_weight"][current_layer]
            
            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())
            
            # Apply RoPE to query and key using the helper function
            apply_rope(q, output=q, head_dim=self.head_dim, offset=0)
            apply_rope(k, output=k, head_dim=self.head_dim, offset=0)
            
            scale = 1.0 / (self.head_dim ** 0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            
            sub_q = q.view(-1, self.num_qo_heads, self.head_dim) # (seq_len, num_qo_heads, head_dim)
            sub_k = k.view(-1, self.num_kv_heads, self.head_dim) # (seq_len, num_kv_heads, head_dim)
            sub_v = v.view(-1, self.num_kv_heads, self.head_dim) # (seq_len, num_kv_heads, head_dim)
            
            n_q = sub_q.shape[0]
            n_k = sub_k.shape[0]
            
            sub_k = sub_k.repeat_interleave(group_size, dim=1)
            sub_v = sub_v.repeat_interleave(group_size, dim=1)
            
            sub_q_t = sub_q.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim)
            sub_k_t = sub_k.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim)
            
            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale # (num_qo_heads, seq_len, seq_len)
            
            causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)) # (seq_len, seq_len)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf")) # (1, seq_len, seq_len)
            
            attn_weights = torch.softmax(scores, dim=-1)
            
            v_t = sub_v.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim)
            attn_output = torch.matmul(attn_weights, v_t) # (num_qo_heads, seq_len, head_dim)
            attn_output = attn_output.permute(1, 0, 2) # (seq_len, num_qo_heads, head_dim)
            
            attn_output = attn_output.reshape(-1, self.num_qo_heads * self.head_dim) # (seq_len, num_qo_heads * head_dim)
            prefill_output = attn_output.matmul(self.weights["o_proj_weight"][current_layer].t()) + hidden_state
            
            # --- Feed-Forward Network (FFN) Block ---
            rms = torch.sqrt(torch.mean(prefill_output ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = prefill_output / rms
            layernormFFN_output = normalized_x.to(torch.float16) * self.weights["layernormFFN_weight"][current_layer]
            
            up_proj_output = layernormFFN_output.matmul(self.weights["up_proj_weight"][current_layer].t())
            gate_proj_output = layernormFFN_output.matmul(self.weights["gate_proj_weight"][current_layer].t())
            
            activation_output = up_proj_output * torch.nn.functional.silu(gate_proj_output)
            hidden_state = activation_output.matmul(self.weights["down_proj_weight"][current_layer].t()) + prefill_output

        # --- Final Layer Normalization and Output Projection ---
        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        model_output = normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
        logits = model_output.matmul(self.weights["lm_head_weight"].t())
        
        sample_output = torch.argmax(logits, dim=1)
        return sample_output[-1].item()
    
    def generate(self, input_string, rounds=20):
        input_ids = self.tokenizer.encode(input_string)

        print("Token IDs:", input_ids)
        output_ids = input_ids.copy()

        new_token = self.run(output_ids)
        output_ids.append(new_token)

        for round in range(rounds - 1):
            print(f"Round {round}")
            new_token = self.run(output_ids, prefill=True)
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