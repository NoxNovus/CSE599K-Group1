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
    # def decode(self, input_ids, prefill = True):
        

    def run(self, input_ids, prefill = True):
        ########################################
        # Already implemented
        ########################################
        input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        hidden_state = self.weights["embedding"][input_tensor]
        
        for current_layer in range(self.layers):
            print("Current layer", current_layer)
            # --- Self-Attention Block ---
            # RMS
            rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x = normalized_x.to(torch.float16) * self.weights["layernormAttn_weight"][current_layer]
            
            # # KVQ calculations
            # # load from KV cache
            # # print(self.kv_cache)
            # if (len(self.kv_cache) <= current_layer):
            #     # do prefil!
            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())

            #     # RoPE on K & Q
            #     self.kv_cache[current_layer] = {"k" : k, "q" : q, "v" : v}
            # else:
            #     # do decode step!
            #     # load KV cache
            #     k = self.kv_cache[current_layer]["k"]
            #     q = self.kv_cache[current_layer]["q"]
            #     v = self.kv_cache[current_layer]["v"]
            #     last_token = x[-1]
            #     last_q = last_token.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())
            #     last_v = last_token.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            #     last_k = last_token.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())

            #     last_q = torch.unsqueeze(last_q, dim=0)
            #     last_k = torch.unsqueeze(last_k, dim=0)
            #     last_v = torch.unsqueeze(last_v, dim=0)

            #     apply_rope(last_q, output=last_q, head_dim=self.head_dim, offset=len(q))
            #     apply_rope(last_k, output=last_k, head_dim=self.head_dim, offset=len(k))
            #     # apply_rope(last_v, output=last_v, head_dim=self.head_dim, offset=0)
            #     k = torch.cat([k, last_k], dim=0)
            #     q = torch.cat([q, last_q], dim=0)
            #     v = torch.cat([v, last_v], dim=0) # REMOVE LATER
            #     self.kv_cache[current_layer]["k"] = k
            #     self.kv_cache[current_layer]["q"] = q
            #     self.kv_cache[current_layer]["v"] = v # REMOVE LATER
                
            # we load entire KV cache
            # but we only need a single query (for the latest token)
            single_q = q[-1]
            # this is a vector

            scale = 1.0 / (self.head_dim ** 0.5)
            # Grouped Query Attention
            group_size = self.num_qo_heads // self.num_kv_heads
            
            # sub_q = q.view(-1, self.num_qo_heads, self.head_dim) # (seq_len, num_qo_heads, head_dim)
            print("single_q shape:", single_q.shape)
            print("x shape:", x.shape)
            single_sub_q = single_q.view(self.num_qo_heads, self.head_dim) # (num_qo_heads, head_dim)
            num_qo_heads =  single_sub_q.shape[0]
            print("num_qo_heads:", num_qo_heads)
            sub_k = k.view(-1, self.num_kv_heads, self.head_dim) # (seq_len, num_kv_heads, head_dim)
            sub_v = v.view(-1, self.num_kv_heads, self.head_dim) # (seq_len, num_kv_heads, head_dim)
            seq_len, num_kv_heads, head_dim = sub_v.shape
            print("seq_len:", seq_len)
            print("num_kv_heads:", num_kv_heads)
            print("head_dim:", head_dim)
            # n_q = sub_q.shape[0]
            n_q = 1
            n_k = sub_k.shape[0]
            
            sub_k = sub_k.repeat_interleave(group_size, dim=1)
            sub_v = sub_v.repeat_interleave(group_size, dim=1)
            
            # sub_q_t = sub_q.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim)
            single_sub_q_t = single_sub_q
            sub_k_t = sub_k.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim)
            
            # scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale # (num_qo_heads, seq_len, seq_len)
            # scores = torch.matmul(sub_k_t.transpose(-2, -1), single_sub_q_t) * scale # (num_qo_heads, seq_len)
            scores = torch.einsum("h s d, h d -> h s", sub_k_t, single_sub_q_t) * scale
            
            # no need for mask, as single token 
            # causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)) # (seq_len, seq_len)
            # scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf")) # (1, seq_len, seq_len)
            
            # attn_weights = torch.softmax(scores, dim=-1)

            # assert(scores.shape == single_q.shape)
            attn_weights = torch.softmax(scores, -1) # (num_qo_heads, seq_len)
            
            v_t = sub_v.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim)
            # attn_output = torch.matmul(attn_weights, v_t) # (num_qo_heads, seq_len, head_dim)


            attn_output = torch.einsum("a b, a b c -> a c", attn_weights, v_t)

            # temp_one_attn_output = torch.matmul(attn_weights, v_t) # (num_qo_heads, head_dim)
            # attn_output = attn_output.permute(1, 0, 2) # (seq_len, num_qo_heads, head_dim)
            attn_output = attn_output.reshape(1, num_qo_heads * head_dim) # (1, num_qo_heads * head_dim)
            print("attn output shape: ", attn_output.shape)
            
            prefill_output = attn_output.matmul(self.weights["o_proj_weight"][current_layer].t()) + hidden_state
            prefill_output = torch.unsqueeze(prefill_output, dim=0) # (1, num_qo_heads * head_dim)
            # --- Feed-Forward Network (FFN) Block ---
            rms = torch.sqrt(torch.mean(prefill_output ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = prefill_output / rms
            layernormFFN_output = normalized_x.to(torch.float16) * self.weights["layernormFFN_weight"][current_layer]
            
            up_proj_output = layernormFFN_output.matmul(self.weights["up_proj_weight"][current_layer].t())
            gate_proj_output = layernormFFN_output.matmul(self.weights["gate_proj_weight"][current_layer].t())
            
            activation_output = up_proj_output * torch.nn.functional.silu(gate_proj_output)
            hidden_state = activation_output.matmul(self.weights["down_proj_weight"][current_layer].t()) + prefill_output
            print("hidden state: ", hidden_state.shape)

        # --- Final Layer Normalization and Output Projection ---
        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        model_output = normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
        logits = model_output.matmul(self.weights["lm_head_weight"].t())
        
        sample_output = torch.argmax(logits, dim=1)
        print("returning sample output: ", sample_output[-1].item().shape)
        return sample_output[-1].item()
    
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