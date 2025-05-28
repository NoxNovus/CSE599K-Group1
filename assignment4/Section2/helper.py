import torch
import os
import tqdm
import safetensors

########################################
# WeightManager: Loading and Managing Weights
########################################

class WeightManager:
    """
    Manages loading and processing model weights from safetensors files.
    """
    
    @staticmethod
    def load_tensors(tensor_path: str) -> dict:
        """
        Loads all tensors from safetensors files in a directory.

        Args:
            tensor_path (str): Path to directory with safetensors files.
            
        Returns:
            dict: Mapping from tensor names to torch.Tensor objects.
        """
        original_tensors = {}
        # Iterate through files in the directory
        for file in tqdm.tqdm(os.listdir(tensor_path), desc="Loading safetensors"):
            if file.endswith(".safetensors"):
                # Open file in PyTorch mode
                tensors = safetensors.safe_open(os.path.join(tensor_path, file), 'pt')
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name)
                    original_tensors[name] = tensor
        return original_tensors

    def __init__(self):
        self.weight_map = {}

    def load_from_safe_tensor(self, tensor_path: str) -> None:
        """
        Loads weights from safetensors files, converts them to fp16, and moves them to GPU.

        Args:
            tensor_path (str): Path to directory with safetensors files.
        """
        self.weight_map = WeightManager.load_tensors(tensor_path)
        # Convert weights to half precision and move to CUDA device
        for key in self.weight_map.keys():
            # self.weight_map[key] = self.weight_map[key].half().to('cuda')
            device = f'cuda:{torch.distributed.get_rank()}'
            self.weight_map[key] = self.weight_map[key].half().to(device)


    def set_weight(self, operation_list, total_layers: int) -> None:
        """
        Applies processing operations on weights.

        Args:
            operation_list (list): List of operations, each having a processWeight() method.
            total_layers (int): Total number of transformer layers.
        """
        for op in operation_list:
            op.processWeight(self.weight_map, total_layers)

########################################
# Rotary Positional Embedding (RoPE) Functions
########################################

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the last half of the tensor along the final dimension.
    
    Args:
        x (torch.Tensor): Input tensor with shape [..., 2*d_half].
        
    Returns:
        torch.Tensor: Tensor with rotated halves.
    """
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]  # First half of features
    x2 = x[..., dim // 2:]   # Second half of features
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x: torch.Tensor, output: torch.Tensor, head_dim: int, offset: int = 0) -> None:
    """
    Applies RoPE (Rotary Positional Embedding) to the input tensor.
    
    RoPE adds position-dependent rotations to the tensor.
    
    Args:
        x (torch.Tensor): Input tensor with shape [seq_len, head_dim].
        output (torch.Tensor): Tensor to store the result (same shape as x).
        head_dim (int): Dimensionality of each attention head.
        offset (int): Positional offset.
    """
    seq_len, _ = x.shape  # [seq_len, head_dim]
    device = x.device
    dtype = x.dtype

    # Create positions: shape [seq_len]
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)

    base = 500000.0
    # Compute inverse frequency: shape [head_dim/2]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64)
                                 .float().to(device) / head_dim))
    
    # Expand dimensions for broadcasting:
    inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)
    position_ids_expanded = positions[None, None, :].float()
    
    # Compute frequency embeddings
    with torch.autocast(device_type=device.type, enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        # Duplicate frequencies for cos and sin parts:
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    
    # Reshape for multi-head compatibility:
    cos = cos.unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [1, 1, seq_len, head_dim]

    # Reshape x for applying RoPE:
    x = x.reshape(1, seq_len, -1, head_dim).transpose(1, 2)
    # Apply RoPE rotation
    x_rotated = x * cos + rotate_half(x) * sin
    # Restore original shape and copy into output
    output.copy_(x_rotated.transpose(1, 2).reshape(seq_len, -1).to(dtype=dtype))

########################################
# Model Weight Extraction
########################################

def extract_model_weights(weight_map: dict, layers: int) -> dict:
    """
    Extracts and organizes model weights from a weight_map into a dictionary.

    Args:
        weight_map (dict): Dictionary containing the full mapping of weight tensors.
        layers (int): Total number of transformer layers.

    Returns:
        dict: Dictionary with keys for embedding, layer-specific weights, and final projection weights.
    """
    weights = {}
    weights["embedding"] = weight_map["model.embed_tokens.weight"]
    weights["layernormAttn_weight"] = [
        weight_map[f"model.layers.{layer}.input_layernorm.weight"] for layer in range(layers)
    ]
    weights["self_attn_k_proj_weight"] = [
        weight_map[f"model.layers.{layer}.self_attn.k_proj.weight"] for layer in range(layers)
    ]
    weights["self_attn_v_proj_weight"] = [
        weight_map[f"model.layers.{layer}.self_attn.v_proj.weight"] for layer in range(layers)
    ]
    weights["self_attn_q_proj_weight"] = [
        weight_map[f"model.layers.{layer}.self_attn.q_proj.weight"] for layer in range(layers)
    ]
    weights["o_proj_weight"] = [
        weight_map[f"model.layers.{layer}.self_attn.o_proj.weight"] for layer in range(layers)
    ]
    weights["layernormFFN_weight"] = [
        weight_map[f"model.layers.{layer}.post_attention_layernorm.weight"] for layer in range(layers)
    ]
    weights["up_proj_weight"] = [
        weight_map[f"model.layers.{layer}.mlp.up_proj.weight"] for layer in range(layers)
    ]
    weights["gate_proj_weight"] = [
        weight_map[f"model.layers.{layer}.mlp.gate_proj.weight"] for layer in range(layers)
    ]
    weights["down_proj_weight"] = [
        weight_map[f"model.layers.{layer}.mlp.down_proj.weight"] for layer in range(layers)
    ]
    weights["model_layernorm_weight"] = weight_map["model.norm.weight"]
    weights["lm_head_weight"] = weight_map["lm_head.weight"]
    return weights