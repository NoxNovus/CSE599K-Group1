from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import torch
import flashinfer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
#  Project utilities (local module)
# ---------------------------------------------------------------------------
# helper.py must live one directory above this file
sys.path.append(str(Path(__file__).resolve().parent.parent))
from helper import WeightManager, extract_model_weights  # noqa: E402


# ---------------------------------------------------------------------------
#  Low-level data structures: paged KV-cache & per-request view
# ---------------------------------------------------------------------------
class DistKVPool:
    """Global *paged* KV-cache ("HND" = head-page-dim layout)."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        capacity: int,
        page_size: int,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.page_size = page_size

        # Simple free-list allocator ------------------------------------------------
        self._free_pages: set[int] = set(range(capacity))

        # Backing storage tensors: (L, N, H, P, D) where
        #   L = num transformer layers
        #   N = total pages in the pool
        kv_shape = (
            num_layers,
            capacity,
            num_kv_heads,
            page_size,
            head_dim,
        )
        self.k_datas = torch.empty(kv_shape, dtype=torch.float16, device="cuda")
        self.v_datas = torch.empty_like(self.k_datas)

    # Free-list helpers -----------------------------------------------------------
    @property
    def num_free_pages(self) -> int:  # noqa: D401  (short property)
        """Number of unallocated pages left in the pool."""
        return len(self._free_pages)

    def alloc_page(self) -> int:
        """Pop a page index off the free list (O(1))."""
        return self._free_pages.pop()

    def free_page(self, idx: int) -> None:
        """Return *idx* back to the pool."""
        assert idx not in self._free_pages, "double-free detected"
        self._free_pages.add(idx)


class DistKVCache:
    """Light-weight *view* of a request's KV pages (no real storage)."""

    def __init__(self, pool: DistKVPool):
        self._pool = pool
        self._indices: list[int] = []  # page indices owned by this request
        self._seqlen: int = 0          # total tokens stored so far
        self.page_size = pool.page_size

    # Convenience properties -----------------------------------------------------
    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    def indices(self) -> list[int]:
        return self._indices

    @property
    def last_page_offset(self) -> int:
        """Number of tokens already present in the *last* page (0-based)."""
        if self._seqlen == 0:
            return 0
        remainder = self._seqlen % self.page_size
        return self.page_size if remainder == 0 else remainder

    # Allocation / release -------------------------------------------------------
    def allocate_tokens(self, num_tokens: int) -> None:
        """Grow the cache so it can hold *num_tokens* additional tokens."""
        assert num_tokens > 0, "must allocate a positive number of tokens"

        # Tokens that still fit into the *current* (possibly partial) page --------
        room_in_last = (
            self.page_size - self.last_page_offset
        ) % self.page_size  # 0 when last page is full

        remaining = max(0, num_tokens - room_in_last)
        pages_needed = (remaining + self.page_size - 1) // self.page_size

        for _ in range(pages_needed):
            self._indices.append(self._pool.alloc_page())

        self._seqlen += num_tokens

    def release(self) -> None:
        """Return all pages back to the global pool (when request finishes)."""
        for idx in self._indices:
            self._pool.free_page(idx)
        self._indices.clear()
        self._seqlen = 0


# ---------------------------------------------------------------------------
#  Helpers to convert a *list* of DistKVCache into FlashInfer ragged metadata
# ---------------------------------------------------------------------------

def build_kv_metadata(kvs: List[DistKVCache]):
    """Return (indptr, indices, last_page_len) - all torch.cuda tensors."""
    kv_indptr: List[int] = [0]
    kv_indices: List[int] = []
    kv_last_page_len: List[int] = []

    for kv in kvs:
        pass
        #########
        # FIXME #
        #########

    device = "cuda"
    return (
        torch.tensor(kv_indptr, dtype=torch.int32, device=device),
        torch.tensor(kv_indices, dtype=torch.int32, device=device),
        torch.tensor(kv_last_page_len, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
#  Simple *request* wrapper (prompt + generation buffer)
# ---------------------------------------------------------------------------
class Request:
    def __init__(self, req_id: int, prompt_ids: torch.Tensor, target_len: int):
        self.request_id = req_id
        self.prompt_token_ids = prompt_ids  # (prompt_len,)
        self.output_length = target_len
        # History buffer (prompt + generated tokens will be appended here)
        self.output_token_ids = prompt_ids.clone()

    # Convenience --------------------------------------------------------------
    @property
    def prompt_length(self) -> int:
        return self.prompt_token_ids.size(0)

    @property
    def current_length(self) -> int:
        return self.output_token_ids.size(0)


# ---------------------------------------------------------------------------
#  Generation *engine*
# ---------------------------------------------------------------------------
class Engine:
    """A minimal Llama-3-8B engine using FlashInfer for attention."""

    # ---------------------------------------------------------------------
    #  Initialisation
    # ---------------------------------------------------------------------
    def __init__(self) -> None:
        # ---- model hyper-parameters --------------------------------------
        self.weight_path = "/model/Meta-Llama-3-8B-Instruct"
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.layers = 32

        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path)

        # ---- load weights -------------------------------------------------
        wm = WeightManager()
        wm.load_from_safe_tensor(self.weight_path)
        self.weights = extract_model_weights(wm.weight_map, self.layers)

        # ---- global paged KV-cache ---------------------------------------
        self.page_size = 16
        self.max_pages = 20_000  # total pages in the pool (across *all* layers)
        self.pool = DistKVPool(
            num_layers=self.layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            capacity=self.max_pages,
            page_size=self.page_size,
        )

        # Mapping: request-id -> DistKVCache
        self.kv_cache_map: Dict[int, DistKVCache] = {}

        # FlashInfer workspace (single allocation for the whole run)
        workspace_bytes = 128 << 20  # 128 MiB
        self._fi_workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device="cuda"
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._fi_workspace, "HND"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._fi_workspace, "HND", use_tensor_cores=True)

    # ---------------------------------------------------------------------
    #  One *step* (mixed prefill + decode) over an *arbitrary* request batch
    # ---------------------------------------------------------------------
    def run(self, requests: List[Request], num_decode_req: int = 0):
        """Run *one* transformer step for ``requests``.

        Parameters
        ----------
        requests : List[Request]
            Full list of requests to be processed this step.
        num_decode_req : int, default=0
            Number of *decode* requests (the **first** N in *requests*).
            Those will feed only their **last** token; the rest are prefills.
        """
        with torch.inference_mode():
            # ----------------------------------------------------------------
            # 1) Build ragged *input* tensor and its CSR *indptr*
            # ----------------------------------------------------------------
            pieces: List[torch.Tensor] = []
            indptr: List[int] = [0]

            for idx, req in enumerate(requests):
                if idx < num_decode_req:  # decode - feed only *last* token
                    pieces.append(req.output_token_ids[-1:])
                    indptr.append(indptr[-1] + 1)
                else:                     # prefill - feed *whole* prompt
                    pieces.append(req.prompt_token_ids)
                    indptr.append(indptr[-1] + req.prompt_length)

            input_tensor = torch.cat(pieces).to("cuda")
            indptr_tensor = torch.tensor(indptr, dtype=torch.int32, device="cuda")

            # ----------------------------------------------------------------
            # 2) Create KV cache for prefill requests in kv_cache_map
            # ----------------------------------------------------------------
            
            #########
            # FIXME #
            #########
                
            seq_lens_before: List[int] = []
            seq_lens_before_t = torch.tensor(seq_lens_before, dtype=torch.int32, device="cuda")

            # ----------------------------------------------------------------
            # 3) Reserve allocate pages for all requests if needed using allocate_tokens function
            # ----------------------------------------------------------------
            
            #########
            # FIXME #
            #########
            
            seq_lens_after = [self.kv_cache_map[r.request_id].seqlen for r in requests]
            seq_lens_after_t = torch.tensor(seq_lens_after, dtype=torch.int32, device="cuda")

            # Build paged-KV metadata **after** the append -------------------
            kv_indptr, kv_indices, kv_last_page_len = build_kv_metadata(
                [self.kv_cache_map[r.request_id] for r in requests]
            )

            # ----------------------------------------------------------------
            # 4) Plan FlashInfer execution for batch
            # ----------------------------------------------------------------
            if not len(requests) - num_decode_req == 0:
                # plan prefill wrapper
                pass
                #########
                # FIXME #
                #########
            if num_decode_req > 0:
                # plan decode wrapper
                pass
                #########
                # FIXME #
                #########

            # ----------------------------------------------------------------
            # 5) Forward pass through all *transformer* layers
            # ----------------------------------------------------------------
            hidden = self.weights["embedding"][input_tensor]

            for layer in range(self.layers):
                # === Self-attention sub-layer ==================================
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_attn_in = (hidden / rms).to(torch.float16) * self.weights["layernormAttn_weight"][layer]

                k = (
                    ln_attn_in
                    .matmul(self.weights["self_attn_k_proj_weight"][layer].T)
                    .view(-1, self.num_kv_heads, self.head_dim)
                )
                v = (
                    ln_attn_in
                    .matmul(self.weights["self_attn_v_proj_weight"][layer].T)
                    .view(-1, self.num_kv_heads, self.head_dim)
                )
                q = (
                    ln_attn_in
                    .matmul(self.weights["self_attn_q_proj_weight"][layer].T)
                    .view(-1, self.num_qo_heads, self.head_dim)
                )

                # ---- Rotary positional embedding ---------------------------
                # Use flashinfer.apply_rope_inplace
                # apply ROPE, Note the the theta is set to 500_000.0 and offsets should be the current sequence length before allocate new tokens
                
                #########
                # FIXME #
                #########

                # ---- Append new tokens to *paged* KV-cache ------------------
                # Use flashinfer.get_batch_indices_positions and flashinfer.append_paged_kv_cache
                # if you use get_batch_indices_positions, seq_lens should be the length after the allocation

                #########
                # FIXME #
                #########

                # ---- Attention itself --------------------------------------
                # run prefill and decode wrappers. Note that for the prefill wrapper, if qo_indptr does not start with 0, first qo_indptr[0] rows of the output tensor will be empty
                attn_out = None
                #########
                # FIXME #
                #########
                
                # aggregate the decode and prefill outputs
                #########
                # FIXME #
                #########
                
                # Residual connection
                hidden = attn_out.matmul(self.weights["o_proj_weight"][layer].T) + hidden

                # === FFN sub-layer ==========================================
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_ffn_in = (hidden / rms).to(torch.float16) * self.weights["layernormFFN_weight"][layer]

                up = ln_ffn_in.matmul(self.weights["up_proj_weight"][layer].T)
                gate = ln_ffn_in.matmul(self.weights["gate_proj_weight"][layer].T)
                hidden = (
                    (up * torch.nn.functional.silu(gate))
                    .matmul(self.weights["down_proj_weight"][layer].T)
                    + hidden
                )

            # ----------------------------------------------------------------
            # 6) Final language-model head ----------------------------------
            rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
            logits = (
                (hidden / rms).to(torch.float16) * self.weights["model_layernorm_weight"]
            ).matmul(self.weights["lm_head_weight"].T)

            sample_ids = torch.argmax(logits, dim=-1)

            # Extract *new* token for each request (last token of each row)
            last_token_indices = (indptr_tensor[1:] - 1).long()
            return sample_ids[last_token_indices].cpu()

    # ---------------------------------------------------------------------
    #  Full batched *generation* loop (prefill + iterative decode)
    # ---------------------------------------------------------------------
    def generate_batched(self, prompts: List[str], rounds: int = 20):
        print(">>> starting batched generation ({} rounds)".format(rounds))

        # Build *Request* objects ------------------------------------------------
        requests: List[Request] = []
        for idx, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            requests.append(Request(idx, prompt_ids, rounds))

        # ---- 1) Prefill pass ---------------------------------------------------
        prefill_outputs = self.run(requests, num_decode_req=0)
        print("prefill pass finished - appending first generated token …")

        for i in range(len(requests)):
            new_tok = prefill_outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat(
                [requests[i].output_token_ids, new_tok], dim=0
            )
        
        # You do not need to support adding new request on the fly for this assignment, but if you want to, you can uncomment the following lines
        # requests.append(Request(999, self.tokenizer("Today is", return_tensors="pt").input_ids[0], rounds))
        # # ---- 1.5) Prefill pass for the new request --------------------------
        # prefill_outputs = self.run(requests, num_decode_req=len(requests) - 1)
        # for i in range(len(requests) - 1):
        #     new_tok = prefill_outputs[i].unsqueeze(0)
        #     requests[i].output_token_ids = torch.cat(
        #         [requests[i].output_token_ids, new_tok], dim=0
        #     )

        # ---- 2) Iterative decode passes ---------------------------------------
        for _ in range(rounds - 1):
            decode_outputs = self.run(requests, num_decode_req=len(requests))
            for i in range(len(requests)):
                new_tok = decode_outputs[i].unsqueeze(0)
                requests[i].output_token_ids = torch.cat(
                    [requests[i].output_token_ids, new_tok], dim=0
                )

        # ---- 3) Decode back to text and return -------------------------------
        return [
            self.tokenizer.decode(r.output_token_ids, skip_special_tokens=True)
            for r in requests
        ]


# ---------------------------------------------------------------------------
#  Entry-point (debug / standalone execution)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example batch: ten identically phrased prompts + ten location prompts
    sample_prompts = (
        ["Hi, who are you?"] * 100
        + ["The University of Washington is located in"] * 100
    )

    engine = Engine()
    generated_texts = engine.generate_batched(sample_prompts, rounds=30)

    for idx, text in enumerate(generated_texts):
        print(f"[request {idx:02d}] {text}\n")
