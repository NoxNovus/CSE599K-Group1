from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time
from typing import Dict, List

import torch
import flashinfer
from transformers import AutoTokenizer
global_var = [None]
# ---------------------------------------------------------------------------
#  Project utilities (local module)
# ---------------------------------------------------------------------------
# helper.py must live one directory above this file
sys.path.append(str(Path(__file__).resolve().parent.parent))
from helper import WeightManager, extract_model_weights  # noqa: E402

def startTimer():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end

def endTimer(start, end):
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / 1000
    return elapsed_time

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
        kv_indices.extend(kv.indices)
        kv_indptr.append(len(kv_indices))
        kv_last_page_len.append(kv.last_page_offset)
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
    def clear(self):
        for ele in self.kv_cache_map:
            self.kv_cache_map[ele].release()

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
            overall_start, overall_end = startTimer()
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

            #S
            hasDecode = num_decode_req > 0
            hasPrefill = not len(requests) - num_decode_req == 0
            seq_lens_before: List[int] = []
            for idx, req in enumerate(requests, num_decode_req):
                self.kv_cache_map[idx] = DistKVCache(self.pool)
                seq_lens_before.append(self.kv_cache_map[req.request_id].seqlen)
           
            seq_lens_before_t = torch.tensor(seq_lens_before, dtype=torch.int32, device="cuda")
            #E

            #########
            # FIXME #
            #########
                
            # ----------------------------------------------------------------
            # 3) Reserve allocate pages for all requests if needed using allocate_tokens function
            # ----------------------------------------------------------------
            
            #S
            for idx, req in enumerate(requests):
                if idx < num_decode_req:
                    self.kv_cache_map[idx].allocate_tokens(1)
                else:
                    self.kv_cache_map[idx].allocate_tokens(req.prompt_length)
            #E

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
                self.prefill_wrapper.plan(
                    qo_indptr=indptr_tensor[num_decode_req:],
                    paged_kv_indices=kv_indices,
                    paged_kv_indptr=kv_indptr[num_decode_req:],  
                    paged_kv_last_page_len=kv_last_page_len,                  
                    num_qo_heads=self.num_qo_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim_qk=self.head_dim,
                    head_dim_vo=self.head_dim,
                    page_size=self.page_size,
                    causal=True,
                )
                pass
                #########
                # FIXME #
                #########
            if num_decode_req > 0:
                # plan decode wrapper
                self.decode_wrapper.plan(
                    indptr=kv_indptr,
                    indices=kv_indices,
                    last_page_len=kv_last_page_len,
                    num_qo_heads=self.num_qo_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                )
                pass
                #########
                # FIXME #
                #########
            # ----------------------------------------------------------------
            # 5) Forward pass through all *transformer* layers
            # ----------------------------------------------------------------
            rms_total = 0
            ln_attn_in_total = 0
            kvq_projection_total = 0
            rope_total = 0
            append_kv_total = 0
            attention_total = 0
            residual_total = 0
            ffn_total = 0
            hidden = self.weights["embedding"][input_tensor]
            for layer in range(self.layers):
                # === Self-attention sub-layer ==================================
                start, end = startTimer()

                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)

                rms_total += endTimer(start, end)                
                ln_attn_in_start, ln_attn_in_end = startTimer()

                ln_attn_in = (hidden / rms).to(torch.float16) * self.weights["layernormAttn_weight"][layer]
                
                ln_attn_in_total += endTimer(ln_attn_in_start, ln_attn_in_end)
                kvq_projection_start, kvq_projection_end = startTimer()

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

                kvq_projection_total += endTimer(kvq_projection_start, kvq_projection_end)

                # ---- Rotary positional embedding ---------------------------
                # Use flashinfer.apply_rope_inplace
                # apply ROPE, Note the the theta is set to 500_000.0 and offsets should be the current sequence length before allocate new tokens
                rope_start, rope_end = startTimer()

                flashinfer.apply_rope_inplace(q, k, indptr_tensor, offsets=seq_lens_before_t, rope_theta=500000)
                rope_total += endTimer(rope_start, rope_end)
                #########
                # FIXME #
                #########

                # ---- Append new tokens to *paged* KV-cache ------------------
                # Use flashinfer.get_batch_indices_positions and flashinfer.append_paged_kv_cache
                # if you use get_batch_indices_positions, seq_lens should be the length after the allocation
                append_KV_start, append_KV_end = startTimer()
                batch_indices, positions = flashinfer.get_batch_indices_positions(indptr_tensor, seq_lens_after_t, indptr_tensor[-1].item())

                flashinfer.append_paged_kv_cache(k, v, batch_indices, positions, (self.pool.k_datas[layer], self.pool.v_datas[layer]), kv_indices, kv_indptr, kv_last_page_len, "HND")
                append_kv_total += endTimer(append_KV_start, append_KV_end)
                #########
                # FIXME #
                #########

                # ---- Attention itself --------------------------------------
                # run prefill and decode wrappers. Note that for the prefill wrapper, if qo_indptr does not start with 0, first qo_indptr[0] rows of the output tensor will be empty
                attention_start, attention_end = startTimer()
                out = []
                if (hasDecode):
                    decode_out = self.decode_wrapper.run(q[:num_decode_req], (self.pool.k_datas[layer], self.pool.v_datas[layer]))
                    out.append(decode_out)
                if (hasPrefill):
                    prefill_out = self.prefill_wrapper.run(q[num_decode_req:], (self.pool.k_datas[layer], self.pool.v_datas[layer]))
                    out.append(prefill_out)
                #########
                # FIXME #
                #########
                
                # aggregate the decode and prefill outputs
                attn_out = torch.cat(out, dim = 0)
                attn_out = attn_out.reshape(attn_out.shape[0], -1)
                #########
                # FIXME #
                #########
                attention_total += endTimer(attention_start, attention_end)
                
                # Residual connection
                residual_start, residual_end = startTimer()
                hidden = attn_out.matmul(self.weights["o_proj_weight"][layer].T) + hidden
                residual_total += endTimer(residual_start, residual_end)

                # === FFN sub-layer ==========================================
                ffn_start, ffn_end = startTimer()
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_ffn_in = (hidden / rms).to(torch.float16) * self.weights["layernormFFN_weight"][layer]

                up = ln_ffn_in.matmul(self.weights["up_proj_weight"][layer].T)
                gate = ln_ffn_in.matmul(self.weights["gate_proj_weight"][layer].T)
                hidden = (
                    (up * torch.nn.functional.silu(gate))
                    .matmul(self.weights["down_proj_weight"][layer].T)
                    + hidden
                )
                ffn_total += endTimer(ffn_start, ffn_end)

            # ----------------------------------------------------------------
            # 6) Final language-model head ----------------------------------
            rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
            logits = (
                (hidden / rms).to(torch.float16) * self.weights["model_layernorm_weight"]
            ).matmul(self.weights["lm_head_weight"].T)

            sample_ids = torch.argmax(logits, dim=-1)

            # Extract *new* token for each request (last token of each row)
            last_token_indices = (indptr_tensor[1:] - 1).long()
            ret_val = sample_ids[last_token_indices].cpu()

            overall_time = endTimer(overall_start, overall_end)
            global_var[-1] = [rms_total, ln_attn_in_total, kvq_projection_total, rope_total, append_kv_total, attention_total, residual_total, ffn_total, overall_time]
            return ret_val

    # ---------------------------------------------------------------------
    #  Full batched *generation* loop (prefill + iterative decode)
    # ---------------------------------------------------------------------
    def generate_batched(self, prompts: List[str], rounds: int = 20, shouldBenchmark = False):
        if (shouldBenchmark):
            overall_start, overall_end = startTimer()
        # Build *Request* objects ------------------------------------------------
        requests: List[Request] = []
        for idx, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            requests.append(Request(idx, prompt_ids, rounds))

        # ---- 1) Prefill pass ---------------------------------------------------
        if (shouldBenchmark):
            prefill_start, prefill_end = startTimer()

        prefill_outputs = self.run(requests, num_decode_req=0)
        for i in range(len(requests)):
            new_tok = prefill_outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat(
                [requests[i].output_token_ids, new_tok], dim=0
            )

            # print(f"prefill pass finished - appending first generated token …")
            # print(f">>> Prefill time: {prefill_end_time - prefill_start_time:.4f} seconds")
        
        # You do not need to support adding new request on the fly for this assignment, but if you want to, you can uncomment the following lines
        # requests.append(Request(999, self.tokenizer("Today is", return_tensors="pt").input_ids[0], rounds))
        # # ---- 1.5) Prefill pass for the new request --------------------------
        # prefill_outputs = self.run(requests, num_decode_req=len(requests) - 1)
        # for i in range(len(requests) - 1):
        #     new_tok = prefill_outputs[i].unsqueeze(0)
        #     requests[i].output_token_ids = torch.cat(
        #         [requests[i].output_token_ids, new_tok], dim=0
        #     )
        if (shouldBenchmark):
            prefill_total = endTimer(prefill_start, prefill_end)
        # ---- 2) Iterative decode passes ---------------------------------------
        if (shouldBenchmark):
            decode_start, decode_end = startTimer()
        # clear global variable
        global_var[-1] = [0 for i in range(9)]

        for _ in range(rounds - 1):
            decode_outputs = self.run(requests, num_decode_req=len(requests))
            for i in range(len(requests)):
                new_tok = decode_outputs[i].unsqueeze(0)
                requests[i].output_token_ids = torch.cat(
                    [requests[i].output_token_ids, new_tok], dim=0
                )
        if (shouldBenchmark):
            decode_total = endTimer(decode_start, decode_end)
            ignore = [
            self.tokenizer.decode(r.output_token_ids, skip_special_tokens=True)
            for r in requests
            ]
            overall_total = endTimer(overall_start, overall_end)
            return (overall_total, decode_total, prefill_total)

        # ---- 3) Decode back to text and return -------------------------------
        return [
            self.tokenizer.decode(r.output_token_ids, skip_special_tokens=True)
            for r in requests
        ]


# ---------------------------------------------------------------------------
#  Entry-point (debug / standalone execution)
# ---------------------------------------------------------------------------
def warmup(engine):
    sample_prompts = (
            ["A" * 100] * 10
        )

    generated_texts = engine.generate_batched(sample_prompts, rounds=10, shouldBenchmark = False)
    engine.clear()


def benchmark():
    engine = Engine()
    BATCH_SIZE = 1
    ROUNDS = 0
    warmup(engine)
    decode_length_benchmark(engine)
    prefill_benchmark(engine)
    batch_benchmark(engine)

# With batch size 128, prefill length 1024, decode length 2^5 - 2^10, profile the prefill time and total decode time. Plot end-to-end time curve with the log(decode length) as the x-axis. Which phase is the key bottleneck? In the last decode cycle, which operation takes the longest time for different decode lengths?
def decode_length_benchmark(engine):
    BATCH_SIZE = 128
    print("doing warmup")
    warmup(engine)

    avg_prefill = [0 for _ in range(5, 10 + 1)]
    avg_decode = [0 for _ in range(5, 10 + 1)]
    avg_overall = [0 for _ in range(5, 10 + 1)]
    decode_length_start = 5
    decode_length_end = 10
    decode_data = [[0 for j in range(9)] for i in range(decode_length_end - decode_length_start + 1)]
    num_trials = 5
    for trial in range(num_trials):
        for i in range(decode_length_start, decode_length_end + 1):
            
            PREFILL_LENGTH = 1024
            decode_length = 2 ** i
            print("trial:", trial, "decode length", decode_length)
            # print("prefill length", prefill_length)

            sample_prompts = (
                ["A" * PREFILL_LENGTH] * BATCH_SIZE
            )

            overall_time, decode_time, prefill_time = engine.generate_batched(
                sample_prompts,
                rounds= decode_length + 1,
                shouldBenchmark=True,
            )
            engine.clear()
            temp = addLists(decode_data[i - decode_length_start], global_var[-1])
            decode_data[i - decode_length_start] = temp

            avg_prefill[i - 5] += prefill_time
            avg_decode[i - 5] += decode_time
            avg_overall[i - 5] += overall_time
    for i in range(len(avg_prefill)):
        avg_prefill[i] = avg_prefill[i] / num_trials
        avg_decode[i] = avg_decode[i] / num_trials
        avg_overall[i] = avg_overall[i] / num_trials
    print("prefil:", avg_prefill)
    print("decode:", avg_decode)
    print("overall:", avg_overall)
    for i in range(len(decode_data)):
        for j in range(len(decode_data[i])):
            decode_data[i][j] = decode_data[i][j] / num_trials
    print("ops data:", decode_data)

# With batch size 1, prefill length 2^8 - 2^16, profile the prefill time. Plot end-to-end time curve with log(prefill length) as x-axis. Examine the prefill time breakdown. What operations are the dominating factor for various prefill lengths?

def addLists(first, second):
    if (len(first) != len(second)):
        assert(1 == 2)
    newList = []
    for i in range(len(first)):
        newList.append(first[i] + second[i])
    return newList

def prefill_benchmark(engine):
    BATCH_SIZE = 1
    ROUNDS = 0
    warmup(engine)
    prefill_length_start = 8
    prefill_length_end = 16
    num_trials = 50
    prefill_data = [[0 for j in range(9)] for i in range(prefill_length_end - prefill_length_start + 1)]
    for _ in range(num_trials):
        for i in range(prefill_length_start, prefill_length_end + 1):
            prefill_length = 2 ** i
            # print("prefill length", prefill_length)

            sample_prompts = (
                ["A" * prefill_length] * BATCH_SIZE
            )
            engine.generate_batched(
                sample_prompts,
                rounds=ROUNDS,
                shouldBenchmark=True,
            )
            temp = addLists(prefill_data[i - prefill_length_start], global_var[-1])
            prefill_data[i - prefill_length_start] = temp
            engine.clear()
        
    for i in range(len(prefill_data)):
        for j in range(len(prefill_data[i])):
            prefill_data[i][j] = prefill_data[i][j] / num_trials
    print(prefill_data)


# With batch size 2^0 - 2^10, prefill length 128, decode length 128, plot the end-to-end time curve with log(batch size) as x-axis. Plot the total throughput ((prefill + decode) / time) curve with log(batch size) as x-axis. When does the performance saturate

def batch_benchmark(engine):
    times = [ 0 for i in range(0, 10 + 1)]
    num_reps = 10
    for rep in range(num_reps):
        for i in range(0, 10+1):
            print("rep", rep, "batch size", i)
            batch_size = 2 ** i
            PREFILL_LENGTH = 128
            DECODE_LENGTH = 128
            sample_prompts = (
                ["A" * PREFILL_LENGTH] * batch_size
            )
            start_time = time.perf_counter()
            engine.generate_batched(sample_prompts, rounds=DECODE_LENGTH + 1, shouldBenchmark=False)
            end_time = time.perf_counter()
            times[i] += end_time - start_time
            engine.clear()
    for i in range(len(times)):
        times[i] = times[i] / num_reps
    throughput = []

    for i in range(len(times)):
        batch_size = 2 ** i
        throughput.append((PREFILL_LENGTH + DECODE_LENGTH) * batch_size / times[i])
    print("end to end time: \n", times)
    print("throughput: \n", throughput)

if __name__ == "__main__":
    print("running main")
    shouldBenchmark = False
    if (shouldBenchmark):
        benchmark()
    else:
        # Example batch: the identically phrased prompts + ten location prompts
        sample_prompts = (
            ["Hi, who are you?"] * 100
            + ["The University of Washington is located in"] * 100
        )
        # parser = argparse.ArgumentParser(description="FlashInfer Llama-3 Inference")
        # parser.add_argument("-b", type=int, help="Number of prompts to process in batch")
        # parser.add_argument("-p", type=int, help="Number of tokens in the prefill phase")
        # parser.add_argument("-d", type=int, help="Number of tokens to decode per prompt")

        # args = parser.parse_args()

        # BATCH_SIZE = args.b
        # PREFILL_LENGTH = args.p
        # DECODE_LENGTH = args.d

        # sample_prompts = (
        #     ["A" * PREFILL_LENGTH] * BATCH_SIZE
        # )

        engine = Engine()
        generated_texts = engine.generate_batched(sample_prompts, rounds=10)
        print("ret gen texts")
        for idx, text in enumerate(generated_texts):
            print(f"[request {idx:02d}] {text}\n")

