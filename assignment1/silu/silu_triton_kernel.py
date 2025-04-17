import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(input_ptr,
                output_ptr,
                n_elements,
                BLOCK_SIZE: tl.constexpr,
                ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # mask for out of bound accesses
    mask = offsets < n_elements
    # load input from DRAM
    input = tl.load(input_ptr + offsets, mask = mask)
    output = input / (1 + tl.exp(-input))
    # write output back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)
    

    


def silu_triton(input):
    # preallocate the output
    output = torch.empty_like(input)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    silu_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output