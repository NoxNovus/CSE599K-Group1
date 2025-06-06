import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import matplotlib.pyplot as plt
import os

# Constants
SEQ_LEN = 1
HIDDEN_DIM = 4096
BATCH_SIZES = [2**i for i in range(5, 16)]  # 32 to 32768

def init_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501' # change if needed
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def run(rank, world_size, return_dict):
    init_distributed(rank, world_size)
    results = []
    
    # do a warm up
    for b in BATCH_SIZES:
        x = torch.randn(b, SEQ_LEN, HIDDEN_DIM, device='cuda')
        dist.barrier()
        torch.cuda.synchronize()
        dist.all_reduce(x)
        torch.cuda.synchronize()

    for b in BATCH_SIZES:
        x = torch.randn(b, SEQ_LEN, HIDDEN_DIM, device='cuda')
        dist.barrier()
        torch.cuda.synchronize()
        start = time.time()
        dist.all_reduce(x)
        torch.cuda.synchronize()
        duration = time.time() - start

        bytes_moved = x.numel() * 4  # float32
        bandwidth = bytes_moved / duration / 1e9  # GB/s
        if rank == 0:
            results.append((b, bandwidth))

    if rank == 0:
        return_dict.update({b: bw for b, bw in results})
    dist.destroy_process_group()

def main():
    world_size = 2
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run, args=(rank, world_size, return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if return_dict:
        results = sorted(return_dict.items())
        batch_sizes, bandwidths = zip(*results)
        plt.plot(batch_sizes, bandwidths, marker='o')
        plt.xlabel("Batch Size (b)")
        plt.ylabel("Bandwidth (GB/s)")
        plt.title("AllReduce Bandwidth Utilization vs Batch Size")
        plt.grid(True)
        # plt.xscale('log', base=2)
        plt.tight_layout()
        plt.savefig('profile_allreduce.png')

if __name__ == "__main__":
    main()
