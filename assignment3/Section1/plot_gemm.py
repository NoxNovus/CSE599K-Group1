import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
# batch_size,N,K,library,tflops
df = pd.read_csv('gemm_perf.csv')

# Get all unique (N, K) shapes
shapes = df[['N', 'K']].drop_duplicates().values.tolist()

# Plot for each shape
for N, K in shapes:
    shape_df = df[(df['N'] == N) & (df['K'] == K)]
    batch_sizes = sorted(shape_df['batch_size'].unique())

    plt.figure(figsize=(8, 5))
    
    for lib in ['cutlass', 'cublas']:
        lib_df = shape_df[shape_df['library'] == lib]
        plt.plot(lib_df['batch_size'], lib_df['tflops'], marker='o', label=lib.capitalize())

    plt.title(f'Performance for Shape N={N}, K={K}')
    plt.xlabel('Batch Size')
    plt.ylabel('TFLOPS')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
