import matplotlib.pyplot as plt

# Synthetic data generated, to be filled in one we get actual data
iteration_ids = list(range(1, 10))
continuous_times = [20 + (i % 10) + (i * 0.05) for i in iteration_ids] 
chunked_times = [25 + ((i % 5) * 1.5) + (i * 0.03) for i in iteration_ids]

plt.figure(figsize=(12, 6))
plt.scatter(iteration_ids, continuous_times, label='Continuous Batching', color='blue', marker='o', alpha=0.7)
plt.scatter(iteration_ids, chunked_times, label='Chunked Prefill', color='orange', marker='x', alpha=0.7)

plt.xlabel('Iteration ID')
plt.ylabel('Iteration Time (ms)')
plt.title('Iteration Time: Continuous Batching vs. Chunked Prefill')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("q2plot.png")
plt.show()
