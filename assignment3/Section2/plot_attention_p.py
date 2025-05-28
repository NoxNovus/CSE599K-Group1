import numpy as np
import matplotlib.pyplot as plt

# Sequence lengths (powers of 2)
p_llama2 = 2 ** np.arange(0, 7)   # 2^7 to 2^12
p_llama3 = 2 ** np.arange(0, 7)   # 2^7 to 2^15

# Fake TFLOPs data generator
def fake_tflops(seq_lens, model_factor):
    return np.log2(seq_lens) * model_factor + np.random.normal(0, 0.5, size=len(seq_lens))


llama2_flashattn = [384.78670458576295, 515.2426428972491, 829.1467040482354, 893.03219442742, 931.1010449218036, 943.6979703295556, 934.2979297636283]
llama2_flashinfer = [286.2394391224077, 534.9740063404281, 611.8592325838033, 675.7138004897015, 722.7780406914951, 724.868649468621, 675.012593396948]

llama3_8b_flashattn = [627.0045244852554, 765.5809510647342, 859.6274885608187, 914.6015851706727, 994.8600519952604, 969.4875709406266, 944.1102748439282]
llama3_8b_flashinfer = [407.1355119710881, 504.4773066338948, 579.4665408911345, 660.9777946151548, 642.9558555027825, 663.6818370479222, 680.2854641108]

llama3_70b_flashattn =  [754.9585100537527, 843.6915475426381, 933.0877075768656, 994.5251704383605, 1000.4230512131758, 957.3992596998927, 933.2568497541441]
llama3_70b_flashinfer = [501.6247827823241, 584.5195862636624, 660.0361143106002, 700.3166632075039, 677.0447600477798, 687.8735560311508, 697.8438413342767]



# Plotting setup
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
models = ['LLaMA2-7B', 'LLaMA3-8B', 'LLaMA3-70B']

# LLaMA2-7B plot
axs[0].plot(p_llama2, llama2_flashattn, label='FlashAttention-3', marker='o')
axs[0].plot(p_llama2, llama2_flashinfer, label='FlashInfer', marker='x')
axs[0].set_xscale('log', base=2)
axs[0].set_title(models[0])
axs[0].set_xlabel('batch size')
axs[0].set_ylabel('Compute Utilization (TFLOPs)')
axs[0].set_xticks(p_llama2)
axs[0].set_xticklabels([str(p) for p in p_llama2])
axs[0].legend()
axs[0].grid(True, which='both')

# LLaMA3-8B plot
axs[1].plot(p_llama3, llama3_8b_flashattn, label='FlashAttention-3', marker='o')
axs[1].plot(p_llama3, llama3_8b_flashinfer, label='FlashInfer', marker='x')
axs[1].set_xscale('log', base=2)
axs[1].set_title(models[1])
axs[1].set_xlabel('batch size')
axs[1].set_xticks(p_llama3)
axs[1].set_xticklabels([str(p) for p in p_llama3])
axs[1].legend()
axs[1].grid(True, which='both')

# LLaMA3-70B plot
axs[2].plot(p_llama3, llama3_70b_flashattn, label='FlashAttention-3', marker='o')
axs[2].plot(p_llama3, llama3_70b_flashinfer, label='FlashInfer', marker='x')
axs[2].set_xscale('log', base=2)
axs[2].set_title(models[2])
axs[2].set_xlabel('batch size')
axs[2].set_xticks(p_llama3)
axs[2].set_xticklabels([str(p) for p in p_llama3])
axs[2].legend()
axs[2].grid(True, which='both')

# Overall figure title and layout
fig.suptitle('Prefill Attention Compute Utilization per Layer', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('prefillcomputebs.png')
