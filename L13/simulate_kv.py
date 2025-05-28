import random, matplotlib.pyplot as plt
from collections import deque

# ----------------------------------
# Simulation parameters
# ----------------------------------
INPUT_LEN   = 1_000
CAPACITY    = 512_000
SIM_STEPS   = 50_000
QUEUE_SIZE  = 120_000          # plenty of pending requests
random.seed(0)

# Preset queue of requests (all identical prompts)
request_queue = deque([None]*QUEUE_SIZE)   # only need placeholders

class Request:
    __slots__ = ("generated", "output_len")
    def __init__(self, output_len):
        self.generated  = 0
        self.output_len = output_len
    @property
    def remaining(self): return self.output_len - self.generated
    @property
    def kv_now(self):    return INPUT_LEN + self.generated
    @property
    def full_ctx(self):  return INPUT_LEN + self.output_len
    @property
    def finished(self):  return self.generated >= self.output_len

def can_admit(candidate_out, active, kv_usage):
    rems = [r.remaining for r in active] + [candidate_out]
    ctxs = [r.full_ctx  for r in active] + [INPUT_LEN + candidate_out]
    usage = kv_usage + INPUT_LEN
    while rems:
        delta = min(rems)
        usage += len(rems)*delta
        if usage > CAPACITY:
            return False
        # evict finished batch(es)
        new_rems, new_ctxs = [], []
        for rem, ctx in zip(rems, ctxs):
            rem -= delta
            if rem == 0:
                usage -= ctx
            else:
                new_rems.append(rem)
                new_ctxs.append(ctx)
        rems, ctxs = new_rems, new_ctxs
    return True

# ----------------------------------
active = []
kv_usage = 0
steps, kv_series, batch_series = [], [], []

for step in range(SIM_STEPS):
    if request_queue:
        cand_out = random.randint(0, 4000)
        if cand_out > 0 and can_admit(cand_out, active, kv_usage):
            request_queue.popleft()
            active.append(Request(cand_out))
            kv_usage += INPUT_LEN

    finished = []
    for req in active:
        req.generated += 1
        kv_usage += 1
        if req.finished:
            finished.append(req)

    for req in finished:
        kv_usage -= req.kv_now
        active.remove(req)

    # metrics
    steps.append(step)
    kv_series.append(kv_usage)
    batch_series.append(len(active))

# ----------------------------------
# Plot KV usage
plt.figure(figsize=(10,4))
plt.plot(steps, kv_series, linewidth=1.0, label="KV cache usage")
plt.axhline(CAPACITY, linestyle="--", linewidth=1, label="Capacity")
plt.title("KV‑cache usage over 50 k decode cycles")
plt.xlabel("Decoding step")
plt.ylabel("Tokens in KV cache")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# Plot batch size with average annotation
avg_batch = sum(batch_series)/len(batch_series)

plt.figure(figsize=(10,4))
plt.plot(steps, batch_series, linewidth=1.0, label="Active batch size")
plt.axhline(avg_batch, color="grey", linestyle="--", linewidth=1.2, label="Average")
# annotate near left edge
plt.text(steps[0], avg_batch + 3, f"avg ≈ {avg_batch:.1f}", va='bottom', ha='left')
plt.title("Total active requests over 50 k cycles")
plt.xlabel("Decoding step")
plt.ylabel("Batch size")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
