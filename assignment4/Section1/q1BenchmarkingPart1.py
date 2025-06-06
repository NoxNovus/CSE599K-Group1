from continous_engine import Engine
from continous_scheduler import Scheduler, InputRequest
import random
import time
# BENCHMARKING NAIVE SCHEDULING
engine = Engine()
scheduler = Scheduler(engine, req_batch_size=256)

numPrompts = 10000
basePrompt = "a" * 1024
prompts = [
    InputRequest(basePrompt[:random.randint(1, 1024)], random.randint(1, 1024)) for i in range(numPrompts)
]

# BENCHMARKING WITH NAIVE SCHEDULING
naiveStart = time.time()
for i in range(numPrompts // 256 + 1):
    # schedule 256 prompts at a time, as that is the batch size
    promptStart = i * 256
    # load up a batch
    for j in range(promptStart, promptStart + 256):
        if j >= numPrompts:
            continue
        prompt = prompts[j]
        scheduler.add_req(prompt)
    scheduler.run() # not sure if we need this
    # run a batch (naive, non continious)
    while not scheduler.finished():
        scheduler.run()
naiveEnd = time.time()
naiveTime = naiveEnd - naiveStart
print(f"End to end execution time of naive scheduling with 10000 requests is: {naiveTime:.4f} seconds")

# BENCHMARKING CONTINIOUS SCHEDULING
continiousStart = time.time()
for prompt in prompts:
    scheduler.add_req(prompt)
    scheduler.run()

# Drain remaining requests
while not scheduler.finished():
    scheduler.run()
continiousEnd = time.time()
continiousTime = continiousEnd - continiousStart
print(f"End to end execution time of continious scheduling with 10000 requests is: {naiveTime:.4f} seconds")

