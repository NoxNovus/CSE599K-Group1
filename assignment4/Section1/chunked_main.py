from chunked_engine import Engine
from chunked_scheduler import Scheduler, InputRequest
import numpy as np

def sample_lognormal_ints(l, mean=6.0, sigma=0.7):
    samples = np.random.lognormal(mean, sigma, size=l)
    return [int(round(x)) for x in samples]


engine = Engine()
scheduler = Scheduler(engine, token_batch_size=1024)

sample_prompts = ["Today is a rainy day"] * 1024 + ["UW is"] * 1024

# Enqueue and run
for prompt in sample_prompts:
    scheduler.add_req(InputRequest(prompt, output_len=100))
    # scheduler.run()

# Drain remaining requests
while not scheduler.finished():
    scheduler.run()

# scheduler.print_completed()
