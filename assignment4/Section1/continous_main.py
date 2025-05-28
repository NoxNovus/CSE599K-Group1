from continous_engine import Engine
from continous_scheduler import Scheduler, InputRequest
engine = Engine()
scheduler = Scheduler(engine, req_batch_size=128)

sample_prompts = ["Today is a rainy day"] * 128 + ["UW is"] * 128

# Enqueue and run
for prompt in sample_prompts:
    scheduler.add_req(InputRequest(prompt, output_len=100))
    scheduler.run()

# Drain remaining requests
while not scheduler.finished():
    scheduler.run()

scheduler.print_completed()
