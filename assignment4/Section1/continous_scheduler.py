from continous_engine import Engine, Request
import torch

class InputRequest:
    def __init__(self, input_str: str, output_len: int):
        self.input_str = input_str
        self.output_len = output_len
        
class Scheduler:
    def __init__(self, engine: Engine, req_batch_size: int):
        self.engine = engine
        self.req_batch_size = req_batch_size
        self.pending_input_req: list[InputRequest] = []
        self.decode_req: list[Request] = []
        self.scheduled_prefill_req: list[Request] = []
        self.completed: list[Request] = []
        self.unique_req_id: int = 0
    
    def add_req(self, input_req: InputRequest):
        self.pending_input_req.append(input_req)
        
    def finished(self) -> bool:
        return not self.pending_input_req and not self.decode_req

    def get_req_batch_size(self) -> int:
        return len(self.decode_req) + len(self.scheduled_prefill_req)

    def run(self):
        # Schedule new prefill requests until batch is full or no pending inputs
        #########
        # FIXME #
        #########

        # Build the list of requests to send to the engine
        request_list_total = []
        decode_num = 0
        #########
        # FIXME #
        #########
        new_tokens = self.engine.run(request_list_total, decode_num)

        # Append newly generated tokens to each request's output buffer
        #########
        # FIXME #
        #########

        # Check which decode requests have finished and remove from the queue
        ongoing_decode: list[Request] = []
        #########
        # FIXME #
        #########
        self.decode_req = ongoing_decode

        # Move scheduled prefill requests into decode queue
        #########
        # FIXME #
        #########
    
    def print_completed(self):
        for i, req in enumerate(self.completed):
            text = self.engine.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            print(f"Id = {i}: {text}")
