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
        while (len(self.pending_input_req) > 0 and self.get_req_batch_size() < self.req_batch_size):
            inputr = self.pending_input_req.pop(0)
            pids = self.engine.tokenizer(inputr.input_str, return_tensors="pt").input_ids[0]
            self.scheduled_prefill_req.append(Request(self.unique_req_id, pids, inputr.output_len))
            self.unique_req_id += 1

        # Build the list of requests to send to the engine
        request_list_total = []
        decode_num = 0
        #########
        # FIXME #
        #########
        decode_num = len(self.decode_req)
        request_list_total.extend(self.decode_req)
        request_list_total.extend(self.scheduled_prefill_req)


        new_tokens = self.engine.run(request_list_total, decode_num)

        # Append newly generated tokens to each request's output buffer
        #########
        # FIXME #
        #########
        for i, request_i in enumerate(request_list_total):
            new_tok = new_tokens[i].unsqueeze(0)
            request_i.output_token_ids = torch.cat(
                [request_i.output_token_ids, new_tok], dim=0
            )

        # Check which decode requests have finished and remove from the queue
        ongoing_decode: list[Request] = []
        #########
        # FIXME #
        #########
        for req in self.decode_req:
            if (req.current_length >= (req.output_length + req.prompt_length)):
                self.completed.append(req)
                if req.request_id in self.engine.kv_cache_map:
                    self.engine.kv_cache_map[req.request_id].release()
                    del self.engine.kv_cache_map[req.request_id]
            else:
                ongoing_decode.append(req)
        self.decode_req = ongoing_decode

        # Move scheduled prefill requests into decode queue
        #########
        # FIXME #
        #########
        self.decode_req.extend(self.scheduled_prefill_req)
        self.scheduled_prefill_req.clear()
    
    def print_completed(self):
        for i, req in enumerate(self.completed):
            text = self.engine.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            print(f"Id = {i}: {text}")
