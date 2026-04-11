import numpy as np
import torch
import torch.multiprocessing as mp
from src.neural_network import AlphaZeroNetwork


class InferenceWorker(mp.Process):
    def __init__(
                self,
                model_state_dict,
                request_queue,
                response_queues_dict,
                device,
                batch_size,
                size1,
                size2,
                policy_size
    ) -> None:
        super().__init__()
        self.state_dict = model_state_dict
        self.request_queue = request_queue
        self.response_queues_dict = response_queues_dict
        self.device = device
        self.batch_size = batch_size
        self.daemon = True   # Dies when the parent dies
        self.size1 = size1
        self.size2 = size2
        self.policy_size = policy_size

    def run(self) -> None:

        self.model = AlphaZeroNetwork(
            game_size1=self.size1,
            game_size2=self.size2,
            policy_size=self.policy_size
        ).to(self.device)
        self.model.load_state_dict(self.state_dict)

        while True:
            requests = []
            while len(requests) < self.batch_size:
                try:
                    req = self.request_queue.get(timeout=0.0001)
                    if req is None:
                        return
                    requests.append(req)
                except Exception:
                    break

            if not requests:
                continue

            wids, states = zip(*requests)
            states_array = np.stack(states, axis=0)
            states_tensor = torch.tensor(states_array, dtype=torch.float32, device=self.device)
            # print(f"Number of threads is {torch.get_num_threads()}")
            # start = time.time()
            policies, values = self.model.predict_batch(states_tensor)
            # print(f"Time taken to make predictions on {len(wids)} states is {time.time() - start}")
            for i, wid in enumerate(wids):
                self.response_queues_dict[wid].put((policies[i], values[i]))
