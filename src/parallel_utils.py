import torch
import time

def worker_generate_games(model_state_dict, game_class, pipeline_class, num_games, num_simulations, threshold):
    torch.set_num_threads(1)
    zero = pipeline_class()

    zero.build_network(2)
    zero.model.load_state_dict(model_state_dict)
    zero.model.eval()

    results = []
    worker_time = time.time()
    for index in range(num_games):
        game_time = time.time()
        states, policies, values = zero.generate_games(num_simulations, threshold)
        for state, policy, value in zip(states, policies, values):
            results.append((state, policy, value))
        print("Time taken for game", index, "is", time.time() - game_time)
    print("Worker time:", time.time() - worker_time)
    return results
