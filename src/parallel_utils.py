import time

def worker_generate_games(
    req_q,
    resp_q,
    pipeline_class,
    num_games,
    num_simulations,
    threshold,
    results_queue,
    wid
) -> None:
    zero = pipeline_class()

    results = []
    worker_time = time.time()
    for index in range(num_games):
        game_time = time.time()
        states, policies, values = zero.generate_games(num_simulations, threshold, req_q, resp_q, wid)
        for state, policy, value in zip(states, policies, values):
            results.append((state, policy, value))
        print(f"[Worker {wid}] Time taken for game {index}: {time.time() - game_time:.3f}")
    print(f"[Worker {wid}] Total time: {time.time() - worker_time:.3f}")
    results_queue.put(results)
