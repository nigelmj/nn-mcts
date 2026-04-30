def generation_worker(
    req_q,
    resp_q,
    pipeline_class,
    num_games,
    num_simulations,
    threshold,
    results_queue,
    wid,
) -> None:
    zero = pipeline_class()

    results = []
    for _ in range(num_games):
        states, policies, values = zero.generate_games(
            num_simulations, threshold, req_q, resp_q, wid
        )
        for state, policy, value in zip(states, policies, values):
            results.append((state, policy, value))
    results_queue.put(results)
