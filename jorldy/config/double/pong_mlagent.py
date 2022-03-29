### Double DQN Pong MLAgent Config ###

env = {"name": "pong_mlagent", "time_scale": 12.0}

agent = {
    "name": "double",
    "network": "discrete_q_network",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_ratio": 0.1,
    "buffer_size": 50000,
    "batch_size": 64,
    "start_train_step": 25000,
    "target_update_period": 1000,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 200000,
    "print_period": 2000,
    "save_period": 50000,
    "eval_iteration": 10,
    # distributed setting
    "update_period": 8,
    "num_workers": 16,
}
