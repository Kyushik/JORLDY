### Munchausen DQN CartPole Config ###

env = {
    "name": "cartpole",
    "action_type": "discrete",
    "render": False,
}

agent = {
    "name": "m_dqn",
    "network": "discrete_q_network",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_ratio": 0.2,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 2000,
    "target_update_period": 500,
    # M-DQN Parameters
    "alpha": 0.9,
    "tau": 0.03,
    "l_0": -1,
}

optim = {
    "name": "adam",
    "lr": 0.0001,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
    # distributed setting
    "update_period": 32,
    "num_workers": 8,
}
