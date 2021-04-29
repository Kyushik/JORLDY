### PER SpaceInvaders Config ###

env = {
    "name": "spaceinvaders",
    "render": False,
    "gray_img": True,
    "img_width": 80,
    "img_height": 80,
    "stack_frame": 4,
}

agent = {
    "name": "per",
    "network": "dqn_cnn",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_term": 10000,
    "alpha": 0.6,
    "beta": 0.4,
    "eps": 0.00001
}

train = {
    "training" : True,
    "load_path" : None, #"./logs/breakout/dqn/20201027142347/",
    "train_step" : 10000000,
    "test_step" : 1000000,
    "print_term" : 10,
    "save_term" : 100,
    "test_iteration": 10,
}