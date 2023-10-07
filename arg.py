class Args:
    def __init__(self,
                 total_episodes=10000,
                 n=9,
                 cluster_size=8,
                 reward_weights=(0.8, 0.1, 0.1),
                 lr=0.005,
                 gamma=0.99,
                 epsilon_max=0.8,
                 epsilon_min=0.01,
                 epsilon_decay=0.9997,
                 capacity=3000,
                 batch_size=64,
                 save_folder="saved_models"):

        self.total_episodes = total_episodes
        self.n = n
        self.cluster_size = cluster_size
        self.reward_weights = reward_weights
        self.lr = lr
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.capacity = capacity
        self.batch_size = batch_size
        self.save_folder = save_folder
