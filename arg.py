class Args:
    def __init__(self,
                 total_episodes=10000,
                 n=9,
                 cluster_size=8,
                 reward_weights=(0.55, 0.3, 0.15),
                 reward_scale=100,
                 lr=0.01,
                 gamma=0.9,
                 epsilon_max=0.9,
                 epsilon_min=0.01,
                 epsilon_decay=0.9997,
                 capacity=3000,
                 batch_size=64,
                 save_folder="saved_models"):

        self.total_episodes = total_episodes
        self.n = n
        self.cluster_size = cluster_size
        self.reward_weights = reward_weights
        self.reward_scale = reward_scale
        self.lr = lr
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.capacity = capacity
        self.batch_size = batch_size
        self.save_folder = save_folder
