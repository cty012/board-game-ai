class Args:
    def __init__(self,
                 total_episodes=10000,
                 n=9,
                 cluster_size=8,
                 reward_weights=(0.8, 0.1, 0.1),
                 lr=0.005,
                 gamma=0.99,
                 capacity=1000,
                 batch_size=64,
                 file_name="{}.???"):

        self.total_episodes = total_episodes
        self.n = n
        self.cluster_size = cluster_size
        self.reward_weights = reward_weights
        self.lr = lr
        self.gamma = gamma
        self.capacity = capacity
        self.batch_size = batch_size
        self.file_name = file_name
