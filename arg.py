class Args:
    def __init__(self, n=9, cluster_size=8, reward_weights=(0.8, 0.1, 0.1)):
        self.n = n
        self.cluster_size = cluster_size
        self.reward_weights = reward_weights
