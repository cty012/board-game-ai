import agent
import arg
import game
import model


def train(args):
    my_game = game.CutAndSlice(args.n)
    my_agent = agent.QLearningAgent(model.SimpleCNN())
    my_reward = game.RewardCutAndSlice(args.n, args.reward_weights, args.cluster_size)

    for episode in range(1000):  # Train for 1000 episodes
        my_game.reset()
        player = 0  # Player 0 starts

        while not my_game.done():
            # Sampling
            state = my_game.get_state()
            action = my_agent.get_action(my_game, player)
            next_state = my_game.move(player, action)

            # Normalize game state to eliminate player difference
            state_normalized = my_game.normalize(state, player)
            next_state_normalized = my_game.normalize(next_state, player)

            # Calculate reward based on game state or outcome
            reward = my_reward.get_reward(state_normalized, action, next_state_normalized)
            my_agent.train(state_normalized, action, reward, next_state_normalized)  # Train the agent

            # Switch player for next turn
            player = (player + 1) % 2


if __name__ == "__main__":
    my_args = arg.Args()
    train(my_args)
