import agent
import arg
import game
import model


log_verbose = 1
log_frequency = 100
episode = 0


def log(message, verbose):
    if ((episode + 1) % log_frequency == 0) and (verbose <= log_verbose):
        print(message)


def train(args):
    global episode

    my_game = game.CutAndSlice(args.n)
    my_replay_buffer = agent.ReplayBuffer(args.capacity)
    my_agent = agent.QLearningAgent(model.SimpleCNN(), args.lr, args.gamma)
    my_reward = game.RewardCutAndSlice(args.n, args.reward_weights, args.cluster_size)

    for episode in range(args.total_episodes):  # Train for 1000 episodes
        my_game.reset()
        player = 0  # Player 0 starts

        while not my_game.done():
            # Sampling
            state = my_game.get_state()
            action = my_agent.sample(my_game, player)
            next_state = my_game.move(player, action)
            my_game.set_state(next_state)

            # Normalize game state to eliminate player difference
            state_normalized = my_game.normalize(state, player)
            next_state_normalized = my_game.normalize(next_state, player)

            # Calculate reward based on game state or outcome
            reward = my_reward.get_reward(state_normalized, action, next_state_normalized)
            my_replay_buffer.push([state_normalized, action, reward, next_state_normalized, my_game.done()])
            my_agent.train(my_replay_buffer.sample(args.batch_size))

            # Switch player for next turn
            player = (player + 1) % 2

        log(f"Episode f{episode + 1}/{args.total_episodes}", 1)


if __name__ == "__main__":
    train(arg.Args())
