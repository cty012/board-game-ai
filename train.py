import datetime
import os
import torch

import agent
import arg
import game
import model


episode = 0
log_verbose = 1
log_frequency = 50
save_frequency = 500


def log(message, verbose):
    """A very simple logging function."""
    if ((episode + 1) % log_frequency == 0) and (verbose <= log_verbose):
        print(message)


def save_model(network, directory):
    """Save the neural network weights in the specified path."""
    if (save_frequency > 0) and ((episode + 1) % save_frequency == 0):
        os.makedirs(directory, exist_ok=True)
        torch.save(network.state_dict(), f"{directory}/{episode + 1}.pth")


def train(args):
    """Train the model, log outputs, and save weights to file."""
    global episode
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Define all components
    my_game = game.CutAndSlice(args.n)
    my_replay_buffer = agent.ReplayBuffer(args.capacity)
    my_network = model.SimpleCNN()
    my_agent = agent.QLearningAgent(my_network, args.lr, args.gamma)
    my_reward = game.RewardCutAndSlice(args.n, args.reward_weights, args.cluster_size)
    my_epsilon = agent.Epsilon(args.epsilon_max, args.epsilon_min, args.epsilon_decay)

    # Train `args.total_episodes` games
    for episode in range(args.total_episodes):
        my_game.reset()
        player = 0
        loss = 0

        while not my_game.done():
            # Sampling
            state = my_game.get_state()
            action = my_agent.sample(my_game, player, my_epsilon.get())
            next_state = my_game.move(player, action)
            my_game.set_state(next_state)

            # Normalize game state to eliminate player difference
            state_normalized = my_game.normalize(state, player)
            next_state_normalized = my_game.normalize(next_state, player)

            # Calculate reward based on game state or outcome
            reward = my_reward.get_reward(state_normalized, action, next_state_normalized)
            my_replay_buffer.push([state_normalized, action, reward, next_state_normalized, my_game.done()])
            loss = my_agent.train(my_replay_buffer.sample(args.batch_size))

            # Switch player and update epsilon
            player = (player + 1) % 2

        log(f"Episode {episode + 1}/{args.total_episodes}: epsilon={my_epsilon.get()}, loss={loss} ", 1)
        save_model(my_network, f"{args.save_folder}/{timestamp_str}")
        my_epsilon.update()


if __name__ == "__main__":
    train(arg.Args())
