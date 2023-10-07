import agent
import game
import model

n = 9

my_game = game.CutAndSlice(n)
my_agent = agent.QLearningAgent(model.SimpleCNN())
my_reward = game.RewardCutAndSlice()

for episode in range(1000):  # Train for 1000 episodes
    done = False
    player = 0  # Player 0 starts

    while not done:
        state = my_game.get_state()  # Get current state
        action = my_agent.get_action(my_game, player)  # Get action from agent

        next_state = my_game.move(player, action)  # Make the move
        reward = my_reward.get_reward(state, action, next_state)  # Calculate reward based on game state or outcome
        done = my_game.winner() != -1  # Check if the game is done

        my_agent.train(state, action, reward, next_state)  # Train the agent

        player = (player + 1) % 2  # Switch player for next turn
