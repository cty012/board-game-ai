import torch

import game
import game_helper
import model


def token_to_char(token):
    if token == 0:
        return "\033[91mX\033[0m"
    elif token == 1:
        return "\033[94mX\033[0m"
    else:
        return " "


def display_game(my_game):
    for row in my_game.get_state():
        print("-------------------------------------")
        for token in row:
            print("| " + token_to_char(token), end=" ")
        print("|")
    print("-------------------------------------")


if __name__ == "__main__":
    my_game = game.CutAndSlice(9)
    player = 0
    my_network = model.SimpleCNN()
    my_network.load_state_dict(
        torch.load("saved_models/2023-10-08_00-30-20/5000.pth", map_location=torch.device('cpu')))

    while not my_game.done():
        display_game(my_game)
        print(f"Player {player}'s turn")
        input("<Press ENTER to continue...>")
        print()

        state = game.normalize(my_game.get_state(), player)
        valid_actions = my_game.get_valid_actions(player)
        with torch.no_grad():
            action = 0 if len(valid_actions) == 0 else my_network.sample(state.unsqueeze(0), valid_actions)

        move = game_helper.action_to_move(action)
        print(f"Player {player}: {move[0]} --> {move[1]}")
        my_game.set_state(my_game.move(player, action))

        player = (player + 1) % 2
