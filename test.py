import torch

import game
import game_helper as helper


def test_game():
    print("TEST_GAME:")
    test_game_process_state()
    test_game_valid_moves()
    test_game_move()
    test_game_normalize()
    test_game_reward()
    print()


def test_game_process_state():
    my_game = game.CutAndSlice(6, torch.tensor([
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, 0, -1, 0, -1],
        [-1, 0, -1, 0, 0, 0],
        [-1, 0, -1, -1, 0, -1],
        [0, -1, -1, -1, 0, 0],
        [-1, -1, -1, -1, -1, -1],
    ]))
    ans_potential = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 2, 0],
        [0, 2, 0, 2, 4, 2],
        [0, 2, 0, 0, 3, 0],
        [1, 0, 0, 0, 3, 2],
        [0, 0, 0, 0, 0, 0],
    ])
    # ans_valid = torch.tensor([
    #     [0, 1, 1, 1, 1, 1],
    #     [1, 1, 0, 1, 0, 1],
    #     [1, 0, 1, 0, 0, 0],
    #     [1, 0, 1, 1, 0, 1],
    #     [0, 1, 1, 1, 0, 0],
    #     [1, 1, 1, 1, 1, 1],
    # ])
    # ans_valid_moves = tuple([(i.item(), j.item()) for i, j in torch.nonzero(ans_valid, as_tuple=False)])

    assert torch.equal(my_game.potential[0], ans_potential),\
        f"Potential calculation \n{my_game.potential[0]}\n is not equal to \n{ans_potential}\n"
    # assert my_game.valid_moves[0] == ans_valid_moves,\
    #     f"Valid move calculation {my_game.valid_moves[0]} is not equal to {ans_valid_moves}"

    print("PASS: test_game_process_state")


def test_game_valid_moves():
    my_game = game.CutAndSlice(5, torch.tensor([
        [-1, -1, -1, -1, -1],
        [0, 0, -1, 0, -1],
        [0, 1, 1, 0, 0],
        [1, -1, -1, 0, -1],
        [0, -1, 1, 0, 0],
    ]))
    ans_valid_moves = [
        ((2, 1), (0, 1)),
        ((2, 1), (1, 2)),
        ((2, 1), (3, 1)),
        ((2, 1), (3, 2)),
        ((2, 1), (4, 1)),
        ((2, 2), (0, 2)),
        ((2, 2), (1, 2)),
        ((2, 2), (3, 1)),
        ((2, 2), (3, 2)),
        ((3, 0), (3, 1)),
        ((4, 2), (3, 2)),
        ((4, 2), (4, 1))
    ]

    assert my_game.valid_moves[1] == ans_valid_moves,\
        f"Valid moves {my_game.valid_moves[1]}\n is not equal to {ans_valid_moves}"

    print("PASS: test_game_valid_moves")


def test_game_move():
    my_game = game.CutAndSlice(5, torch.tensor([
        [-1, -1, -1, -1, -1],
        [0, 0, -1, 0, -1],
        [0, 1, 1, 0, 0],
        [1, -1, -1, 0, -1],
        [0, -1, 1, 0, 0],
    ]))
    new_state = my_game.move(0, helper.move_to_action(((2, 3), (3, 1))))
    ans_move = torch.tensor([
        [-1, -1, -1, -1, -1],
        [0, 0, -1, 0, -1],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, -1],
        [0, -1, 1, 0, 0],
    ])

    assert torch.equal(new_state, ans_move),\
        f"New state {new_state}\n is not equal to {ans_move}"

    print("PASS: test_game_move")


def test_game_normalize():
    state = torch.tensor([
        [-1, -1, -1, -1, -1],
        [0, 0, -1, 0, -1],
        [0, 1, 1, 0, 0],
        [1, -1, -1, 0, -1],
        [0, -1, 1, 0, 0],
    ])
    state_normalized = game.normalize(state, 1)
    ans_normalized = torch.tensor([
        [-1, -1, -1, -1, -1],
        [1, 1, -1, 1, -1],
        [1, 0, 0, 1, 1],
        [0, -1, -1, 1, -1],
        [1, -1, 0, 1, 1],
    ])

    assert torch.equal(state_normalized, ans_normalized),\
        f"Normalized state {state_normalized}\n is not equal to {ans_normalized}"

    print("PASS: test_game_normalize")


def test_game_reward():
    state_0 = torch.tensor([
        [-1, -1, -1, -1, -1],
        [0, 0, -1, 0, -1],
        [0, 1, 1, 0, 0],
        [1, -1, -1, 0, -1],
        [0, -1, 1, 0, 0],
    ])
    next_state_0 = torch.tensor([
        [-1, -1, -1, -1, -1],
        [0, 0, -1, 0, -1],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, -1],
        [0, -1, 1, 0, 0],
    ])
    state_1 = game.normalize(state_0, 1)
    next_state_1 = game.normalize(next_state_0, 1)
    my_reward = game.RewardCutAndSlice(5, [0.5, 0.3, 0.2], 4)
    ans_reward_0 = my_reward.get_reward(state_0, None, next_state_0)
    ans_reward_1 = my_reward.get_reward(state_1, None, next_state_1)

    assert ans_reward_0 + ans_reward_1 == 0,\
        f"The sum of the reward {ans_reward_0} + {ans_reward_1} = {ans_reward_0 + ans_reward_1} \n is not equal to zero"

    print("PASS: test_game_reward")


if __name__ == "__main__":
    test_game()
