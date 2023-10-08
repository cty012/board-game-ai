import torch

import game
import game_helper as helper


def test_game():
    print("TEST_GAME:")
    test_game_process_state()
    test_game_move()
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


if __name__ == "__main__":
    test_game()
