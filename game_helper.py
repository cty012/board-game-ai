import torch


def dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def generate_kernel(d):
    return [
        [1 if dist((d, d), (i, j)) <= d else 0 for j in range(2 * d + 1)]
        for i in range(2 * d + 1)
    ]


def move_to_action(move):
    (i1, j1), (i2, j2) = move
    return (i1 * 9 + j1) * 81 + (i2 * 9 + j2)


def action_to_move(action):
    src = action // 81
    dest = action % 81
    i1 = src // 9
    j1 = src % 9
    i2 = dest // 9
    j2 = dest % 9
    return (i1, j1), (i2, j2)


def grid_pass_thru_by_line(start, end):
    """Get all grids passed through by the line start---end"""
    x1, y1 = start
    x2, y2 = end
    result = grid_pass_thru_by_line_helper1((x2 - x1, y2 - y1))
    return result if result.size(0) == 0 else result + torch.tensor([x1, y1])


def grid_pass_thru_by_line_helper1(end):
    """If start at (0, 0)"""
    x, y = end
    direction = torch.tensor([-1 if x < 0 else 1, -1 if y < 0 else 1])
    result = grid_pass_thru_by_line_helper2((abs(x), abs(y)))
    return result if result.size(0) == 0 else result * direction


def grid_pass_thru_by_line_helper2(end):
    """If start at (0, 0) and end[0] >= 0 and end[1] >= 0"""
    result = []
    if (end[0] == end[1]) and (end[0] <= 2):
        result = [[i, i] for i in range(end[0] + 1)]
    elif end[0] == 0:
        result = [[0, j] for j in range(end[1] + 1)]
    elif end[1] == 0:
        result = [[i, 0] for i in range(end[0] + 1)]
    elif end == (1, 2):
        result = [[0, 0], [0, 1], [1, 1], [1, 2]]
    elif end == (1, 3):
        result = [[0, 0], [0, 1], [1, 2], [1, 3]]
    elif end == (2, 1):
        result = [[0, 0], [1, 0], [1, 1], [2, 1]]
    elif end == (3, 1):
        result = [[0, 0], [1, 0], [2, 1], [3, 1]]
    return torch.tensor(result)
