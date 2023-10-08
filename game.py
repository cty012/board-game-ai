import math
import torch

import game_helper as helper


def normalize(board, player):
    if player:
        return board.clone()
    mask_neg1 = (board == -1).int()
    mask_0 = (board == 0).int()
    return mask_neg1 * -1 + mask_0 * 1


class CutAndSlice:
    def __init__(self, n, state=None):
        self.n = n
        self.init_board = torch.full((n, n), -1, dtype=torch.int8)
        self.init_board[n-1, 0] = 0
        self.init_board[0, n-1] = 1
        self.kernels = [
            torch.tensor(helper.generate_kernel(d), dtype=torch.int8)
            for d in range(5)
        ]

        self.board = None
        self.potential = [torch.full((n, n), 0, dtype=torch.int8) for i in range(2)]
        self.valid_moves = [(), ()]

        if state is None:
            self.reset()
        else:
            self.set_state(state)

    def _calc_potential(self, player):
        n = self.n
        potential = torch.full((n, n), 0, dtype=torch.int8)

        friendly_mask = (self.board == player).int()
        scatter_kernel = torch.full((n+2, n+2), 0, dtype=torch.int8)

        for i, j in zip(*(friendly_mask.nonzero(as_tuple=True))):
            scatter_kernel[i:i+3, j:j+3] = self.kernels[1]
            potential += scatter_kernel[1:n+1, 1:n+1]
            scatter_kernel[i:i+3, j:j+3].fill_(0)

        return torch.clamp(potential * friendly_mask, max=4)

    def _calc_valid_moves(self, player):
        n = self.n
        empty_mask = (self.board == -1).int()
        scatter_kernel = torch.full((n+8, n+8), 0, dtype=torch.int8)
        result = []

        for i, j in zip(*(self.potential[player].nonzero(as_tuple=True))):
            d = self.potential[player][i, j]
            scatter_kernel[i+4-d:i+5+d, j+4-d:j+5+d] = self.kernels[d]
            valid = torch.nonzero(scatter_kernel[4:n+4, 4:n+4] * empty_mask, as_tuple=False)
            for r, c in valid:
                result.append(((i.item(), j.item()), (r.item(), c.item())))
            scatter_kernel[i+4-d:i+5+d, j+4-d:j+5+d].fill_(0)

        return result

    def get_state(self):
        return self.board.clone()

    def set_state(self, state):
        assert state.size() == (self.n, self.n), f"Size of state {state.size()} is not equal to {(self.n, self.n)}."
        self.board = state.clone()
        self.potential[0] = self._calc_potential(0)
        self.potential[1] = self._calc_potential(1)
        self.valid_moves[0] = self._calc_valid_moves(0)
        self.valid_moves[1] = self._calc_valid_moves(1)

    def reset(self):
        self.set_state(self.init_board)

    def get_valid_actions(self, player):
        return tuple([helper.move_to_action(move) for move in self.valid_moves[player]])

    def is_valid(self, player, action):
        src, dest = helper.action_to_move(action)
        return self.board[src] == player and self.board[dest] == -1 and \
            helper.dist(src, dest) <= self.potential[player][src]

    def move(self, player, action):
        new_board = self.board.clone()

        if not self.is_valid(player, action):
            return new_board

        # Calculate new board
        src, dest = helper.action_to_move(action)
        for grid in helper.grid_pass_thru_by_line(src, dest):
            new_board[tuple(grid)] = player

        return new_board

    def winner(self):
        if torch.sum(torch.eq(self.board, -1)).item() > 0:
            return -1
        elif torch.sum(torch.eq(self.board, 0)) * 2 > self.n * self.n:
            return 0
        else:
            return 1

    def done(self):
        return self.winner() != -1


class RewardCutAndSlice:
    def __init__(self, n, weights, cluster_size):
        self.n = n
        self.weights = weights
        self.cluster_size = cluster_size

        # Generate kernels for faster cluster analysis
        self.cluster_kernel_size = []
        self.cluster_kernels = [[], []]
        self.cluster = torch.full((self.n, self.n), 0, dtype=torch.int8)
        self._generate_cluster_kernels(self.cluster_size)

    def _generate_cluster_kernels(self, cluster_size):
        self.cluster_kernel_size = []
        self.cluster_kernels = [[], []]

        # Generate kernel size
        prev = self.n + 1
        for rows in range(1, min(cluster_size, self.n) + 1):
            cols = math.ceil(cluster_size / rows)
            if cols >= prev:
                continue
            self.cluster_kernel_size.append((rows, cols))
            prev = cols

        # Generate kernels
        for rows, cols in self.cluster_kernel_size:
            self.cluster_kernels[0].append(torch.full((rows, cols), 0, dtype=torch.int8))
            self.cluster_kernels[1].append(torch.full((rows, cols), 1, dtype=torch.int8))

    def _winning_reward(self, next_state):
        return CutAndSlice(self.n, next_state).winner() == 0

    def _quantity_reward(self, state, next_state):
        score = torch.sum(torch.eq(state, 0)) - torch.sum(torch.eq(state, 1))
        new_score = torch.sum(torch.eq(next_state, 0)) - torch.sum(torch.eq(next_state, 1))
        return (new_score - score) / (self.n ** 2)

    def _cluster_analysis(self, state, player):
        self.cluster.fill_(0)
        for k in range(len(self.cluster_kernel_size)):
            # Cluster analysis for k-th kernal
            rows, cols = self.cluster_kernel_size[k]
            for i in range(self.n - rows + 1):
                for j in range(self.n - cols + 1):
                    if torch.equal(state[i:i+rows, j:j+cols], self.cluster_kernels[player][k]):
                        self.cluster[i, j] = 1
        return torch.sum(self.cluster)

    def _cluster_reward(self, state, next_state):
        return (self._cluster_analysis(next_state, 0) - self._cluster_analysis(next_state, 1)) - \
            (self._cluster_analysis(state, 0) - self._cluster_analysis(state, 1))

    def get_reward(self, state, action, next_state):
        return self._winning_reward(next_state) * self.weights[0] + \
            self._quantity_reward(state, next_state) + \
            self._cluster_reward(state, next_state)
