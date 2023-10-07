import torch
import game_helper as gh


class CutAndSlice:
    def __init__(self, n, state=None):
        self.n = n
        self.init_board = torch.full((n, n), -1, dtype=torch.int8)
        self.init_board[n-1, 0] = 0
        self.init_board[0, n-1] = 1
        self.kernels = [
            torch.tensor(gh.generate_kernel(d), dtype=torch.int8)
            for d in range(5)
        ]

        self.board = None
        self.potential = [torch.full((n, n), 0, dtype=torch.int8) for i in range(2)]
        self.valid = [torch.full((n, n), 0, dtype=torch.int8) for i in range(2)]
        self.valid_moves = [(), ()]

        if state is None:
            self.reset_state()
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

    def _calc_valid_moves(self, player, board=None, potential=None):
        n = self.n
        if (board is None) or (potential is None):
            board = self.board
            potential = self.potential[player]
            valid = self.valid[player]
            valid.fill_(0)
        else:
            valid = torch.full((n, n), 0, dtype=torch.int8)

        empty_mask = (board == -1).int()
        scatter_kernel = torch.full((n+8, n+8), 0, dtype=torch.int8)

        for i, j in zip(*(potential.nonzero(as_tuple=True))):
            d = potential[i, j]
            scatter_kernel[i+4-d:i+5+d, j+4-d:j+5+d] = self.kernels[d]
            valid += scatter_kernel[4:n+4, 4:n+4]
            scatter_kernel[i+4-d:i+5+d, j+4-d:j+5+d].fill_(0)

        valid *= empty_mask
        result = torch.nonzero(valid * empty_mask, as_tuple=False)
        return tuple([(i.item(), j.item()) for i, j in result])

    def get_state(self):
        return self.board.clone()

    def set_state(self, state):
        assert state.size() == (self.n, self.n), f"Size of state {state.size()} is not equal to {(self.n, self.n)}."
        self.board = state.clone()
        self.potential[0] = self._calc_potential(0)
        self.potential[1] = self._calc_potential(1)
        self.valid_moves[0] = self._calc_valid_moves(0)
        self.valid_moves[1] = self._calc_valid_moves(1)

    def reset_state(self):
        self.set_state(self.init_board)

    def get_valid_moves(self, player):
        return tuple(i * self.n + j for i, j in self.valid_moves[player])

    def move(self, player, action):
        n = self.n
        pos = (action // n, action % n)
        new_board = self.board.clone()

        if self.valid[player][pos] == 0:
            return new_board

        # Calculate new board
        t, b, l, r = min(pos[0], 4), min(n-1-pos[0], 4), min(pos[1], 4), min(n-1-pos[1], 4)
        target = new_board[pos[0]-t:pos[0]+b+1, pos[1]-l:pos[1]+r+1]
        sources = torch.nonzero(self.potential[player][pos[0]-t:pos[0]+b+1, pos[1]-l:pos[1]+r+1] + torch.tensor([
            [-gh.dist((t, l), (i, j)) for j in range(l + r + 1)]
            for i in range(t + b + 1)
        ], dtype=torch.int8) >= 0, as_tuple=False)

        for i, j in sources:
            for grid in gh.grid_pass_thru_by_line((t, l), (i, j)):
                target[tuple(grid)] = player

        return new_board

    def winner(self):
        if torch.sum(torch.eq(self.board, -1)).item() > 0:
            return -1
        elif torch.sum(torch.eq(self.board, 0)) * 2 > self.n * self.n:
            return 0
        else:
            return 1


class RewardCutAndSlice:
    # TODO
    def get_reward(self, state, action, new_state):
        # TODO
        return 0
