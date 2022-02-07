import numpy as np
import random


class FourRooms:

    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 2 if c == 'w' else 0, line)) for line in layout.splitlines()])

        # Four possible actions
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.action_space = np.array([0, 1, 2, 3])
        self.observation_space = np.zeros(np.sum(self.occupancy == 0))
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]

        # Random number generator
        self.rng = np.random.RandomState(1234)

        self.goal = (7, 9)  # East doorway

        # state is defind as (current_x, current_y, goal_x, goal_y)
        states = self.update_state_with_goal()

    def update_state_with_goal(self):
        # state is defind as (current_x, current_y, goal_x, goal_y)
        states = []
        R, C = self.occupancy.shape
        for x in range(R):
            for y in range(C):
                if self.occupancy[x, y] == 0:
                    states.append((x, y))

        self.init_states = states
        self.init_states.remove((self.goal[0], self.goal[1]))
        return states

    def render(self, show_goal=True):
        current_grid = np.array(self.occupancy)
        current_grid[self.current_cell[0], self.current_cell[1]] = -1
        if show_goal:
            goal_cell = self.goal
            current_grid[goal_cell[0], goal_cell[1]] = -1
        return current_grid

    def reset(self):
        self.update_state_with_goal()
        state = self.init_states[self.rng.choice(len(self.init_states))]
        self.current_cell = state
        return state

    def check_available_cells(self, cell):
        available_cells = []

        for action in range(len(self.action_space)):
            next_cell = (cell[0] + self.directions[action][0], cell[1] + self.directions[action][1])

            if not self.occupancy[(next_cell[0], next_cell[1])]:
                available_cells.append(next_cell)

        return available_cells

    def step(self, action):
        '''
        Takes a step in the environment with 2/3 probability. And takes a step in the
        other directions with probability 1/3 with all of them being equally likely.
        '''

        next_cell = (
        self.current_cell[0] + self.directions[action][0], self.current_cell[1] + self.directions[action][1])

        if not self.occupancy[(next_cell[0], next_cell[1])]:

            if self.rng.uniform() < 1 / 3:
                available_cells = self.check_available_cells(self.current_cell)
                self.current_cell = available_cells[self.rng.randint(len(available_cells))]

            else:
                self.current_cell = next_cell

        state = self.current_cell

        # When goal is reached, it is done
        done = (state[0], state[1]) == self.goal
        return state, float(done), done, None


if __name__ == '__main__':
    four_rooms = FourRooms()
    state = four_rooms.reset()
    four_rooms.goal = (12, 0)
    state2 = four_rooms.reset()
    four_rooms.render(show_goal=True)
    four_rooms.step(2)