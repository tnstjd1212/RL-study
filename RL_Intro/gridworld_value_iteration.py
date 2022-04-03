import numpy as np

ACTIONS = ('U', 'D', 'L', 'R')
DELTA_THRESHOLD = 1e-3
GAMMA = 0.9

TRANSITION_PROB = 0.8

class Grid: 
	def __init__(self, rows, cols, start):
		self.rows = rows
		self.cols = cols
		self.i = start[0]
		self.j = start[1]

	def set(self, rewards, actions):
		self.rewards = rewards
		self.actions = actions

	def set_state(self, s):
		self.i = s[0]
		self.j = s[1]

	def current_state(self):
		return (self.i, self.j)

	# check state is finish 
	def is_terminal(self, s):
		return s not in self.actions

	def move(self, action):
		if action in self.actions[(self.i, self.j)]:
			if action == 'U':
				self.i -= 1
			elif action == 'D':
				self.i += 1
			elif action == 'R':
				self.j += 1
			elif action == 'L':
				self.j -= 1
		return self.rewards.get((self.i, self.j), 0)

	def all_states(self):
		# possibly buggy but simple way to get all states
		# either a position that has possible next actions
		# or a position that yields a reward
		return set(self.actions.keys()) | set(self.rewards.keys())

	def set_transition_prob(self, transition_prob):
		self.transition_prob_intended = transition_prob
		self.transition_prob_unintended = (1 - transition_prob) / 2

	def get_transition_prob(self, action):
		currentState = (self.i, self.j)
		transitions = []
		if self.is_terminal(currentState):
			return transitions
		# intended
		if action in self.actions[currentState]:
			r = self.move(action)
			transitions.append((self.transition_prob_intended, r, (self.i, self.j)))
			self.set_state(currentState)
		# unintended
		if (action == 'U' )| (action == 'D'):
			if 'L' in self.actions[currentState]:
				r = self.move('L')
				transitions.append((self.transition_prob_unintended, r, (self.i, self.j)))
				self.set_state(currentState)
			else:
				transitions.append((self.transition_prob_unintended, 0, currentState))
			if 'R' in self.actions[currentState]:
				r = self.move('R')
				transitions.append((self.transition_prob_unintended, r, (self.i, self.j)))
				self.set_state(currentState)
			else:
				transitions.append((self.transition_prob_unintended, 0, currentState))
		elif (action == 'L') | (action == 'R'):
			if 'U' in self.actions[currentState]:
				r = self.move('U')
				transitions.append((self.transition_prob_unintended, r, (self.i, self.j)))
				self.set_state(currentState)
			else:
				transitions.append((self.transition_prob_unintended, 0, currentState))
			if 'D' in self.actions[currentState]:
				r = self.move('D')
				transitions.append((self.transition_prob_unintended, r, (self.i, self.j)))
				self.set_state(currentState)
			else:
				transitions.append((self.transition_prob_unintended, 0, currentState))

		return transitions

def best_action_value(grid, V, s):
	bestAction = None
	bestValue = float('-inf')
	grid.set_state(s)
	for action in ACTIONS:
		transitions = grid.get_transition_prob(action)
		rewardExpectation = 0
		valueExpectation = 0
		# calc expectation of reward, value
		for (prob, r, nextState) in transitions:
			rewardExpectation += prob * r
			valueExpectation += prob * V[nextState]
		# Bellman eq
		value = rewardExpectation + GAMMA * valueExpectation
		if value > bestValue:
			bestValue = value
			bestAction = action
	return bestAction, bestValue

def standard_grid():
	grid = Grid(3, 4, (2, 0))
	rewards = {(0, 3): 1, (1, 3): -1}
	actions = {
		(0, 0): ('D', 'R'),
		(0, 1): ('L', 'R'),
		(0, 2): ('L', 'D', 'R'),
		(1, 0): ('U', 'D'),
		(1, 2): ('U', 'D', 'R'),
		(2, 0): ('U', 'R'),
		(2, 1): ('L', 'R'),
		(2, 2): ('L', 'R', 'U'),
		(2, 3): ('L', 'U'),
	}
	grid.set(rewards, actions)
	grid.set_transition_prob(TRANSITION_PROB)
	return grid

def print_values(V, grid):
	for i in range(grid.rows):
		print("---------------------------")
		for j in range(grid.cols):
			value = V.get((i, j), 0)
			if value >= 0:
				print("%.2f | " % value, end = "")
			else:
				print("%.2f | " % value, end = "") # -ve sign takes up an extra space
		print("")

def print_policy(P, grid):
	for i in range(grid.rows):
		print("---------------------------")
		for j in range(grid.cols):
			action = P.get((i, j), ' ')
			print("  %s  |" % action, end = "")
		print("")

if __name__ == '__main__':
    
	grid = standard_grid()

	
	print("\n reward: ")
	print_values(grid.rewards, grid)

	# get init policy
	policy = {}
	for s in grid.actions.keys():
		policy[s] = np.random.choice(ACTIONS)

	
	print("\n init policy:")
	print_policy(policy, grid)

	# get init value function
	V = {}
	states = grid.all_states()
	for s in states:
		# V[s] = 0
		if s in grid.actions:
			V[s] = np.random.random()
		else:
			
			V[s] = 0

	# value iteration
	i = 0
	while True:
		maxChange = 0
		for s in grid.actions.keys():
			oldValue = V[s]
			_, newValue = best_action_value(grid, V, s)
			V[s] = newValue
			maxChange = max(maxChange, np.abs(oldValue - V[s]))

		print("\n%i iteration" % i, end = "\n")
		print_values(V, grid)
		i += 1 

		if maxChange < DELTA_THRESHOLD:
			break

	
	for s in policy.keys():
		bestAction, _ = best_action_value(grid, V, s)
		policy[s] = bestAction

	
	print("\n value function: ")
	print_values(V, grid)

	print("\n policy: ")
	print_policy(policy, grid)