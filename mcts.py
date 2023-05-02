from abc import ABC, abstractmethod
from collections import defaultdict
import math

#Monte Carlo tree searcher. First rollout the tree then choose a move.
class MCTS:

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
    
    # Choose the best successor of node. (Choose a move in the game)
    def choose(self, node):
        if node.is_terminal():
            raise RuntimeError(f"choose called a terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoids all unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)
    
    # Make the tree one layer better. (Train for one iteration.)
    def rollout(self, node):
        path = self.select(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)

    # Find an unexplored descendent of `node`
    def select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self.uct_select(node)  # descend a layer deeper
    
    # Update the `children` dict with the children of `node`
    def expand(self, node):
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()
    
    # Returns the reward for a random simulation (to completion) of `node`
    def simulate(self, node):
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward
    
    # Send the reward back up to the ancestors of the leaf
    def backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa
    
    # Select a child of node, balancing exploration & exploitation
    def uct_select(self, node):

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(vertex / self.N[n])

        return max(self.children[node], key=uct)

# Node Class for game states
class Node(ABC):
    #All possible successors of this board state
    @abstractmethod
    def find_children(self):
        return set()

    #Random successor of this board state (for more efficient simulation)
    @abstractmethod
    def find_random_child(self):
        return None

    #Returns True if the node has no children
    @abstractmethod
    def is_terminal(self):
        return True

    #Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc
    @abstractmethod
    def reward(self):
        return 0

    #Nodes must be hashable for quick in memory searching
    @abstractmethod
    def __hash__(self):
        return 987654321

    #Nodes must be comparable
    @abstractmethod
    def __eq__(node1, node2):
        return True