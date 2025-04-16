from typing import Dict, Any, Optional, Callable, List, Tuple
import random
import time
from .node import MCTSNode

class MCTS:
    """
    Monte Carlo Tree Search for multimodal question answering.
    This implementation adapts MCTS to handle multimodal inputs and uses 
    answer outcomes as feedback for the search process.
    """
    
    def __init__(
        self,
        state_generator: Callable[[Dict[str, Any], str], Dict[str, Any]],
        simulator: Callable[[Dict[str, Any]], float],
        time_limit: float = 5.0,
        max_iterations: int = 100,
        exploration_weight: float = 1.0,
    ):
        """
        Initialize the MCTS algorithm.
        
        Args:
            state_generator: Function that generates new states given the current state and action
            simulator: Function that simulates a rollout and returns a reward
            time_limit: Maximum time (in seconds) for search
            max_iterations: Maximum number of iterations for search
            exploration_weight: Weight for UCB1 exploration term
        """
        self.state_generator = state_generator
        self.simulator = simulator
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        
    def search(self, initial_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Perform MCTS to find the best action given an initial state.
        
        Args:
            initial_state: Starting state for search
            
        Returns:
            Tuple of (best action, resulting state)
        """
        root = MCTSNode(state=initial_state, exploration_weight=self.exploration_weight)
        
        start_time = time.time()
        iterations = 0
        
        # Main MCTS loop
        while (time.time() - start_time < self.time_limit and 
               iterations < self.max_iterations):
            
            # 1. Selection: traverse the tree to find a node to expand
            node = self._select(root)
            
            # 2. Expansion: add a new child to the selected node
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)
            
            # 3. Simulation: perform rollout from the selected node
            reward = self._simulate(node)
            
            # 4. Backpropagation: update statistics along the path
            self._backpropagate(node, reward)
            
            iterations += 1
        
        # Select the best action from the root
        best_child = root.best_child(exploration_weight=0.0)  # Pure exploitation
        
        return best_child.action_taken, best_child.state
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node to expand using UCB1.
        
        Args:
            node: Current node
            
        Returns:
            Selected node for expansion
        """
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand a node by adding a child with an untried action.
        
        Args:
            node: Node to expand
            
        Returns:
            New child node
        """
        untried_actions = node.get_untried_actions()
        if not untried_actions:
            return node
        
        action = random.choice(untried_actions)
        new_state = self.state_generator(node.state, action)
        
        return node.add_child(new_state, action)
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate a random rollout from the node.
        
        Args:
            node: Node to simulate from
            
        Returns:
            Reward from simulation
        """
        return self.simulator(node.state)
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Update statistics for all nodes along the path to the root.
        
        Args:
            node: Current node
            reward: Reward from simulation
        """
        while node is not None:
            node.update(reward)
            node = node.parent 