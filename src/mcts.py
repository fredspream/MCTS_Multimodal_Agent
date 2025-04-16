import math
import time
import logging
import random
from typing import Dict, Any, List, Optional, Callable, Tuple

class Node:
    """
    A node in the Monte Carlo Tree Search tree.
    Each node represents a state in the search space.
    """
    
    def __init__(
        self,
        state: Dict[str, Any],
        parent=None,
        action: str = None,
        exploration_weight: float = 1.0
    ):
        """
        Initialize a new node.
        
        Args:
            state: The state represented by this node
            parent: The parent node
            action: The action that led to this state
            exploration_weight: Weight for exploration vs exploitation
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.exploration_weight = exploration_weight
        
        # Initialize node statistics
        self.total_value = 0.0
        self.visit_count = 0
        
        # Initialize children
        self.children: List[Node] = []
        self.unexplored_actions: List[str] = []
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions from this node have been explored."""
        return len(self.unexplored_actions) == 0
    
    def is_terminal(self) -> bool:
        """
        Check if this node is terminal.
        A node is terminal if it represents an end state (e.g., a final answer).
        """
        # Check if the state has a final answer
        return "final_answer" in self.state and self.state["final_answer"]
    
    def best_child(self, exploration_weight: Optional[float] = None) -> 'Node':
        """
        Select the best child node according to the UCB formula.
        
        Args:
            exploration_weight: Weight for exploration vs exploitation
                                (if None, use the node's default)
                                
        Returns:
            The best child node
        """
        if not self.children:
            raise ValueError("Node has no children")
            
        # Use the node's exploration weight if none provided
        if exploration_weight is None:
            exploration_weight = self.exploration_weight
            
        # Select child with highest UCB value
        def ucb_score(n: Node) -> float:
            # UCB formula: exploitation + exploration
            exploitation = n.total_value / n.visit_count if n.visit_count > 0 else 0.0
            exploration = math.sqrt(2.0 * math.log(self.visit_count) / n.visit_count) if n.visit_count > 0 else float('inf')
            return exploitation + exploration_weight * exploration
            
        return max(self.children, key=ucb_score)
    
    def expand(self, actions: List[str], state_generator: Callable) -> 'Node':
        """
        Expand the node by adding a child node for an unexplored action.
        
        Args:
            actions: List of possible actions
            state_generator: Function to generate a new state given current state and action
            
        Returns:
            The newly created child node
        """
        # If this is the first expansion, initialize unexplored actions
        if not self.unexplored_actions and not self.children:
            self.unexplored_actions = actions.copy()
            
        # If already fully expanded, return None
        if not self.unexplored_actions:
            return None
            
        # Choose an unexplored action
        action = self.unexplored_actions.pop()
        
        # Generate the new state
        new_state = state_generator(self.state, action)
        
        # Create a child node
        child = Node(
            state=new_state,
            parent=self,
            action=action,
            exploration_weight=self.exploration_weight
        )
        
        # Add the child to this node's children
        self.children.append(child)
        
        return child
    
    def update(self, reward: float) -> None:
        """
        Update the node's statistics based on the reward.
        
        Args:
            reward: The reward value
        """
        self.visit_count += 1
        self.total_value += reward
    
    def __str__(self) -> str:
        """String representation of the node."""
        action = self.action if self.action else "root"
        value = self.total_value / self.visit_count if self.visit_count > 0 else 0
        return f"Node({action}, value={value:.3f}, visits={self.visit_count})"


class MCTS:
    """
    Monte Carlo Tree Search algorithm implementation.
    This implementation is designed for question answering with tools.
    """
    
    def __init__(
        self,
        state_generator: Callable,
        simulator: Callable,
        available_actions: List[str] = None,
        exploration_weight: float = 1.0,
        time_limit: float = 10.0,
        max_iterations: int = 100
    ):
        """
        Initialize the MCTS algorithm.
        
        Args:
            state_generator: Function to generate a new state given current state and action
            simulator: Function to simulate and evaluate a state
            available_actions: List of available actions (default: caption, ocr, answer)
            exploration_weight: Weight for exploration vs exploitation
            time_limit: Time limit for search in seconds
            max_iterations: Maximum number of iterations
        """
        self.logger = logging.getLogger(__name__)
        self.state_generator = state_generator
        self.simulator = simulator
        self.available_actions = available_actions or ["caption", "ocr", "answer"]
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit
        self.max_iterations = max_iterations
    
    def search(self, initial_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Run the MCTS algorithm from the initial state.
        
        Args:
            initial_state: The initial state
            
        Returns:
            Tuple of (best_action, final_state)
        """
        self.logger.info("Starting MCTS search")
        
        # Create the root node
        root = Node(
            state=initial_state,
            exploration_weight=self.exploration_weight
        )
        
        # Run the search algorithm
        start_time = time.time()
        iteration = 0
        
        while (
            iteration < self.max_iterations and 
            time.time() - start_time < self.time_limit
        ):
            # Selection and expansion phase
            node = self._select_and_expand(root)
            
            # Simulation phase
            reward = self.simulator(node.state)
            
            # Backpropagation phase
            self._backpropagate(node, reward)
            
            iteration += 1
            
        # Calculate search statistics
        elapsed_time = time.time() - start_time
        self.logger.info(f"MCTS search completed: {iteration} iterations in {elapsed_time:.2f} seconds")
        
        # Select the best action
        if not root.children:
            # If no children, just return the default action "answer"
            self.logger.warning("Root node has no children, using default action")
            final_state = self.state_generator(initial_state, "answer")
            return "answer", final_state
        
        # Get the best child node (using exploitation only)
        best_child = max(
            root.children, 
            key=lambda n: n.total_value / n.visit_count if n.visit_count > 0 else 0
        )
        
        # Log the selected action
        best_action = best_child.action
        visit_count = best_child.visit_count
        value = best_child.total_value / visit_count if visit_count > 0 else 0
        self.logger.info(f"Selected action: {best_action} (value: {value:.3f}, visits: {visit_count})")
        
        return best_action, best_child.state
    
    def _select_and_expand(self, root: Node) -> Node:
        """
        Select a node to expand using the UCB formula,
        then expand it by adding a child node.
        
        Args:
            root: The root node
            
        Returns:
            The newly created leaf node
        """
        # Start at the root
        node = root
        
        # Traverse the tree to find a node to expand
        while not node.is_terminal():
            # If the node is not fully expanded, expand it
            if not node.is_fully_expanded():
                return node.expand(self.available_actions, self.state_generator)
                
            # Otherwise, select the best child
            node = node.best_child()
            
        # Return the selected node
        return node
    
    def _backpropagate(self, node: Node, reward: float) -> None:
        """
        Backpropagate the reward up the tree.
        
        Args:
            node: The node to start from
            reward: The reward value
        """
        # Update all nodes in the path from the leaf to the root
        while node is not None:
            node.update(reward)
            node = node.parent 