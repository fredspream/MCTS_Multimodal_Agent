import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

class MCTSNode:
    """
    Node for Monte Carlo Tree Search in multimodal question answering.
    Each node represents a state in the search tree.
    """
    
    def __init__(
        self, 
        state: Dict[str, Any],
        parent: Optional['MCTSNode'] = None,
        action_taken: Optional[str] = None,
        exploration_weight: float = 1.0
    ):
        """
        Initialize an MCTS node.
        
        Args:
            state: Dictionary containing the current state information
            parent: Parent node
            action_taken: Action taken to reach this node from parent
            exploration_weight: Weight for UCB1 exploration term
        """
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: List['MCTSNode'] = []
        self.exploration_weight = exploration_weight
        
        # MCTS statistics
        self.visit_count = 0
        self.total_reward = 0.0
        self.available_actions = self._get_available_actions()
        
    def _get_available_actions(self) -> List[str]:
        """
        Get all available actions from the current state.
        Overridden in specific implementations.
        """
        # This will be implemented based on the specific QA task
        return []
        
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried at least once."""
        return len(self.children) == len(self.available_actions)
    
    def is_terminal(self) -> bool:
        """Check if the node is a terminal state (e.g., final answer)."""
        # This will be overridden based on the specific implementation
        return False
    
    def get_untried_actions(self) -> List[str]:
        """Get actions that haven't been tried yet."""
        tried_actions = [child.action_taken for child in self.children]
        return [action for action in self.available_actions if action not in tried_actions]
    
    def get_ucb_score(self, exploration_weight: Optional[float] = None) -> float:
        """
        Calculate UCB1 score for node selection.
        
        Args:
            exploration_weight: Weight for the exploration term
            
        Returns:
            UCB1 score
        """
        if exploration_weight is None:
            exploration_weight = self.exploration_weight
            
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visit_count
        exploration = exploration_weight * math.sqrt(
            2 * math.log(self.parent.visit_count) / self.visit_count
        ) if self.parent else 0
        
        return exploitation + exploration
    
    def update(self, reward: float) -> None:
        """
        Update node statistics after simulation.
        
        Args:
            reward: Reward received from simulation
        """
        self.visit_count += 1
        self.total_reward += reward
    
    def best_child(self, exploration_weight: Optional[float] = None) -> 'MCTSNode':
        """
        Select best child node based on UCB scores.
        
        Args:
            exploration_weight: Weight for exploration term
            
        Returns:
            Best child node
        """
        if not self.children:
            raise ValueError("No children to select from")
        
        if exploration_weight is None:
            exploration_weight = self.exploration_weight
            
        scores = [
            child.get_ucb_score(exploration_weight) for child in self.children
        ]
        
        return self.children[np.argmax(scores)]
    
    def add_child(self, state: Dict[str, Any], action: str) -> 'MCTSNode':
        """
        Add a child node.
        
        Args:
            state: State of the new node
            action: Action taken to reach this state
            
        Returns:
            The new child node
        """
        child = MCTSNode(
            state=state,
            parent=self,
            action_taken=action,
            exploration_weight=self.exploration_weight
        )
        self.children.append(child)
        return child 