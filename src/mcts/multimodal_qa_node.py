from typing import List, Dict, Any, Optional, Tuple
from .node import MCTSNode

class MultimodalQANode(MCTSNode):
    """
    Specialized MCTS node for multimodal question answering.
    This extends the base MCTSNode with multimodal-specific functionality.
    """
    
    def __init__(
        self,
        state: Dict[str, Any],
        parent: Optional['MCTSNode'] = None,
        action_taken: Optional[str] = None,
        exploration_weight: float = 1.0,
        available_tools: Optional[List[str]] = None
    ):
        """
        Initialize a MultimodalQANode.
        
        Args:
            state: Dictionary containing the current state information including:
                - question: The question text
                - image: Optional image data
                - context: Optional context information
                - history: List of previous actions and outputs
                - current_answer: Current partial answer if any
            parent: Parent node
            action_taken: Action taken to reach this node from parent
            exploration_weight: Weight for UCB1 exploration term
            available_tools: List of available tools (e.g., "caption", "ocr", "answer")
        """
        self.available_tools = available_tools or ["caption", "ocr", "answer"]
        super().__init__(state, parent, action_taken, exploration_weight)
    
    def _get_available_actions(self) -> List[str]:
        """
        Get all available actions from the current state.
        For multimodal QA, this includes tool usage and answer generation.
        
        Returns:
            List of available actions
        """
        # If we've already provided a final answer, no more actions
        if self.state.get("final_answer") is not None:
            return []
        
        # Otherwise return all available tools
        return self.available_tools
    
    def is_terminal(self) -> bool:
        """
        Check if this is a terminal node (i.e., final answer has been given).
        
        Returns:
            True if terminal, False otherwise
        """
        return self.state.get("final_answer") is not None
    
    def get_state_representation(self) -> Dict[str, Any]:
        """
        Get a clean representation of the state for the LLM.
        
        Returns:
            State representation
        """
        return {
            "question": self.state.get("question", ""),
            "image_description": self.state.get("image_description", ""),
            "extracted_text": self.state.get("extracted_text", ""),
            "history": self.state.get("history", []),
            "current_answer": self.state.get("current_answer", "")
        } 