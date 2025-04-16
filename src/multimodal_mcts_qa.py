import os
import logging
import random
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import json
from PIL import Image

from src.models import LLaVAModel, LLaMAReAct
from src.tools import OCRTool
from src.mcts import MCTS, MCTSNode

class MultimodalMCTSQA:
    """
    Multimodal Monte Carlo Tree Search for Question Answering.
    This system uses MCTS to explore different reasoning paths and
    tool combinations to answer questions with image and text inputs.
    """
    
    def __init__(
        self,
        llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
        llama_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cpu",
        ocr_config: Optional[Dict[str, Any]] = None,
        mcts_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Multimodal MCTS QA system.
        
        Args:
            llava_model_name: Name of the LLaVA model
            llama_model_name: Name of the LLaMA model
            device: Device to run models on
            ocr_config: OCR configuration
            mcts_config: MCTS configuration
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        
        # Initialize models and tools
        self.logger.info(f"Initializing Multimodal MCTS QA with device: {device}")
        self.logger.info(f"LLaVA model: {llava_model_name}")
        self.logger.info(f"LLaMA model: {llama_model_name}")
        
        # Initialize LLaVA model for image understanding
        self.llava_model = LLaVAModel(
            model_name=llava_model_name,
            device=device
        )
        
        # Initialize LLaMA model for reasoning
        self.llama_model = LLaMAReAct(
            model_name=llama_model_name,
            device=device
        )
        
        # Initialize OCR tool
        self.ocr_tool = OCRTool(**(ocr_config or {}))
        
        # Store current state for MCTS
        self.current_state = None
        
        # Initialize MCTS configuration
        self.mcts_config = mcts_config or {
            "time_limit": 30.0,
            "max_iterations": 100,
            "exploration_weight": 1.0
        }
        
        # Initialize MCTS
        self.mcts = MCTS(
            state_generator=self._state_generator,
            simulator=self._simulator,
            time_limit=self.mcts_config["time_limit"],
            max_iterations=self.mcts_config["max_iterations"],
            exploration_weight=self.mcts_config["exploration_weight"]
        )
        
        # Define available tools
        self.available_tools = {
            "caption": self._execute_caption_tool,
            "ocr": self._execute_ocr_tool,
            "answer": self._execute_answer_tool
        }
    
    def _execute_caption_tool(self, action_input: str) -> str:
        """
        Execute the caption tool to generate a description of the image.
        
        Args:
            action_input: Input for the tool
            
        Returns:
            Generated caption
        """
        if not self.current_state or "image" not in self.current_state:
            return "No image available for captioning."
            
        try:
            image = self.current_state["image"]
            
            # If image is a path, load it
            if isinstance(image, str) and os.path.exists(image):
                image = Image.open(image)
                
            # Generate caption
            caption = self.llava_model.generate_caption(image)
            
            # Update state
            self.current_state["image_description"] = caption
            
            return caption
            
        except Exception as e:
            self.logger.error(f"Error executing caption tool: {e}")
            return f"Error generating caption: {str(e)}"
    
    def _execute_ocr_tool(self, action_input: str) -> str:
        """
        Execute the OCR tool to extract text from the image.
        
        Args:
            action_input: Input for the tool
            
        Returns:
            Extracted text
        """
        if not self.current_state or "image" not in self.current_state:
            return "No image available for OCR."
            
        try:
            image = self.current_state["image"]
            
            # If image is a path, load it
            if isinstance(image, str) and os.path.exists(image):
                image = Image.open(image)
                
            # Extract text
            extracted_text = self.ocr_tool(image)
            
            # Update state
            self.current_state["extracted_text"] = extracted_text
            
            return extracted_text
            
        except Exception as e:
            self.logger.error(f"Error executing OCR tool: {e}")
            return f"Error extracting text: {str(e)}"
    
    def _execute_answer_tool(self, action_input: str) -> str:
        """
        Execute the answer tool to provide the final answer.
        
        Args:
            action_input: The final answer
            
        Returns:
            Confirmation of the answer
        """
        # Update state
        self.current_state["final_answer"] = action_input
        
        return f"Final answer provided: {action_input}"
    
    def _state_generator(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """
        Generate a new state given the current state and an action.
        This is used by MCTS to simulate state transitions.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            New state
        """
        # Copy the state to avoid modifying the original
        new_state = state.copy()
        
        # Set current state for tool execution
        self.current_state = new_state
        
        # Initialize history if not present
        if "history" not in new_state:
            new_state["history"] = []
            
        # Execute action if it's a valid tool
        if action in self.available_tools:
            # Get appropriate input for the tool
            if action == "caption":
                action_input = "Generate a detailed caption for the image"
            elif action == "ocr":
                action_input = "Extract text from the image"
            elif action == "answer":
                # For answer tool, generate an answer based on current state
                prompt = self._build_answer_prompt(new_state)
                action_input = self.llama_model.generate(prompt, max_tokens=100)
            else:
                action_input = ""
                
            # Execute the tool
            result = self.available_tools[action](action_input)
            
            # Update history
            new_state["history"].append({
                "action": action,
                "action_input": action_input,
                "observation": result
            })
            
        return new_state
    
    def _simulator(self, state: Dict[str, Any]) -> float:
        """
        Simulate the reward for a state.
        This is used by MCTS to evaluate states.
        
        Args:
            state: Current state
            
        Returns:
            Reward value
        """
        # If we have a final answer, evaluate it
        if "final_answer" in state and state["final_answer"]:
            # Calculate reward based on context and answer quality
            reward = self._calculate_reward(state)
            return reward
            
        # If no final answer, simulate forward to get one
        sim_state = state.copy()
        
        # Keep track of how many steps we've taken
        steps_taken = len(sim_state.get("history", []))
        
        # If we've already taken many steps, encourage ending
        if steps_taken >= 3:
            # Generate answer if we have useful information
            if ("image_description" in sim_state or "extracted_text" in sim_state):
                # Generate an answer
                prompt = self._build_answer_prompt(sim_state)
                final_answer = self.llama_model.generate(prompt, max_tokens=100)
                
                # Update state
                sim_state["final_answer"] = final_answer
                
                # Calculate reward
                reward = self._calculate_reward(sim_state)
                
                # Penalize slightly for not ending earlier
                reward *= 0.9
                
                return reward
                
        # Otherwise, take another action based on LLaMA's recommendation
        response = self.llama_model.generate_response(
            sim_state.get("question", ""),
            {
                "image_description": sim_state.get("image_description", ""),
                "extracted_text": sim_state.get("extracted_text", ""),
                "history": sim_state.get("history", [])
            }
        )
        
        action = response.get("action", "")
        action_input = response.get("action_input", "")
        
        # If the model chose to answer, evaluate it
        if action == "answer":
            sim_state["final_answer"] = action_input
            return self._calculate_reward(sim_state)
            
        # If it chose another tool, simulate executing it
        if action in self.available_tools:
            # Temporarily set current state for tool execution
            prev_state = self.current_state
            self.current_state = sim_state
            
            # Execute the tool
            result = self.available_tools[action](action_input)
            
            # Restore current state
            self.current_state = prev_state
            
            # Update history
            if "history" not in sim_state:
                sim_state["history"] = []
                
            sim_state["history"].append({
                "action": action,
                "action_input": action_input,
                "observation": result
            })
            
            # Recursively simulate (limited depth)
            if len(sim_state.get("history", [])) < 5:
                return 0.8 * self._simulator(sim_state)
            else:
                # If we've gone too deep, return a modest reward
                return 0.2
                
        # If we get here, the action wasn't valid
        return 0.1
    
    def _calculate_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate the reward for a state based on the quality of the answer.
        
        Args:
            state: Current state
            
        Returns:
            Reward value
        """
        question = state.get("question", "")
        final_answer = state.get("final_answer", "")
        
        # Base reward
        reward = 0.5
        
        # Reward for having used tools
        if "history" in state:
            tools_used = set(item.get("action", "") for item in state.get("history", []))
            
            # Reward for using both caption and OCR when appropriate
            if "caption" in tools_used:
                reward += 0.15
                
            if "ocr" in tools_used:
                reward += 0.15
                
            # Penalize for not using any tools
            if len(tools_used) <= 1 and "answer" in tools_used:
                reward -= 0.2
                
        # Reward for answer quality
        if final_answer:
            # Penalize very short answers
            if len(final_answer.split()) < 3:
                reward -= 0.1
                
            # Reward for answers that look complete (not just numbers or single words)
            if len(final_answer.split()) > 5:
                reward += 0.1
                
            # If ground truth is available, check accuracy
            if "answer" in state and state["answer"]:
                ground_truth = state["answer"]
                
                # Crude string matching for testing
                if ground_truth.lower() in final_answer.lower():
                    reward += 0.3
                    
                if final_answer.lower() in ground_truth.lower():
                    reward += 0.2
                    
                if final_answer.lower() == ground_truth.lower():
                    reward += 0.5
                    
        return min(1.0, max(0.0, reward))
    
    def _build_answer_prompt(self, state: Dict[str, Any]) -> str:
        """
        Build a prompt for generating the final answer.
        
        Args:
            state: Current state
            
        Returns:
            Prompt for the model
        """
        prompt = "Based on the following information, please answer the question concisely and accurately.\n\n"
        
        if "question" in state:
            prompt += f"Question: {state['question']}\n\n"
            
        if "image_description" in state:
            prompt += f"Image description: {state['image_description']}\n\n"
            
        if "extracted_text" in state:
            prompt += f"Text extracted from image: {state['extracted_text']}\n\n"
            
        prompt += "Answer: "
        
        return prompt
    
    def answer_question(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer a question using MCTS to explore tool combinations.
        
        Args:
            example: Example dictionary with question and metadata
            
        Returns:
            Dictionary with answer, reasoning path, and metadata
        """
        question = example.get("question", "")
        self.logger.info(f"Answering question: {question}")
        
        # Set current state for MCTS
        self.current_state = example.copy()
        
        # If we have a list of choices, format them
        if "choices" in example and example["choices"]:
            choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(example["choices"])])
            self.current_state["choices_text"] = choices_text
            
        # Run MCTS search
        best_action, final_state = self.mcts.search(self.current_state)
        
        # Extract the final answer and reasoning path
        final_answer = final_state.get("final_answer", "")
        reasoning_path = final_state.get("history", [])
        
        # If we don't have a final answer, generate one based on collected information
        if not final_answer and reasoning_path:
            prompt = self._build_answer_prompt(final_state)
            final_answer = self.llama_model.generate(prompt, max_tokens=200)
            final_state["final_answer"] = final_answer
            
            # Add the final answer to the reasoning path
            reasoning_path.append({
                "action": "answer",
                "action_input": final_answer,
                "observation": "Final answer provided."
            })
            
        # Calculate confidence
        confidence = self._calculate_reward(final_state)
        
        # Extract tools used
        tools_used = [step["action"] for step in reasoning_path]
        
        # Return result
        return {
            "question": question,
            "final_answer": final_answer,
            "confidence": confidence,
            "reasoning_path": reasoning_path,
            "tools_used": tools_used,
            "image_description": final_state.get("image_description", ""),
            "extracted_text": final_state.get("extracted_text", "")
        }
    
    def evaluate_on_dataset(
        self,
        dataset,
        num_examples: int = 10,
        output_dir: str = "results"
    ) -> Dict[str, Any]:
        """
        Evaluate the system on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            num_examples: Number of examples to evaluate
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating on {num_examples} examples")
        
        # Limit number of examples
        num_examples = min(num_examples, len(dataset))
        
        # Process each example
        correct = 0
        results = []
        
        for i in range(num_examples):
            # Get example
            example = dataset.get_example_for_mcts(i)
            
            # Answer question
            result = self.answer_question(example)
            
            # Check if answer is correct
            ground_truth = example.get("answer", "")
            is_correct = self._check_answer(result["final_answer"], ground_truth)
            
            if is_correct:
                correct += 1
            
            # Add to results
            results.append({
                "id": example.get("id", ""),
                "question": example.get("question", ""),
                "ground_truth": ground_truth,
                "predicted": result["final_answer"],
                "correct": is_correct,
                "confidence": result["confidence"],
                "reasoning_path": result["reasoning_path"],
                "tools_used": result["tools_used"]
            })
        
        # Calculate metrics
        accuracy = correct / num_examples if num_examples > 0 else 0
        
        # Save detailed results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Return summary
        return {
            "accuracy": accuracy,
            "correct": correct,
            "num_examples": num_examples
        }
    
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if the predicted answer is correct.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            True if correct, False otherwise
        """
        # For multiple-choice questions, extract the option letter/number
        if len(ground_truth) == 1 and ground_truth.upper() in "ABCD1234":
            # Check if the prediction contains the correct option
            predicted_lower = predicted.lower()
            
            # Extract the option from the beginning of the answer
            if ground_truth.upper() in "ABCD":
                option_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                option_idx = option_map.get(ground_truth.upper(), -1)
                option_keywords = [
                    f"option {ground_truth.upper()}",
                    f"answer {ground_truth.upper()}",
                    f"choice {ground_truth.upper()}"
                ]
            else:
                option_idx = int(ground_truth) - 1
                option_keywords = [
                    f"option {ground_truth}",
                    f"answer {ground_truth}",
                    f"choice {ground_truth}"
                ]
                
            # Check for option keywords
            if any(keyword in predicted_lower for keyword in option_keywords):
                return True
                
            # Check for numeric/letter alone at the beginning
            if predicted.strip().upper().startswith(ground_truth.upper()):
                return True
                
            return False
                
        # For free-form answers, do exact matching
        return predicted.lower() == ground_truth.lower() 