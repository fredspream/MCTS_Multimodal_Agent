import os
import re
import logging
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLaMAReAct:
    """
    LLaMA model with ReAct framework for reasoning and tool use.
    This class provides a wrapper for the LLaMA model that implements
    the ReAct framework for reasoning and tool use.
    """
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the LLaMA ReAct model.
        
        Args:
            model_name: Name or path of the model
            device: Device to use (cuda or cpu)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            system_prompt: System prompt to use
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Initializing LLaMA ReAct with model {model_name} on device {self.device}")
        
        # Set system prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.logger.info("LLaMA model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading LLaMA model: {e}")
            # Create dummy tokenizer and model for testing
            self.tokenizer = None
            self.model = None
            self.logger.warning("Using dummy LLaMA model for testing")
    
    def _default_system_prompt(self) -> str:
        """Get the default system prompt for ReAct."""
        return """You are a helpful assistant that can use tools to answer questions. You have access to the following tools:

1. caption: Generate a detailed caption of the image
2. ocr: Extract text from the image using OCR
3. answer: Provide the final answer to the question

For each step, think about what information you need to answer the question correctly. Use the tools to gather that information. Then, provide your final answer.

To use a tool, respond with:
Action: [tool name]
Action Input: [input for the tool]

After you receive the observation from the tool, you can use another tool or provide the final answer.
"""
    
    def generate_response(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the ReAct framework.
        
        Args:
            query: The user's question
            context: Additional context for the query (e.g., image description)
            
        Returns:
            Dictionary with the action, action input, and observation
        """
        if self.model is None or self.tokenizer is None:
            self.logger.warning("Using dummy response because model is not loaded")
            return self._dummy_response(query, context)
            
        # Build the prompt
        prompt = self._build_prompt(query, context)
        
        try:
            # Generate text
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Get the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the response (after the prompt)
            response = generated_text[len(prompt):].strip()
            
            # Parse the action and action input from the response
            action_info = self._parse_action(response)
            
            return action_info
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "action": "error",
                "action_input": f"Error: {str(e)}",
                "observation": ""
            }
    
    def _build_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the prompt for the model."""
        # Start with the system prompt
        prompt = f"{self.system_prompt}\n\n"
        
        # Add the query
        prompt += f"Question: {query}\n\n"
        
        # Add context if available
        if context:
            if "image_description" in context and context["image_description"]:
                prompt += f"Image description: {context['image_description']}\n\n"
                
            if "extracted_text" in context and context["extracted_text"]:
                prompt += f"Text extracted from image: {context['extracted_text']}\n\n"
                
            # Add conversation history
            if "history" in context and context["history"]:
                for item in context["history"]:
                    action = item.get("action", "")
                    action_input = item.get("action_input", "")
                    observation = item.get("observation", "")
                    
                    prompt += f"Action: {action}\n"
                    prompt += f"Action Input: {action_input}\n"
                    prompt += f"Observation: {observation}\n\n"
        
        prompt += "Action: "
        
        return prompt
    
    def _parse_action(self, response: str) -> Dict[str, Any]:
        """Parse the action and action input from the response."""
        # Try to extract action and action input using regex
        action_match = re.search(r"Action:\s*(\w+)", response)
        action_input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", response, re.DOTALL)
        
        action = action_match.group(1).strip() if action_match else "answer"
        action_input = action_input_match.group(1).strip() if action_input_match else response
        
        # If no action input is found, use everything after "Action: action"
        if not action_input_match and action_match:
            action_start = action_match.end()
            action_input_start = response[action_start:].find("\n")
            if action_input_start != -1:
                action_input = response[action_start + action_input_start:].strip()
            else:
                action_input = ""
        
        return {
            "action": action,
            "action_input": action_input,
            "observation": ""
        }
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The prompt text
            max_tokens: Maximum number of tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            self.logger.warning("Using dummy response because model is not loaded")
            return "This is a dummy response as the LLaMA model is not loaded."
            
        try:
            # Set parameters
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            # Generate text
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Get the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the response (after the prompt)
            if prompt in generated_text:
                response = generated_text.split(prompt, 1)[1].strip()
            else:
                response = generated_text.strip()
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"
    
    def _dummy_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a dummy response for testing."""
        # Check context to decide which action to take
        if context and context.get("history"):
            history = context["history"]
            
            # If we've already used the caption tool
            if any(item.get("action") == "caption" for item in history):
                # If we've already used OCR, provide an answer
                if any(item.get("action") == "ocr" for item in history):
                    return {
                        "action": "answer",
                        "action_input": f"Based on the image and text, the answer is related to '{query}'.",
                        "observation": ""
                    }
                # Otherwise, use OCR next
                else:
                    return {
                        "action": "ocr",
                        "action_input": "Extract text from the image",
                        "observation": ""
                    }
            # Start with caption
            else:
                return {
                    "action": "caption",
                    "action_input": "Generate a detailed caption of the image",
                    "observation": ""
                }
        
        # If no history, start with caption
        return {
            "action": "caption",
            "action_input": "Generate a detailed caption of the image",
            "observation": ""
        } 