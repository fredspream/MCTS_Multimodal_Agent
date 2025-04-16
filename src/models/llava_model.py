import os
import logging
import torch
from typing import Optional, Dict, Any, Union
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class LLaVAModel:
    """
    LLaVA model for multimodal understanding.
    This class wraps the HuggingFace implementation of LLaVA.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLaVA model.
        
        Args:
            model_name: Name or path of the model
            device: Device to use (cuda or cpu)
            **kwargs: Additional arguments to pass to the model
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Initializing LLaVA model {model_name} on device {self.device}")
        
        # Load processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                **kwargs
            ).to(self.device)
            
            self.logger.info("LLaVA model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading LLaVA model: {e}")
            # Create dummy processor and model for testing
            self.processor = None
            self.model = None
            self.logger.warning("Using dummy LLaVA model for testing")
    
    def generate_with_image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text based on an image and a prompt.
        
        Args:
            prompt: Text prompt to guide the generation
            image: PIL Image or path to image file
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        if self.model is None or self.processor is None:
            self.logger.warning("Using dummy response because model is not loaded")
            return self._dummy_response(prompt, image)
            
        try:
            # Load image if path is provided
            if isinstance(image, str):
                if os.path.exists(image):
                    image = Image.open(image)
                else:
                    self.logger.error(f"Image file not found: {image}")
                    return "Error: Image file not found."
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    **kwargs
                )
            
            # Decode output
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part (after the prompt)
            if prompt in generated_text:
                response = generated_text.split(prompt, 1)[1].strip()
            else:
                response = generated_text.strip()
                
            return response
        
        except Exception as e:
            self.logger.error(f"Error generating with image: {e}")
            return f"Error: {str(e)}"
    
    def generate_caption(self, image: Union[Image.Image, str]) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            Generated caption
        """
        prompt = "Please describe this image in detail."
        return self.generate_with_image(prompt, image, max_tokens=100)
    
    def visual_question_answering(
        self,
        question: str,
        image: Union[Image.Image, str],
        max_tokens: int = 100
    ) -> str:
        """
        Answer a question about an image.
        
        Args:
            question: Question about the image
            image: PIL Image or path to image file
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Answer to the question
        """
        prompt = f"Please answer the following question about the image: {question}"
        return self.generate_with_image(prompt, image, max_tokens=max_tokens)
    
    def _dummy_response(self, prompt: str, image: Union[Image.Image, str]) -> str:
        """Generate a dummy response for testing."""
        if "describe" in prompt.lower() or "caption" in prompt.lower():
            return "This is an image showing various elements that would be described in more detail if the actual model was loaded."
            
        if "question" in prompt.lower():
            question = prompt.split("question:", 1)[1].strip() if "question:" in prompt else prompt
            return f"I would answer the question about the image, but the model is not loaded for inference."
            
        return "This is a dummy response as the LLaVA model is not loaded." 