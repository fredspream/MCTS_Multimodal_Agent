import os
import logging
from typing import Union, Dict, Any, Optional
from PIL import Image
import pytesseract

class OCRTool:
    """
    OCR tool for extracting text from images using Tesseract.
    """
    
    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        lang: str = "eng",
        config: str = "--psm 6",
        **kwargs
    ):
        """
        Initialize the OCR tool.
        
        Args:
            tesseract_cmd: Path to Tesseract executable (if not in PATH)
            lang: Language to use for OCR
            config: Configuration for Tesseract
            **kwargs: Additional arguments
        """
        self.logger = logging.getLogger(__name__)
        self.lang = lang
        self.config = config
        
        # Set Tesseract executable path if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        # Check if Tesseract is installed
        try:
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")
            self.tesseract_available = True
        except Exception as e:
            self.logger.warning(f"Tesseract is not available: {e}")
            self.logger.warning("OCR will not work. Please install Tesseract OCR.")
            self.tesseract_available = False
    
    def __call__(
        self,
        image: Union[str, Image.Image],
        **kwargs
    ) -> str:
        """
        Extract text from an image.
        
        Args:
            image: PIL Image or path to image file
            **kwargs: Additional arguments to pass to pytesseract
            
        Returns:
            Extracted text
        """
        if not self.tesseract_available:
            self.logger.warning("Tesseract is not available. Returning dummy OCR result.")
            return self._dummy_ocr_result(image)
        
        try:
            # Convert path to image if necessary
            if isinstance(image, str):
                if os.path.exists(image):
                    image = Image.open(image)
                else:
                    self.logger.error(f"Image file not found: {image}")
                    return "Error: Image file not found."
            
            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.config,
                **kwargs
            )
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return f"Error extracting text: {str(e)}"
    
    def extract_text(
        self,
        image: Union[str, Image.Image],
        **kwargs
    ) -> str:
        """
        Extract text from an image (alias for __call__).
        
        Args:
            image: PIL Image or path to image file
            **kwargs: Additional arguments to pass to pytesseract
            
        Returns:
            Extracted text
        """
        return self(image, **kwargs)
    
    def _dummy_ocr_result(self, image: Union[str, Image.Image]) -> str:
        """Generate a dummy OCR result for testing."""
        return "This is a simulated text extraction result as Tesseract OCR is not available."
    
    def use(
        self,
        question: str,
        image: Optional[Union[str, Image.Image]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Use the OCR tool in the context of a question.
        
        Args:
            question: Question being asked
            image: Image to analyze
            context: Additional context
            
        Returns:
            Extracted text
        """
        if image is None:
            return "No image provided for OCR."
            
        return self.extract_text(image) 