import pytesseract
from PIL import Image
import numpy as np
from typing import Union, Dict, Any, List
import cv2
import logging

logger = logging.getLogger(__name__)

class OCRTool:
    """
    OCR tool for extracting text from images using Tesseract.
    """
    
    def __init__(
        self, 
        tesseract_cmd: str = None,
        lang: str = "eng",
        config: str = "--psm 6"
    ):
        """
        Initialize OCR tool.
        
        Args:
            tesseract_cmd: Path to tesseract executable
            lang: Language to use for OCR
            config: Configuration string for tesseract
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.lang = lang
        self.config = config
        
        # Test if tesseract is installed and accessible
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.error(f"Tesseract not accessible: {e}")
            logger.warning("Make sure Tesseract OCR is installed and properly configured.")
    
    def extract_text(
        self, 
        image: Union[str, np.ndarray, Image.Image]
    ) -> str:
        """
        Extract text from an image.
        
        Args:
            image: Path to image, numpy array, or PIL Image
            
        Returns:
            Extracted text
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=self.lang, config=self.config)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def extract_structured_data(
        self, 
        image: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Extract structured data from an image including text, bounding boxes, and confidence.
        
        Args:
            image: Path to image, numpy array, or PIL Image
            
        Returns:
            Dictionary with detected data
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Extract data
            data = pytesseract.image_to_data(
                image, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT
            )
            
            # Filter out low confidence and empty text
            results = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0 and data['text'][i].strip():
                    results.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'bbox': (
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        )
                    })
            
            return {
                'full_text': " ".join([r['text'] for r in results]),
                'segments': results
            }
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            return {'full_text': "", 'segments': []}
    
    def __call__(
        self, 
        image: Union[str, np.ndarray, Image.Image],
        structured: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Process an image with OCR.
        
        Args:
            image: Input image
            structured: Whether to return structured data
            
        Returns:
            Extracted text or structured data
        """
        if structured:
            return self.extract_structured_data(image)
        else:
            return self.extract_text(image) 