from abc import ABC, abstractmethod
from typing import Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseNLUModel(ABC):
    """Base class for all NLU models"""
    
    @abstractmethod
    def predict(self, text: str) -> Dict:
        """Predict intent from text"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is loaded and ready"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get model information"""
        pass