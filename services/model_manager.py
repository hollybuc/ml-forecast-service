"""
Simple model manager for ML worker
"""
import os
import pickle
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage saving and loading of trained models"""
    
    def __init__(self):
        self.models_dir = os.getenv("MODELS_PATH", "/shared/models")
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model(self, model, model_name: str, project_id: str, metadata: dict = None) -> str:
        """
        Save a trained model to disk
        
        Args:
            model: The trained model object
            model_name: Name of the model (e.g., 'prophet', 'arima')
            project_id: Project ID
            metadata: Optional metadata dict
            
        Returns:
            str: Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{project_id}_{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        logger.info(f"Saving model to: {filepath}")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': model,
                'model_name': model_name,
                'project_id': project_id,
                'metadata': metadata or {},
                'saved_at': timestamp
            }, f)
        
        logger.info(f"Model saved successfully: {filepath}")
        return filepath
    
    def load_model(self, filepath: str):
        """Load a model from disk"""
        logger.info(f"Loading model from: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info(f"Model loaded successfully: {filepath}")
        return model_data['model']

