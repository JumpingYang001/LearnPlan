# Project: Multi-Framework Integration

*Duration: 3-4 weeks*  
*Difficulty: Advanced*  
*Prerequisites: TensorFlow/PyTorch basics, ONNX knowledge, Python proficiency*

## Objective

Create a comprehensive system that seamlessly integrates machine learning models from different frameworks (TensorFlow, PyTorch, Scikit-learn, XGBoost) into a unified inference pipeline. This project demonstrates real-world production deployment challenges and solutions for heterogeneous ML systems.

### Project Goals
- **Integration**: Combine models from 4+ different ML frameworks
- **Standardization**: Implement unified preprocessing and postprocessing pipelines
- **Optimization**: Optimize for production deployment with performance benchmarking
- **Scalability**: Design for horizontal scaling and containerized deployment
- **Monitoring**: Add comprehensive logging, metrics, and error handling

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   API Gateway                               │
│              (FastAPI/Flask)                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               Model Router                                  │
│           (Route by model type)                            │
└─────┬───────┬───────┬───────┬───────┬────────────────────────┘
      │       │       │       │       │
┌─────▼─┐ ┌───▼──┐ ┌──▼───┐ ┌─▼────┐ ┌▼──────┐
│TensorF│ │PyTorch│ │Scikit│ │XGBoost│ │ ONNX  │
│ low   │ │       │ │-learn│ │       │ │Runtime│
└───────┘ └──────┘ └──────┘ └──────┘ └───────┘
```

## Key Features

### 1. Multi-Framework Model Integration

**Supported Frameworks:**
- **TensorFlow/Keras**: Deep learning models, complex neural networks
- **PyTorch**: Research models, dynamic computation graphs  
- **Scikit-learn**: Traditional ML algorithms, ensemble methods
- **XGBoost**: Gradient boosting, tabular data excellence
- **ONNX Runtime**: Cross-platform optimized inference

### 2. Common Preprocessing and Postprocessing

**Unified Data Pipeline:**
- Input validation and sanitization
- Feature engineering and transformation
- Model-specific preprocessing adaptation
- Output standardization and formatting
- Error handling and fallback mechanisms

### 3. Production Optimization

**Performance Features:**
- Model caching and warm-up strategies
- Batch processing capabilities
- Asynchronous inference pipelines
- Resource monitoring and auto-scaling
- A/B testing framework for model comparison

## Implementation Guide

### Phase 1: Core Infrastructure Setup

#### 1.1 Project Structure
```
multi_framework_ml/
├── src/
│   ├── models/
│   │   ├── tensorflow_models.py
│   │   ├── pytorch_models.py
│   │   ├── sklearn_models.py
│   │   ├── xgboost_models.py
│   │   └── onnx_models.py
│   ├── preprocessing/
│   │   ├── data_validator.py
│   │   ├── feature_transformer.py
│   │   └── preprocessor_factory.py
│   ├── postprocessing/
│   │   ├── output_formatter.py
│   │   └── result_aggregator.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── middleware.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── metrics.py
│   └── tests/
├── models/
│   ├── tensorflow/
│   ├── pytorch/
│   ├── sklearn/
│   ├── xgboost/
│   └── onnx/
├── data/
├── configs/
├── docker/
└── requirements.txt
```

#### 1.2 Environment Setup
```python
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
tensorflow==2.15.0
torch==2.1.0
torchvision==0.16.0
scikit-learn==1.3.2
xgboost==2.0.2
onnxruntime==1.16.3
numpy==1.24.4
pandas==2.1.4
pydantic==2.5.0
prometheus-client==0.19.0
redis==5.0.1
docker==6.1.3
pytest==7.4.3
```

### Phase 2: Model Abstractions

#### 2.1 Base Model Interface
```python
# src/models/base_model.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class ModelMetadata:
    name: str
    framework: str
    version: str
    input_shape: tuple
    output_shape: tuple
    model_type: str  # 'classification', 'regression', 'detection', etc.
    preprocessing_required: List[str]
    postprocessing_required: List[str]

@dataclass
class InferenceResult:
    predictions: Union[np.ndarray, List]
    probabilities: Union[np.ndarray, None] = None
    confidence_scores: Union[np.ndarray, None] = None
    inference_time: float = 0.0
    model_name: str = ""
    metadata: Dict[str, Any] = None

class BaseMLModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, model_path: str, config: Dict[str, Any] = None):
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.metadata = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model from disk."""
        pass
    
    @abstractmethod
    def predict(self, inputs: Union[np.ndarray, List]) -> InferenceResult:
        """Make predictions on input data."""
        pass
    
    @abstractmethod
    def predict_batch(self, inputs: List[Union[np.ndarray, List]]) -> List[InferenceResult]:
        """Make batch predictions."""
        pass
    
    def warm_up(self, sample_input: Union[np.ndarray, List] = None) -> None:
        """Warm up the model with dummy inference."""
        if sample_input is None:
            sample_input = self._generate_dummy_input()
        
        start_time = time.time()
        self.predict(sample_input)
        warmup_time = time.time() - start_time
        print(f"Model {self.metadata.name} warmed up in {warmup_time:.3f}s")
    
    def _generate_dummy_input(self) -> np.ndarray:
        """Generate dummy input for warm-up."""
        if self.metadata and self.metadata.input_shape:
            return np.random.randn(*self.metadata.input_shape)
        return np.random.randn(1, 224, 224, 3)  # Default image input
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.metadata.name if self.metadata else "Unknown",
            "framework": self.metadata.framework if self.metadata else "Unknown",
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "config": self.config
        }
```

#### 2.2 TensorFlow Model Implementation
```python
# src/models/tensorflow_models.py
import tensorflow as tf
import numpy as np
from typing import Union, List, Dict, Any
from .base_model import BaseMLModel, ModelMetadata, InferenceResult
import time

class TensorFlowModel(BaseMLModel):
    """TensorFlow/Keras model wrapper."""
    
    def __init__(self, model_path: str, config: Dict[str, Any] = None):
        super().__init__(model_path, config)
        self.session = None
        
    def load_model(self) -> None:
        """Load TensorFlow model."""
        try:
            if self.model_path.endswith('.h5') or self.model_path.endswith('.keras'):
                self.model = tf.keras.models.load_model(self.model_path)
            elif self.model_path.endswith('.pb'):
                self.model = tf.saved_model.load(self.model_path)
            else:
                # Try SavedModel format
                self.model = tf.saved_model.load(self.model_path)
            
            # Extract metadata
            self._extract_metadata()
            self.is_loaded = True
            print(f"TensorFlow model loaded: {self.metadata.name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorFlow model: {str(e)}")
    
    def _extract_metadata(self) -> None:
        """Extract model metadata."""
        if hasattr(self.model, 'input_shape'):
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape
        else:
            # For SavedModel
            input_shape = None
            output_shape = None
            
        self.metadata = ModelMetadata(
            name=self.config.get('name', 'tensorflow_model'),
            framework='tensorflow',
            version=tf.__version__,
            input_shape=input_shape,
            output_shape=output_shape,
            model_type=self.config.get('model_type', 'classification'),
            preprocessing_required=['normalize', 'resize'],
            postprocessing_required=['softmax', 'argmax']
        )
    
    def predict(self, inputs: Union[np.ndarray, List]) -> InferenceResult:
        """Make single prediction."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Ensure input is numpy array
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        
        # Add batch dimension if needed
        if len(inputs.shape) == len(self.metadata.input_shape) - 1:
            inputs = np.expand_dims(inputs, axis=0)
        
        try:
            if hasattr(self.model, 'predict'):
                # Keras model
                predictions = self.model.predict(inputs, verbose=0)
            else:
                # SavedModel
                predictions = self.model(inputs)
                if isinstance(predictions, dict):
                    predictions = list(predictions.values())[0]
                predictions = predictions.numpy()
            
            inference_time = time.time() - start_time
            
            # Calculate probabilities and confidence
            probabilities = None
            confidence_scores = None
            
            if self.metadata.model_type == 'classification':
                if predictions.shape[-1] > 1:  # Multi-class
                    probabilities = tf.nn.softmax(predictions).numpy()
                    confidence_scores = np.max(probabilities, axis=-1)
                else:  # Binary
                    probabilities = tf.nn.sigmoid(predictions).numpy()
                    confidence_scores = np.abs(probabilities - 0.5) * 2
            
            return InferenceResult(
                predictions=predictions,
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                inference_time=inference_time,
                model_name=self.metadata.name,
                metadata={'framework': 'tensorflow'}
            )
            
        except Exception as e:
            raise RuntimeError(f"TensorFlow prediction failed: {str(e)}")
    
    def predict_batch(self, inputs: List[Union[np.ndarray, List]]) -> List[InferenceResult]:
        """Make batch predictions."""
        # Convert to batch format
        batch_inputs = np.array(inputs)
        
        start_time = time.time()
        
        if hasattr(self.model, 'predict'):
            batch_predictions = self.model.predict(batch_inputs, verbose=0)
        else:
            batch_predictions = self.model(batch_inputs).numpy()
        
        total_time = time.time() - start_time
        per_sample_time = total_time / len(inputs)
        
        results = []
        for i, pred in enumerate(batch_predictions):
            # Process individual predictions
            result = InferenceResult(
                predictions=np.expand_dims(pred, axis=0),
                inference_time=per_sample_time,
                model_name=self.metadata.name,
                metadata={'framework': 'tensorflow', 'batch_index': i}
            )
            results.append(result)
        
        return results

# Example usage and factory
class TensorFlowModelFactory:
    @staticmethod
    def create_image_classifier(model_path: str, num_classes: int) -> TensorFlowModel:
        config = {
            'name': 'tf_image_classifier',
            'model_type': 'classification',
            'num_classes': num_classes
        }
        return TensorFlowModel(model_path, config)
    
    @staticmethod
    def create_object_detector(model_path: str) -> TensorFlowModel:
        config = {
            'name': 'tf_object_detector',
            'model_type': 'detection'
        }
        return TensorFlowModel(model_path, config)
```

#### 2.3 PyTorch Model Implementation
```python
# src/models/pytorch_models.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Any
from .base_model import BaseMLModel, ModelMetadata, InferenceResult
import time

class PyTorchModel(BaseMLModel):
    """PyTorch model wrapper."""
    
    def __init__(self, model_path: str, config: Dict[str, Any] = None):
        super().__init__(model_path, config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> None:
        """Load PyTorch model."""
        try:
            if self.model_path.endswith('.pth') or self.model_path.endswith('.pt'):
                # Load state dict
                if 'model_class' in self.config:
                    model_class = self.config['model_class']
                    self.model = model_class()
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                else:
                    # Load entire model
                    self.model = torch.load(self.model_path, map_location=self.device)
            elif self.model_path.endswith('.onnx'):
                # Load ONNX model through PyTorch
                import torch.onnx
                self.model = torch.jit.load(self.model_path, map_location=self.device)
            else:
                # Try torchscript
                self.model = torch.jit.load(self.model_path, map_location=self.device)
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self._extract_metadata()
            self.is_loaded = True
            print(f"PyTorch model loaded on {self.device}: {self.metadata.name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {str(e)}")
    
    def _extract_metadata(self) -> None:
        """Extract model metadata."""
        # Try to infer input/output shapes
        input_shape = self.config.get('input_shape', (1, 3, 224, 224))
        output_shape = self.config.get('output_shape', None)
        
        self.metadata = ModelMetadata(
            name=self.config.get('name', 'pytorch_model'),
            framework='pytorch',
            version=torch.__version__,
            input_shape=input_shape,
            output_shape=output_shape,
            model_type=self.config.get('model_type', 'classification'),
            preprocessing_required=['normalize', 'resize', 'to_tensor'],
            postprocessing_required=['softmax', 'argmax']
        )
    
    def predict(self, inputs: Union[np.ndarray, List, torch.Tensor]) -> InferenceResult:
        """Make single prediction."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Convert to PyTorch tensor
        if isinstance(inputs, (list, np.ndarray)):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Add batch dimension if needed
        if len(inputs.shape) == len(self.metadata.input_shape) - 1:
            inputs = inputs.unsqueeze(0)
        
        try:
            with torch.no_grad():
                predictions = self.model(inputs)
                
                # Handle different output types
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                
                # Move back to CPU for consistent interface
                predictions = predictions.cpu().numpy()
            
            inference_time = time.time() - start_time
            
            # Calculate probabilities and confidence
            probabilities = None
            confidence_scores = None
            
            if self.metadata.model_type == 'classification':
                if predictions.shape[-1] > 1:  # Multi-class
                    probabilities = F.softmax(torch.tensor(predictions), dim=-1).numpy()
                    confidence_scores = np.max(probabilities, axis=-1)
                else:  # Binary
                    probabilities = torch.sigmoid(torch.tensor(predictions)).numpy()
                    confidence_scores = np.abs(probabilities - 0.5) * 2
            
            return InferenceResult(
                predictions=predictions,
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                inference_time=inference_time,
                model_name=self.metadata.name,
                metadata={'framework': 'pytorch', 'device': str(self.device)}
            )
            
        except Exception as e:
            raise RuntimeError(f"PyTorch prediction failed: {str(e)}")
    
    def predict_batch(self, inputs: List[Union[np.ndarray, List]]) -> List[InferenceResult]:
        """Make batch predictions with optimized GPU usage."""
        # Convert to batch tensor
        if isinstance(inputs[0], (list, np.ndarray)):
            batch_inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        else:
            batch_inputs = torch.stack(inputs)
        
        batch_inputs = batch_inputs.to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            batch_predictions = self.model(batch_inputs)
            if isinstance(batch_predictions, tuple):
                batch_predictions = batch_predictions[0]
            batch_predictions = batch_predictions.cpu().numpy()
        
        total_time = time.time() - start_time
        per_sample_time = total_time / len(inputs)
        
        results = []
        for i, pred in enumerate(batch_predictions):
            result = InferenceResult(
                predictions=np.expand_dims(pred, axis=0),
                inference_time=per_sample_time,
                model_name=self.metadata.name,
                metadata={'framework': 'pytorch', 'device': str(self.device), 'batch_index': i}
            )
            results.append(result)
        
        return results
```

### Phase 3: ONNX Runtime Integration

#### 3.1 ONNX Model Implementation
```python
# src/models/onnx_models.py
import onnxruntime as ort
import numpy as np
from typing import Union, List, Dict, Any
from .base_model import BaseMLModel, ModelMetadata, InferenceResult
import time

class ONNXModel(BaseMLModel):
    """ONNX Runtime model wrapper for cross-framework compatibility."""
    
    def __init__(self, model_path: str, config: Dict[str, Any] = None):
        super().__init__(model_path, config)
        self.session = None
        self.input_names = None
        self.output_names = None
        
    def load_model(self) -> None:
        """Load ONNX model."""
        try:
            # Configure ONNX Runtime providers
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Create inference session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path, 
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output information
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self._extract_metadata()
            self.is_loaded = True
            print(f"ONNX model loaded: {self.metadata.name}")
            print(f"Available providers: {self.session.get_providers()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")
    
    def _extract_metadata(self) -> None:
        """Extract model metadata from ONNX model."""
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        
        # Extract shapes (handle dynamic dimensions)
        input_shape = []
        for dim in input_info.shape:
            if isinstance(dim, str) or dim is None:
                input_shape.append(-1)  # Dynamic dimension
            else:
                input_shape.append(dim)
        
        output_shape = []
        for dim in output_info.shape:
            if isinstance(dim, str) or dim is None:
                output_shape.append(-1)
            else:
                output_shape.append(dim)
        
        self.metadata = ModelMetadata(
            name=self.config.get('name', 'onnx_model'),
            framework='onnx',
            version=ort.__version__,
            input_shape=tuple(input_shape),
            output_shape=tuple(output_shape),
            model_type=self.config.get('model_type', 'classification'),
            preprocessing_required=['normalize', 'resize'],
            postprocessing_required=['softmax', 'argmax']
        )
    
    def predict(self, inputs: Union[np.ndarray, List]) -> InferenceResult:
        """Make single prediction."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Ensure input is numpy array with correct dtype
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        
        inputs = inputs.astype(np.float32)
        
        # Add batch dimension if needed
        if len(inputs.shape) == len(self.metadata.input_shape) - 1:
            inputs = np.expand_dims(inputs, axis=0)
        
        try:
            # Prepare input dictionary
            input_dict = {self.input_names[0]: inputs}
            
            # Run inference
            outputs = self.session.run(self.output_names, input_dict)
            predictions = outputs[0]
            
            inference_time = time.time() - start_time
            
            # Calculate probabilities and confidence
            probabilities = None
            confidence_scores = None
            
            if self.metadata.model_type == 'classification':
                if predictions.shape[-1] > 1:  # Multi-class
                    # Apply softmax
                    exp_pred = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
                    probabilities = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
                    confidence_scores = np.max(probabilities, axis=-1)
                else:  # Binary
                    probabilities = 1 / (1 + np.exp(-predictions))  # Sigmoid
                    confidence_scores = np.abs(probabilities - 0.5) * 2
            
            return InferenceResult(
                predictions=predictions,
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                inference_time=inference_time,
                model_name=self.metadata.name,
                metadata={
                    'framework': 'onnx',
                    'providers': self.session.get_providers(),
                    'input_names': self.input_names,
                    'output_names': self.output_names
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"ONNX prediction failed: {str(e)}")
    
    def predict_batch(self, inputs: List[Union[np.ndarray, List]]) -> List[InferenceResult]:
        """Make batch predictions."""
        # Convert to batch format
        batch_inputs = np.array(inputs, dtype=np.float32)
        
        start_time = time.time()
        
        input_dict = {self.input_names[0]: batch_inputs}
        outputs = self.session.run(self.output_names, input_dict)
        batch_predictions = outputs[0]
        
        total_time = time.time() - start_time
        per_sample_time = total_time / len(inputs)
        
        results = []
        for i, pred in enumerate(batch_predictions):
            result = InferenceResult(
                predictions=np.expand_dims(pred, axis=0),
                inference_time=per_sample_time,
                model_name=self.metadata.name,
                metadata={'framework': 'onnx', 'batch_index': i}
            )
            results.append(result)
        
        return results

# ONNX Conversion Utilities
class ONNXConverter:
    """Utility class for converting models to ONNX format."""
    
    @staticmethod
    def convert_tensorflow_to_onnx(tf_model_path: str, onnx_output_path: str, 
                                 input_shape: tuple = None) -> str:
        """Convert TensorFlow model to ONNX."""
        try:
            import tf2onnx
            import tensorflow as tf
            
            # Load TensorFlow model
            model = tf.keras.models.load_model(tf_model_path)
            
            # Convert to ONNX
            spec = (tf.TensorSpec(input_shape or model.input_shape, tf.float32, name="input"),)
            output_path = onnx_output_path
            
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, 
                                                      output_path=output_path)
            
            print(f"TensorFlow model converted to ONNX: {output_path}")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"TensorFlow to ONNX conversion failed: {str(e)}")
    
    @staticmethod
    def convert_pytorch_to_onnx(torch_model, dummy_input: torch.Tensor, 
                              onnx_output_path: str) -> str:
        """Convert PyTorch model to ONNX."""
        try:
            import torch
            
            # Set model to eval mode
            torch_model.eval()
            
            # Export to ONNX
            torch.onnx.export(
                torch_model,
                dummy_input,
                onnx_output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"PyTorch model converted to ONNX: {onnx_output_path}")
            return onnx_output_path
            
        except Exception as e:
            raise RuntimeError(f"PyTorch to ONNX conversion failed: {str(e)}")
```

### Phase 4: Unified Preprocessing Pipeline

#### 4.1 Data Validation and Sanitization
```python
# src/preprocessing/data_validator.py
from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, validator
import logging

class InputValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

class ImageInputSchema(BaseModel):
    """Schema for image input validation."""
    data: Union[List, np.ndarray]
    format: str = "rgb"  # rgb, bgr, grayscale
    height: Optional[int] = None
    width: Optional[int] = None
    channels: Optional[int] = None
    
    @validator('data')
    def validate_data(cls, v):
        if isinstance(v, list):
            v = np.array(v)
        if not isinstance(v, np.ndarray):
            raise ValueError("Data must be numpy array or list")
        if v.ndim < 2 or v.ndim > 4:
            raise ValueError("Image data must be 2D, 3D, or 4D array")
        return v
    
    @validator('format')
    def validate_format(cls, v):
        if v not in ['rgb', 'bgr', 'grayscale']:
            raise ValueError("Format must be 'rgb', 'bgr', or 'grayscale'")
        return v

class TabularInputSchema(BaseModel):
    """Schema for tabular data validation."""
    data: Union[Dict, List[Dict], pd.DataFrame]
    features: List[str]
    required_features: List[str] = []
    
    @validator('data')
    def validate_data(cls, v):
        if isinstance(v, dict):
            return [v]  # Convert single dict to list
        elif isinstance(v, pd.DataFrame):
            return v.to_dict('records')
        elif isinstance(v, list):
            if not all(isinstance(item, dict) for item in v):
                raise ValueError("List must contain dictionaries")
            return v
        else:
            raise ValueError("Data must be dict, list of dicts, or DataFrame")

class DataValidator:
    """Centralized data validation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_image_input(self, data: Any, **kwargs) -> np.ndarray:
        """Validate image input data."""
        try:
            schema = ImageInputSchema(data=data, **kwargs)
            validated_data = schema.data
            
            # Additional validation
            if validated_data.dtype not in [np.uint8, np.float32, np.float64]:
                self.logger.warning(f"Unexpected dtype: {validated_data.dtype}")
            
            # Check value ranges
            if validated_data.dtype == np.uint8:
                if np.any(validated_data > 255) or np.any(validated_data < 0):
                    raise InputValidationError("uint8 image values must be in [0, 255]")
            elif validated_data.dtype in [np.float32, np.float64]:
                if np.any(validated_data > 1.0) or np.any(validated_data < 0.0):
                    self.logger.warning("Float image values outside [0, 1] range")
            
            return validated_data
            
        except Exception as e:
            raise InputValidationError(f"Image validation failed: {str(e)}")
    
    def validate_tabular_input(self, data: Any, features: List[str], 
                             required_features: List[str] = None) -> List[Dict]:
        """Validate tabular input data."""
        try:
            schema = TabularInputSchema(
                data=data, 
                features=features,
                required_features=required_features or []
            )
            validated_data = schema.data
            
            # Check for required features
            for item in validated_data:
                missing_features = [f for f in schema.required_features if f not in item]
                if missing_features:
                    raise InputValidationError(f"Missing required features: {missing_features}")
                
                # Check for unexpected features
                extra_features = [f for f in item.keys() if f not in features]
                if extra_features:
                    self.logger.warning(f"Unexpected features: {extra_features}")
            
            return validated_data
            
        except Exception as e:
            raise InputValidationError(f"Tabular validation failed: {str(e)}")
    
    def sanitize_input(self, data: Union[np.ndarray, List, Dict]) -> Union[np.ndarray, List, Dict]:
        """Sanitize input data."""
        if isinstance(data, np.ndarray):
            # Remove NaN and infinite values
            if np.any(np.isnan(data)):
                self.logger.warning("NaN values found, replacing with 0")
                data = np.nan_to_num(data, nan=0.0)
            
            if np.any(np.isinf(data)):
                self.logger.warning("Infinite values found, replacing with finite values")
                data = np.nan_to_num(data, posinf=1e6, neginf=-1e6)
        
        elif isinstance(data, list):
            # Handle lists of dictionaries (tabular data)
            for item in data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                            item[key] = 0.0
        
        return data
```

### Phase 5: Complete Production System

#### 5.1 Model Manager
```python
# src/models/model_manager.py
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from pathlib import Path

from .tensorflow_models import TensorFlowModel
from .pytorch_models import PyTorchModel
from .onnx_models import ONNXModel
from .sklearn_models import SklearnModel
from .xgboost_models import XGBoostModel
from ..preprocessing.preprocessor_factory import PreprocessorFactory
from ..postprocessing.output_formatter import OutputFormatter
from ..utils.config import Config
from ..utils.metrics import MetricsCollector

class ModelManager:
    """Centralized model management and routing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.preprocessors: Dict[str, Any] = {}
        self.postprocessors: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.metrics = MetricsCollector()
        
    async def initialize(self) -> None:
        """Initialize model manager and load configured models."""
        try:
            # Load model configurations
            self.model_configs = self.config.get_model_configs()
            
            # Pre-load specified models
            preload_models = self.config.get('preload_models', [])
            for model_name in preload_models:
                await self.load_model(model_name)
            
            self.logger.info(f"Model manager initialized with {len(self.loaded_models)} models")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model manager: {str(e)}")
            raise
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific model."""
        if model_name in self.loaded_models:
            self.logger.info(f"Model {model_name} already loaded")
            return True
        
        if model_name not in self.model_configs:
            self.logger.error(f"Model configuration not found: {model_name}")
            return False
        
        try:
            config = self.model_configs[model_name]
            framework = config.get('framework')
            model_path = config.get('model_path')
            
            # Create model instance based on framework
            if framework == 'tensorflow':
                model = TensorFlowModel(model_path, config)
            elif framework == 'pytorch':
                model = PyTorchModel(model_path, config)
            elif framework == 'onnx':
                model = ONNXModel(model_path, config)
            elif framework == 'sklearn':
                model = SklearnModel(model_path, config)
            elif framework == 'xgboost':
                model = XGBoostModel(model_path, config)
            else:
                raise ValueError(f"Unsupported framework: {framework}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, model.load_model)
            
            # Create preprocessor and postprocessor
            preprocessor = PreprocessorFactory.create_preprocessor(framework, config)
            postprocessor = OutputFormatter(config)
            
            # Store in manager
            with self.lock:
                self.loaded_models[model_name] = model
                self.preprocessors[model_name] = preprocessor
                self.postprocessors[model_name] = postprocessor
            
            # Warm up model
            if config.get('warmup', True):
                await loop.run_in_executor(self.executor, model.warm_up)
            
            self.logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model."""
        try:
            with self.lock:
                if model_name in self.loaded_models:
                    del self.loaded_models[model_name]
                    del self.preprocessors[model_name]
                    del self.postprocessors[model_name]
                    
                    self.logger.info(f"Model {model_name} unloaded")
                    return True
                else:
                    self.logger.warning(f"Model {model_name} not loaded")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to unload model {model_name}: {str(e)}")
            return False
    
    async def predict(self, model_name: str, data: Any, 
                     preprocessing_config: Optional[Dict] = None,
                     postprocessing_config: Optional[Dict] = None) -> Any:
        """Make prediction with specified model."""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        start_time = time.time()
        
        try:
            # Get model and processors
            model = self.loaded_models[model_name]
            preprocessor = self.preprocessors[model_name]
            postprocessor = self.postprocessors[model_name]
            
            # Preprocess data
            if preprocessing_config:
                preprocessor.update_config(preprocessing_config)
            
            processed_data = preprocessor.transform(data)
            
            # Make prediction in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                model.predict, 
                processed_data
            )
            
            # Postprocess result
            if postprocessing_config:
                postprocessor.update_config(postprocessing_config)
            
            formatted_result = postprocessor.format(result)
            
            # Record metrics
            inference_time = time.time() - start_time
            self.metrics.record_prediction(model_name, inference_time, True)
            
            return formatted_result
            
        except Exception as e:
            self.metrics.record_prediction(model_name, time.time() - start_time, False)
            self.logger.error(f"Prediction failed for model {model_name}: {str(e)}")
            raise
    
    async def predict_batch(self, model_name: str, data: List[Any], 
                           batch_size: int = 32,
                           preprocessing_config: Optional[Dict] = None,
                           postprocessing_config: Optional[Dict] = None) -> List[Any]:
        """Make batch predictions."""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        results = []
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.predict(model_name, item, preprocessing_config, postprocessing_config)
                for item in batch_data
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    async def list_models(self) -> Dict[str, Dict]:
        """List all available models."""
        model_info = {}
        
        for name, config in self.model_configs.items():
            is_loaded = name in self.loaded_models
            
            info = {
                'framework': config.get('framework'),
                'model_type': config.get('model_type'),
                'status': 'loaded' if is_loaded else 'available',
                'input_shape': config.get('input_shape'),
                'output_shape': config.get('output_shape'),
                'last_used': config.get('last_used')
            }
            
            if is_loaded:
                model = self.loaded_models[name]
                if hasattr(model, 'metadata'):
                    info.update({
                        'input_shape': model.metadata.input_shape,
                        'output_shape': model.metadata.output_shape
                    })
            
            model_info[name] = info
        
        return model_info
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found")
        
        config = self.model_configs[model_name]
        is_loaded = model_name in self.loaded_models
        
        info = {
            'name': model_name,
            'framework': config.get('framework'),
            'status': 'loaded' if is_loaded else 'available',
            'config': config
        }
        
        if is_loaded:
            model = self.loaded_models[model_name]
            info.update(model.get_model_info())
        
        return info
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Unload all models
            model_names = list(self.loaded_models.keys())
            for name in model_names:
                await self.unload_model(name)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Model manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
```

#### 5.2 Production Deployment Configuration

**Docker Configuration:**
```dockerfile
# docker/Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose for Multi-Service Setup:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./configs:/app/configs:ro
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379
      - PROMETHEUS_ENABLED=true
    depends_on:
      - redis
      - prometheus
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  redis_data:
  grafana_data:
```

### Phase 6: Testing and Benchmarking

#### 6.1 Comprehensive Test Suite
```python
# tests/test_integration.py
import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch

from src.models.model_manager import ModelManager
from src.utils.config import Config
from src.api.main import app
from fastapi.testclient import TestClient

class TestMultiFrameworkIntegration:
    """Integration tests for multi-framework system."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return Config({
            'models': {
                'test_tf_model': {
                    'framework': 'tensorflow',
                    'model_path': 'tests/fixtures/test_model.h5',
                    'model_type': 'classification',
                    'input_shape': (1, 224, 224, 3),
                    'warmup': False
                },
                'test_pytorch_model': {
                    'framework': 'pytorch',
                    'model_path': 'tests/fixtures/test_model.pt',
                    'model_type': 'classification',
                    'input_shape': (1, 3, 224, 224),
                    'warmup': False
                }
            }
        })
    
    @pytest.fixture
    async def model_manager(self, config):
        """Model manager fixture."""
        manager = ModelManager(config)
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_model_loading(self, model_manager):
        """Test model loading for different frameworks."""
        # Test TensorFlow model loading
        success = await model_manager.load_model('test_tf_model')
        assert success
        assert 'test_tf_model' in model_manager.loaded_models
        
        # Test PyTorch model loading
        success = await model_manager.load_model('test_pytorch_model')
        assert success
        assert 'test_pytorch_model' in model_manager.loaded_models
    
    @pytest.mark.asyncio
    async def test_prediction_consistency(self, model_manager):
        """Test prediction consistency across frameworks."""
        # Load models
        await model_manager.load_model('test_tf_model')
        await model_manager.load_model('test_pytorch_model')
        
        # Create test input
        test_input = np.random.randn(224, 224, 3).astype(np.float32)
        
        # Get predictions from both models
        tf_result = await model_manager.predict('test_tf_model', test_input)
        pytorch_result = await model_manager.predict('test_pytorch_model', test_input)
        
        # Verify both predictions are valid
        assert tf_result is not None
        assert pytorch_result is not None
        assert hasattr(tf_result, 'predictions')
        assert hasattr(pytorch_result, 'predictions')
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, model_manager):
        """Test batch processing functionality."""
        await model_manager.load_model('test_tf_model')
        
        # Create batch of test inputs
        batch_size = 5
        batch_data = [np.random.randn(224, 224, 3).astype(np.float32) 
                     for _ in range(batch_size)]
        
        # Process batch
        results = await model_manager.predict_batch('test_tf_model', batch_data)
        
        # Verify results
        assert len(results) == batch_size
        assert all(result is not None for result in results)
    
    def test_api_endpoints(self):
        """Test API endpoints."""
        client = TestClient(app)
        
        # Test health check
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test model listing
        response = client.get("/models")
        assert response.status_code == 200
        
        # Test prediction endpoint
        test_data = {
            "model_name": "test_tf_model",
            "data": np.random.randn(224, 224, 3).tolist()
        }
        response = client.post("/predict", json=test_data)
        # Note: This may fail if model not loaded, which is expected in test environment
    
    @pytest.mark.performance
    def test_performance_benchmarks(self, model_manager):
        """Test performance benchmarks."""
        # This would include throughput, latency, and resource usage tests
        pass

# Performance benchmarking script
class PerformanceBenchmark:
    """Performance benchmarking for multi-framework system."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.results = {}
    
    async def benchmark_single_prediction(self, model_name: str, num_runs: int = 100):
        """Benchmark single prediction performance."""
        test_input = np.random.randn(224, 224, 3).astype(np.float32)
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            await self.model_manager.predict(model_name, test_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        self.results[f"{model_name}_single"] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': 1.0 / np.mean(times)  # predictions per second
        }
    
    async def benchmark_batch_prediction(self, model_name: str, batch_sizes: List[int]):
        """Benchmark batch prediction performance."""
        for batch_size in batch_sizes:
            batch_data = [np.random.randn(224, 224, 3).astype(np.float32) 
                         for _ in range(batch_size)]
            
            start_time = time.time()
            await self.model_manager.predict_batch(model_name, batch_data)
            end_time = time.time()
            
            total_time = end_time - start_time
            throughput = batch_size / total_time
            
            self.results[f"{model_name}_batch_{batch_size}"] = {
                'total_time': total_time,
                'per_sample_time': total_time / batch_size,
                'throughput': throughput
            }
    
    def generate_report(self) -> str:
        """Generate performance report."""
        report = "# Multi-Framework Performance Benchmark Report\n\n"
        
        for test_name, metrics in self.results.items():
            report += f"## {test_name}\n"
            for metric, value in metrics.items():
                report += f"- {metric}: {value:.4f}\n"
            report += "\n"
        
        return report
```

## Learning Objectives and Assessment

### Project Completion Criteria

By the end of this project, you should have:

✅ **Technical Implementation:**
- [ ] Functional multi-framework integration system
- [ ] Unified preprocessing and postprocessing pipelines
- [ ] Production-ready API with proper error handling
- [ ] Docker containerization and deployment configuration
- [ ] Comprehensive test suite with >80% coverage
- [ ] Performance benchmarking and optimization

✅ **Understanding Demonstrated:**
- [ ] Explain trade-offs between different ML frameworks
- [ ] Implement framework-agnostic model interfaces
- [ ] Design scalable inference pipelines
- [ ] Handle production deployment challenges
- [ ] Optimize for performance and resource usage

### Assessment Rubric

| Criteria | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|----------|---------------|----------|------------------|----------------------|
| **Framework Integration** | Seamlessly integrates 4+ frameworks with consistent interfaces | Integrates 3+ frameworks with minor inconsistencies | Basic integration of 2-3 frameworks | Limited framework support |
| **Code Quality** | Clean, well-documented, follows best practices | Good structure with adequate documentation | Functional code with basic documentation | Poor structure, minimal documentation |
| **Performance** | Optimized for production with benchmarking | Good performance with some optimization | Adequate performance | Poor performance |
| **Testing** | Comprehensive test suite with >80% coverage | Good test coverage >60% | Basic testing >40% | Minimal testing |
| **Documentation** | Complete documentation with examples | Good documentation | Basic documentation | Poor documentation |

### Real-World Applications

This project simulates challenges faced in:

🏢 **Enterprise ML Platforms:** Integrating models from different teams using various frameworks  
🚀 **MLOps Pipelines:** Standardizing model deployment across diverse ML stack  
🔄 **Model Migration:** Moving between frameworks while maintaining consistency  
⚡ **Production Optimization:** Balancing accuracy, latency, and resource usage  
📊 **A/B Testing:** Comparing models from different frameworks fairly  

### Next Steps and Extensions

**Advanced Features to Implement:**
- Model versioning and rollback capabilities
- Automated model optimization (quantization, pruning)
- Multi-GPU and distributed inference
- Advanced monitoring and alerting
- CI/CD pipeline integration
- Auto-scaling based on load

**Career Development:**
- Study MLOps best practices and tools
- Learn about model serving platforms (Seldon, Kubeflow)
- Explore cloud ML services (AWS SageMaker, Google AI Platform)
- Understand MLOps monitoring and observability
- Practice with production ML system design

This comprehensive project provides hands-on experience with the complexities of production ML systems while demonstrating mastery of multiple frameworks and deployment strategies!
