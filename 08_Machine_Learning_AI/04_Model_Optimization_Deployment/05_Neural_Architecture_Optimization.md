# Neural Architecture Optimization

## Topics
- Manual architecture design principles
- Neural Architecture Search (NAS)
- Hardware-aware architecture optimization
- Efficient model architectures

### Example: Simple NAS with KerasTuner
```python
import keras_tuner as kt
def build_model(hp):
    model = ...  # define model using hp
    return model
tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=5)
# tuner.search(x_train, y_train, epochs=5)
```
