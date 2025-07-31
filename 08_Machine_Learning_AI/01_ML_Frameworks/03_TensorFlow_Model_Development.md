# TensorFlow Model Development

*Duration: 2-3 weeks*

## Keras API in TensorFlow

Keras is TensorFlow's high-level API that makes building and training deep learning models intuitive and fast. It provides multiple approaches to model building, from simple sequential models to complex custom architectures.

### Sequential API

The Sequential API is the simplest way to build models for straightforward stack of layers where each layer has exactly one input and one output.

#### Basic Sequential Model
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Simple classification model
def create_sequential_model():
    model = tf.keras.Sequential([
        # Input layer (28x28 flattened images)
        layers.Flatten(input_shape=(28, 28)),
        
        # Hidden layers
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.Dropout(0.2),  # Regularization
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dropout(0.2),
        
        # Output layer (10 classes for digits)
        layers.Dense(10, activation='softmax', name='output')
    ])
    
    return model

# Create and compile model
model = create_sequential_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
print(model.summary())
```

#### Advanced Sequential Model for CNN
```python
def create_cnn_sequential():
    model = tf.keras.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model
```

### Functional API

The Functional API is more flexible and allows you to create models with non-linear topology, shared layers, and multiple inputs/outputs.

#### Basic Functional Model
```python
def create_functional_model():
    # Define inputs
    inputs = tf.keras.Input(shape=(28, 28))
    
    # Define the flow of data
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
```

#### Multi-Input, Multi-Output Model
```python
def create_multi_io_model():
    # Multiple inputs
    text_input = tf.keras.Input(shape=(100,), name='text')
    image_input = tf.keras.Input(shape=(28, 28, 1), name='image')
    
    # Text processing branch
    text_features = layers.Dense(64, activation='relu')(text_input)
    text_features = layers.Dropout(0.5)(text_features)
    
    # Image processing branch
    image_features = layers.Flatten()(image_input)
    image_features = layers.Dense(64, activation='relu')(image_features)
    image_features = layers.Dropout(0.5)(image_features)
    
    # Combine features
    combined = layers.concatenate([text_features, image_features])
    
    # Multiple outputs
    classification_output = layers.Dense(10, activation='softmax', name='classification')(combined)
    regression_output = layers.Dense(1, name='regression')(combined)
    
    model = tf.keras.Model(
        inputs=[text_input, image_input],
        outputs=[classification_output, regression_output]
    )
    
    return model

# Compile with multiple losses
model = create_multi_io_model()
model.compile(
    optimizer='adam',
    loss={
        'classification': 'sparse_categorical_crossentropy',
        'regression': 'mse'
    },
    loss_weights={'classification': 1.0, 'regression': 0.2},
    metrics={
        'classification': ['accuracy'],
        'regression': ['mae']
    }
)
```

### Custom Layers and Models

#### Custom Layer Implementation
```python
class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        # Create trainable weights
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='weights'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super(CustomDenseLayer, self).build(input_shape)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super(CustomDenseLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config

# Usage of custom layer
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    CustomDenseLayer(128, activation='relu'),
    layers.Dropout(0.2),
    CustomDenseLayer(10, activation='softmax')
])
```

#### Custom Model Class
```python
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.add = layers.Add()
        self.relu = layers.ReLU()
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.add([x, inputs])
        return self.relu(x)

class CustomResNet(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(CustomResNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        # Initial layers
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D(3, strides=2, padding='same')
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.res_block3 = ResidualBlock(128)
        self.res_block4 = ResidualBlock(128)
        
        # Final layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        x = self.global_pool(x)
        return self.classifier(x)

# Usage
custom_model = CustomResNet(num_classes=10)
custom_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Training and Evaluation

#### Basic Training Pipeline
```python
def train_model_basic():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Create model
    model = create_sequential_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return model, history

# Plotting training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

### Callbacks and Monitoring

Callbacks provide powerful ways to monitor and control the training process.

#### Essential Callbacks
```python
def create_callbacks(model_name):
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'best_{model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs/{model_name}',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ),
        
        # Custom progress callback
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"Epoch {epoch+1}: Loss={logs['loss']:.4f}, "
                f"Acc={logs['accuracy']:.4f}, "
                f"Val_Loss={logs['val_loss']:.4f}, "
                f"Val_Acc={logs['val_accuracy']:.4f}"
            )
        )
    ]
    
    return callbacks

# Custom callback example
class CustomTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=5):
        super(CustomTrainingCallback, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.best_loss = np.inf
        self.wait = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
            print(f"New best validation loss: {current_loss:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True

# Training with callbacks
def train_with_callbacks():
    model = create_sequential_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = create_callbacks('mnist_model')
    
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=50,  # Will stop early if needed
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history
```

## Custom Training Loops

While Keras' high-level training API is convenient, custom training loops give you full control over the training process. This is essential for advanced techniques like adversarial training, custom loss functions, or complex optimization strategies.

### GradientTape Usage

`tf.GradientTape` is TensorFlow's automatic differentiation engine that records operations for automatic differentiation.

#### Basic Custom Training Loop
```python
import tensorflow as tf
import numpy as np

def custom_training_loop():
    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Convert to tensors
    x_train = tf.cast(x_train, tf.float32)
    y_train = tf.cast(y_train, tf.int64)
    
    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(32)
    
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Define optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    # Training step function
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss(loss)
        train_accuracy(labels, predictions)
    
    # Training loop
    EPOCHS = 10
    for epoch in range(EPOCHS):
        # Reset metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        # Training
        for images, labels in train_dataset:
            train_step(images, labels)
        
        print(f'Epoch {epoch + 1}, '
              f'Loss: {train_loss.result():.4f}, '
              f'Accuracy: {train_accuracy.result():.4f}')

# Advanced custom training with validation
def advanced_custom_training():
    # Data preparation
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train, x_val = x_train / 255.0, x_val / 255.0
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
    
    # Model and optimizer
    model = create_cnn_sequential()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Metrics
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_loss = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
            # Add L2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                               if 'bias' not in v.name]) * 0.001
            total_loss = loss + l2_loss
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(total_loss)
        train_accuracy(labels, predictions)
        
        return total_loss
    
    @tf.function
    def val_step(images, labels):
        predictions = model(images, training=False)
        loss = loss_fn(labels, predictions)
        
        val_loss(loss)
        val_accuracy(labels, predictions)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 5
    wait = 0
    
    for epoch in range(50):
        # Reset metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        
        # Training
        for images, labels in train_ds:
            train_step(images, labels)
        
        # Validation
        for images, labels in val_ds:
            val_step(images, labels)
        
        # Print results
        print(f'Epoch {epoch + 1}:')
        print(f'  Train Loss: {train_loss.result():.4f}, Train Acc: {train_accuracy.result():.4f}')
        print(f'  Val Loss: {val_loss.result():.4f}, Val Acc: {val_accuracy.result():.4f}')
        
        # Early stopping
        if val_loss.result() < best_val_loss:
            best_val_loss = val_loss.result()
            wait = 0
            # Save best weights
            model.save_weights('best_model_weights.h5')
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
```

### Optimizers

Understanding different optimizers and their use cases is crucial for effective model training.

#### Optimizer Comparison
```python
def compare_optimizers():
    """Compare different optimizers on the same model"""
    
    # Prepare data
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train, x_val = x_train / 255.0, x_val / 255.0
    
    optimizers = {
        'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
        'SGD_Momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
        'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
        'AdaGrad': tf.keras.optimizers.Adagrad(learning_rate=0.01),
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name} optimizer...")
        
        # Create fresh model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=10,
            validation_data=(x_val, y_val),
            verbose=0
        )
        
        results[name] = history.history
        print(f"{name} - Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return results

# Custom optimizer configuration
def custom_optimizer_schedule():
    """Example of custom learning rate schedules"""
    
    # Exponential decay
    initial_lr = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    
    # Cosine decay
    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1,
        decay_steps=1000
    )
    
    # Piecewise constant decay
    boundaries = [1000, 2000]
    values = [0.1, 0.01, 0.001]
    piecewise_lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )
    
    # Custom learning rate schedule
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()
            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.warmup_steps = warmup_steps
        
        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    custom_lr = CustomSchedule(128)
    
    # Use with optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    return optimizer
```

### Training Metrics

#### Custom Metrics Implementation
```python
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

class TopKCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, k=5, name='top_k_categorical_accuracy', **kwargs):
        super(TopKCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.k = k
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        
        # Get top k predictions
        top_k_pred = tf.nn.top_k(y_pred, k=self.k).indices
        
        # Check if true label is in top k
        matches = tf.reduce_any(tf.equal(tf.expand_dims(y_true, 1), top_k_pred), axis=1)
        matches = tf.cast(matches, tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            matches = tf.multiply(matches, sample_weight)
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        
        self.count.assign_add(tf.reduce_sum(matches))
    
    def result(self):
        return self.count / self.total
    
    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)

# Usage in custom training
def training_with_custom_metrics():
    model = create_sequential_model()
    
    # Custom training loop with metrics
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Initialize custom metrics
    f1_metric = F1Score()
    top3_accuracy = TopKCategoricalAccuracy(k=3)
    train_loss = tf.keras.metrics.Mean()
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss(loss)
        
        # For F1 score (binary classification adaptation)
        binary_pred = tf.cast(tf.argmax(predictions, axis=1), tf.float32)
        binary_true = tf.cast(y, tf.float32)
        f1_metric(binary_true, binary_pred)
        
        # Top-K accuracy
        top3_accuracy(y, predictions)
        
        return loss
    
    return train_step, [train_loss, f1_metric, top3_accuracy]
```

### Distributed Training

#### Multi-GPU Training Strategy
```python
def distributed_training_setup():
    """Setup for distributed training across multiple GPUs"""
    
    # Create distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    
    # Global batch size
    BATCH_SIZE_PER_REPLICA = 64
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
    with strategy.scope():
        # Create model within strategy scope
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return strategy, model, GLOBAL_BATCH_SIZE

def custom_distributed_training():
    """Custom training loop with distributed strategy"""
    
    strategy, model, global_batch_size = distributed_training_setup()
    
    # Prepare distributed dataset
    def make_datasets():
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
        x_train = x_train / 255.0
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(1000).batch(global_batch_size)
        
        return strategy.experimental_distribute_dataset(train_dataset)
    
    train_dist_dataset = make_datasets()
    
    with strategy.scope():
        # Loss and metrics
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )
        
        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        
        # Optimizer and metrics
        optimizer = tf.keras.optimizers.Adam()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        def train_step(inputs):
            images, labels = inputs
            
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_loss(labels, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_accuracy.update_state(labels, predictions)
            return loss
        
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        
        # Training loop
        for epoch in range(10):
            total_loss = 0.0
            num_batches = 0
            
            for x in train_dist_dataset:
                total_loss += distributed_train_step(x)
                num_batches += 1
            
            train_loss = total_loss / num_batches
            
            print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy.result():.4f}')
            train_accuracy.reset_states()
```

## TensorFlow Datasets

The `tf.data` API is a powerful tool for building efficient, scalable input pipelines. It allows you to work with large datasets that don't fit in memory, apply complex preprocessing, and optimize performance.

### tf.data API Fundamentals

#### Creating Datasets from Different Sources
```python
import tensorflow as tf
import numpy as np

def create_datasets_examples():
    # From numpy arrays
    x = np.random.random((1000, 32))
    y = np.random.randint(0, 10, (1000,))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # From generators
    def data_generator():
        for i in range(100):
            yield i, i**2
    
    generator_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    # From file patterns
    file_dataset = tf.data.Dataset.list_files('/path/to/images/*.jpg')
    
    # From CSV files
    csv_dataset = tf.data.experimental.make_csv_dataset(
        'data.csv',
        batch_size=32,
        label_name='target'
    )
    
    # From TFRecord files
    tfrecord_dataset = tf.data.TFRecordDataset(['file1.tfrecord', 'file2.tfrecord'])
    
    return dataset

# Basic dataset operations
def basic_dataset_operations():
    # Create sample dataset
    dataset = tf.data.Dataset.range(100)
    
    # Basic transformations
    dataset = dataset.map(lambda x: x * 2)  # Apply function to each element
    dataset = dataset.filter(lambda x: x < 50)  # Filter elements
    dataset = dataset.batch(10)  # Group into batches
    dataset = dataset.shuffle(buffer_size=1000)  # Shuffle
    dataset = dataset.repeat(2)  # Repeat dataset twice
    dataset = dataset.take(5)  # Take only first 5 batches
    
    # Iterate through dataset
    for batch in dataset:
        print(batch.numpy())
    
    return dataset
```

### Data Pipelines

#### Image Data Pipeline
```python
def create_image_pipeline(image_dir, batch_size=32, img_height=224, img_width=224):
    """Complete image preprocessing pipeline"""
    
    # Create dataset from directory
    dataset = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # Normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2)
    ])
    
    def preprocess_image(image, label):
        # Normalize
        image = normalization_layer(image)
        # Apply augmentation (only during training)
        image = data_augmentation(image, training=True)
        return image, label
    
    # Apply preprocessing
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Performance optimizations
    dataset = dataset.cache()  # Cache preprocessed data
    dataset = dataset.shuffle(1000)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch
    
    return dataset

# Text data pipeline
def create_text_pipeline(text_data, labels, vocab_size=10000, max_length=100):
    """Text preprocessing pipeline with tokenization"""
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((text_data, labels))
    
    # Create text vectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=max_length,
        output_mode='int'
    )
    
    # Adapt vectorizer to data
    text_only_dataset = dataset.map(lambda x, y: x)
    vectorize_layer.adapt(text_only_dataset)
    
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label
    
    # Apply vectorization
    dataset = dataset.map(vectorize_text)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, vectorize_layer
```

### Preprocessing Operations

#### Advanced Preprocessing Techniques
```python
def advanced_preprocessing():
    """Advanced preprocessing operations for different data types"""
    
    # Image preprocessing
    def preprocess_image_advanced(image_path, label):
        # Read image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        
        # Random crop and resize
        image = tf.image.random_crop(image, size=[224, 224, 3])
        image = tf.image.resize(image, [256, 256])
        
        # Color augmentation
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        
        # Normalize
        image = tf.image.per_image_standardization(image)
        
        return image, label
    
    # Audio preprocessing
    def preprocess_audio(audio_path, label):
        # Read audio file
        audio = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio)
        audio = tf.squeeze(audio, axis=-1)
        
        # Resample if needed
        audio = tf.cast(audio, tf.float32)
        
        # Compute spectrogram
        spectrogram = tf.signal.stft(
            audio,
            frame_length=255,
            frame_step=128,
            fft_length=256
        )
        spectrogram = tf.abs(spectrogram)
        
        # Convert to mel-scale
        mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=80,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=sample_rate
        )
        mel_spectrogram = tf.tensordot(spectrogram, mel_spectrogram, 1)
        
        # Log transform
        mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        
        return mel_spectrogram, label
    
    # Mixed data types preprocessing
    def preprocess_mixed_data(features):
        """Preprocess mixed data types (numerical, categorical, text)"""
        processed_features = {}
        
        # Numerical features
        if 'numerical' in features:
            numerical = features['numerical']
            # Normalize
            numerical = tf.nn.l2_normalize(numerical)
            processed_features['numerical'] = numerical
        
        # Categorical features
        if 'categorical' in features:
            categorical = features['categorical']
            # One-hot encode
            categorical = tf.one_hot(categorical, depth=10)
            processed_features['categorical'] = categorical
        
        # Text features
        if 'text' in features:
            text = features['text']
            # Assuming vectorization layer is already created
            text = vectorize_layer(text)
            processed_features['text'] = text
        
        return processed_features
    
    return preprocess_image_advanced, preprocess_audio, preprocess_mixed_data
```

### Performance Optimization

#### Dataset Performance Best Practices
```python
def optimize_dataset_performance():
    """Demonstrate dataset performance optimization techniques"""
    
    # Create sample dataset
    dataset = tf.data.Dataset.range(1000000)
    
    # BAD: Sequential operations
    def bad_pipeline(dataset):
        dataset = dataset.map(lambda x: tf.py_function(slow_function, [x], tf.int64))
        dataset = dataset.batch(32)
        dataset = dataset.repeat()
        return dataset
    
    # GOOD: Optimized pipeline
    def good_pipeline(dataset):
        # 1. Use vectorized operations instead of py_function when possible
        dataset = dataset.map(lambda x: x * 2, num_parallel_calls=tf.data.AUTOTUNE)
        
        # 2. Cache expensive operations
        dataset = dataset.cache()
        
        # 3. Shuffle before batching
        dataset = dataset.shuffle(buffer_size=1000)
        
        # 4. Batch data
        dataset = dataset.batch(32)
        
        # 5. Repeat after batching
        dataset = dataset.repeat()
        
        # 6. Prefetch for pipeline parallelism
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    # Memory-efficient large dataset handling
    def memory_efficient_pipeline(file_pattern):
        # Process files in parallel
        files = tf.data.Dataset.list_files(file_pattern)
        dataset = files.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Parse records in parallel
        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Optimize for training
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    return good_pipeline

# Performance monitoring
def monitor_dataset_performance(dataset):
    """Monitor dataset performance"""
    import time
    
    start_time = time.time()
    
    # Time first few iterations
    for i, batch in enumerate(dataset.take(10)):
        if i == 0:
            first_batch_time = time.time() - start_time
            print(f"First batch time: {first_batch_time:.2f}s")
        
        iteration_start = time.time()
        # Simulate model training step
        tf.reduce_mean(batch)
        iteration_time = time.time() - iteration_start
        print(f"Iteration {i}: {iteration_time:.3f}s")
    
    total_time = time.time() - start_time
    print(f"Total time for 10 batches: {total_time:.2f}s")

# Benchmarking dataset operations
def benchmark_dataset_operations():
    """Benchmark different dataset configurations"""
    
    def time_dataset(dataset, description, num_batches=100):
        print(f"\nTiming {description}:")
        start_time = time.time()
        
        for i, batch in enumerate(dataset.take(num_batches)):
            if i % 20 == 0:
                print(f"  Batch {i}")
        
        total_time = time.time() - start_time
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per batch: {total_time/num_batches:.3f}s")
        
        return total_time
    
    # Create base dataset
    base_dataset = tf.data.Dataset.range(10000).map(lambda x: tf.random.normal([100]))
    
    # Test different configurations
    configs = {
        "Basic": base_dataset.batch(32),
        "With cache": base_dataset.cache().batch(32),
        "With prefetch": base_dataset.batch(32).prefetch(tf.data.AUTOTUNE),
        "Optimized": base_dataset.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    }
    
    results = {}
    for name, dataset in configs.items():
        results[name] = time_dataset(dataset, name)
    
    return results

# Working with large datasets that don't fit in memory
def handle_large_datasets():
    """Strategies for handling datasets larger than memory"""
    
    # Strategy 1: Use tf.data with file-based datasets
    def create_file_based_dataset(file_pattern):
        files = tf.data.Dataset.list_files(file_pattern, shuffle=True)
        
        # Process files in parallel
        dataset = files.interleave(
            lambda x: tf.data.TextLineDataset(x),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset
    
    # Strategy 2: Use sharding for distributed training
    def create_sharded_dataset(file_pattern, num_shards, shard_index):
        files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        files = files.shard(num_shards, shard_index)
        
        dataset = files.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.AUTOTUNE
        )
        
        return dataset
    
    # Strategy 3: Streaming from cloud storage
    def create_cloud_dataset(gcs_pattern):
        # Works with Google Cloud Storage, AWS S3, etc.
        files = tf.data.Dataset.list_files(gcs_pattern)
        dataset = files.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset
    
    return create_file_based_dataset, create_sharded_dataset, create_cloud_dataset
```

## TensorFlow Hub

TensorFlow Hub is a library for the publication, discovery, and consumption of reusable parts of machine learning models. It provides pre-trained models that can be used for transfer learning, feature extraction, and fine-tuning.

### Pre-trained Models

#### Loading and Using Pre-trained Models
```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_pretrained_models():
    """Examples of loading different types of pre-trained models"""
    
    # Image classification models
    mobilenet_v2 = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")
    efficientnet = hub.load("https://tfhub.dev/tensorflow/efficientnet/b0/classification/1")
    
    # Text embedding models
    universal_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    bert_encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    
    # Object detection models
    ssd_mobilenet = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    
    # Style transfer models
    style_transfer = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
    
    return {
        'mobilenet_v2': mobilenet_v2,
        'universal_encoder': universal_encoder,
        'ssd_mobilenet': ssd_mobilenet,
        'style_transfer': style_transfer
    }

# Image classification example
def image_classification_with_hub():
    """Complete image classification pipeline using TensorFlow Hub"""
    
    # Load pre-trained MobileNet model
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    model = hub.load(model_url)
    
    def load_and_preprocess_image(image_path):
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        # Resize to model input size
        image = tf.image.resize(image, [224, 224])
        
        # Add batch dimension
        image = tf.expand_dims(image, 0)
        
        return image
    
    def predict_image_class(image_path, top_k=5):
        # Preprocess image
        image = load_and_preprocess_image(image_path)
        
        # Make prediction
        predictions = model(image)
        
        # Get top K predictions
        top_k_indices = tf.nn.top_k(predictions, k=top_k).indices[0]
        top_k_values = tf.nn.top_k(predictions, k=top_k).values[0]
        
        # Load ImageNet labels
        labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
        labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', labels_url)
        imagenet_labels = np.array(open(labels_path).read().splitlines())
        
        # Print results
        print(f"Top {top_k} predictions for {image_path}:")
        for i in range(top_k):
            label = imagenet_labels[top_k_indices[i]]
            confidence = top_k_values[i].numpy()
            print(f"  {i+1}. {label}: {confidence:.4f}")
        
        return top_k_indices, top_k_values
    
    return predict_image_class

# Text embedding example
def text_embedding_with_hub():
    """Text similarity using Universal Sentence Encoder"""
    
    # Load Universal Sentence Encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    def compute_text_similarity(texts):
        # Get embeddings
        embeddings = embed(texts)
        
        # Compute similarity matrix
        similarity_matrix = tf.linalg.matmul(embeddings, embeddings, transpose_b=True)
        
        return embeddings, similarity_matrix
    
    # Example usage
    sample_texts = [
        "The cat sat on the mat.",
        "A feline rested on the carpet.",
        "Dogs are great pets.",
        "I love machine learning.",
        "Artificial intelligence is fascinating."
    ]
    
    embeddings, similarity = compute_text_similarity(sample_texts)
    
    # Visualize similarity matrix
    def plot_similarity_matrix(texts, similarity_matrix):
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='viridis')
        plt.colorbar()
        
        # Add text labels
        plt.xticks(range(len(texts)), texts, rotation=45, ha='right')
        plt.yticks(range(len(texts)), texts)
        plt.title('Text Similarity Matrix')
        plt.tight_layout()
        plt.show()
    
    plot_similarity_matrix(sample_texts, similarity.numpy())
    
    return compute_text_similarity
```

### Transfer Learning

#### Image Classification Transfer Learning
```python
def transfer_learning_image_classification():
    """Complete transfer learning pipeline for image classification"""
    
    # Load base model from TensorFlow Hub
    base_model_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
    base_model = hub.KerasLayer(base_model_url, trainable=False)
    
    # Build complete model
    def create_transfer_model(num_classes, fine_tune=False):
        model = tf.keras.Sequential([
            # Preprocessing
            tf.keras.layers.Rescaling(1./255),
            
            # Base model
            base_model,
            
            # Custom head
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Set base model trainability
        base_model.trainable = fine_tune
        
        return model
    
    # Create model for 5 classes
    model = create_transfer_model(num_classes=5)
    
    # Compile with appropriate learning rate
    initial_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=initial_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Advanced transfer learning with fine-tuning
def advanced_transfer_learning():
    """Two-stage transfer learning: feature extraction + fine-tuning"""
    
    base_model_url = "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"
    
    def create_model_for_transfer_learning(num_classes):
        # Create base model
        base_model = hub.KerasLayer(
            base_model_url,
            input_shape=(224, 224, 3),
            trainable=False  # Freeze base model initially
        )
        
        # Add custom layers
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            base_model,
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model, base_model
    
    def two_stage_training(model, base_model, train_ds, val_ds, num_classes):
        """Two-stage training: feature extraction then fine-tuning"""
        
        # Stage 1: Train only the classifier head
        print("Stage 1: Training classifier head...")
        base_model.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for a few epochs
        history1 = model.fit(
            train_ds,
            epochs=10,
            validation_data=val_ds
        )
        
        # Stage 2: Fine-tune the entire model
        print("Stage 2: Fine-tuning entire model...")
        base_model.trainable = True
        
        # Use a lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001/10),  # Lower LR
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune for more epochs
        history2 = model.fit(
            train_ds,
            epochs=10,
            validation_data=val_ds
        )
        
        return history1, history2
    
    return create_model_for_transfer_learning, two_stage_training
```

### Fine-tuning

#### Advanced Fine-tuning Strategies
```python
def advanced_fine_tuning_strategies():
    """Advanced techniques for fine-tuning pre-trained models"""
    
    def gradual_unfreezing(model, base_model, train_ds, val_ds):
        """Gradually unfreeze layers during training"""
        
        # Start with all layers frozen except the last few
        for i, layer in enumerate(base_model.layers):
            if i < len(base_model.layers) - 10:  # Freeze all but last 10 layers
                layer.trainable = False
            else:
                layer.trainable = True
        
        # Training schedule
        learning_rates = [0.0001, 0.00005, 0.00001]
        unfreeze_points = [5, 10, 15]  # Epochs at which to unfreeze more layers
        
        histories = []
        
        for stage, (lr, unfreeze_point) in enumerate(zip(learning_rates, unfreeze_points)):
            print(f"Stage {stage + 1}: LR={lr}, Unfreezing from layer {unfreeze_point}")
            
            # Update trainable layers
            for i, layer in enumerate(base_model.layers):
                if i >= len(base_model.layers) - unfreeze_point:
                    layer.trainable = True
            
            # Recompile with new learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train for this stage
            history = model.fit(
                train_ds,
                epochs=5,
                validation_data=val_ds,
                verbose=1
            )
            
            histories.append(history)
        
        return histories
    
    def discriminative_learning_rates(model, base_model):
        """Apply different learning rates to different parts of the model"""
        
        # Define layer groups with different learning rates
        base_lr = 0.0001
        
        optimizers_and_layers = [
            (tf.keras.optimizers.Adam(lr=base_lr/10), base_model.layers[:50]),  # Early layers: very low LR
            (tf.keras.optimizers.Adam(lr=base_lr/5), base_model.layers[50:100]),  # Middle layers: low LR
            (tf.keras.optimizers.Adam(lr=base_lr), base_model.layers[100:]),  # Late layers: normal LR
        ]
        
        # This would require custom training loop implementation
        # for more control over gradient application
        pass
    
    def layer_wise_learning_rate_decay(model):
        """Implement layer-wise learning rate decay"""
        
        def custom_optimizer_config():
            # Custom learning rate schedule for different layers
            layer_configs = []
            
            for i, layer in enumerate(model.layers):
                # Decay rate based on layer depth
                decay_factor = 0.95 ** (len(model.layers) - i - 1)
                layer_lr = 0.0001 * decay_factor
                
                layer_configs.append({
                    'layer': layer,
                    'learning_rate': layer_lr
                })
            
            return layer_configs
        
        return custom_optimizer_config()
    
    return gradual_unfreezing, discriminative_learning_rates, layer_wise_learning_rate_decay
```

### Feature Extraction

#### Using Pre-trained Models for Feature Extraction
```python
def feature_extraction_examples():
    """Examples of using pre-trained models for feature extraction"""
    
    def image_feature_extraction():
        """Extract features from images using pre-trained models"""
        
        # Load feature extraction model
        feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
        feature_extractor = hub.KerasLayer(
            feature_extractor_url,
            input_shape=(299, 299, 3),
            trainable=False
        )
        
        def extract_features_from_images(image_paths):
            features = []
            
            for image_path in image_paths:
                # Load and preprocess image
                image = tf.io.read_file(image_path)
                image = tf.image.decode_image(image, channels=3)
                image = tf.image.resize(image, [299, 299])
                image = tf.cast(image, tf.float32) / 255.0
                image = tf.expand_dims(image, 0)
                
                # Extract features
                feature_vector = feature_extractor(image)
                features.append(feature_vector.numpy().flatten())
            
            return np.array(features)
        
        return extract_features_from_images
    
    def text_feature_extraction():
        """Extract features from text using pre-trained models"""
        
        # Load text encoder
        encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        
        def extract_text_features(texts):
            # Get embeddings
            embeddings = encoder(texts)
            return embeddings.numpy()
        
        def build_text_classifier_with_features(texts, labels, test_texts, test_labels):
            # Extract features
            train_features = extract_text_features(texts)
            test_features = extract_text_features(test_texts)
            
            # Build simple classifier on top of features
            classifier = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(train_features.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')
            ])
            
            classifier.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train classifier
            history = classifier.fit(
                train_features, labels,
                epochs=20,
                validation_data=(test_features, test_labels),
                verbose=1
            )
            
            return classifier, history
        
        return extract_text_features, build_text_classifier_with_features
    
    def multimodal_feature_extraction():
        """Extract and combine features from multiple modalities"""
        
        # Load different feature extractors
        image_encoder = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2/feature_vector/4")
        text_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        
        def extract_multimodal_features(images, texts):
            # Extract image features
            image_features = image_encoder(images)
            
            # Extract text features
            text_features = text_encoder(texts)
            
            # Combine features
            combined_features = tf.concat([image_features, text_features], axis=1)
            
            return combined_features
        
        def build_multimodal_classifier(image_data, text_data, labels):
            # Extract features
            features = extract_multimodal_features(image_data, text_data)
            
            # Build classifier
            classifier = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(features.shape[1],)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')
            ])
            
            classifier.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return classifier, features
        
        return extract_multimodal_features, build_multimodal_classifier
    
    return image_feature_extraction, text_feature_extraction, multimodal_feature_extraction

# Practical example: Building a custom model with TensorFlow Hub
def complete_hub_example():
    """Complete example using TensorFlow Hub for a real-world task"""
    
    def build_document_classifier():
        """Build a document classifier using pre-trained text embeddings"""
        
        # Load pre-trained text encoder
        text_encoder_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        hub_layer = hub.KerasLayer(text_encoder_url, input_shape=[], dtype=tf.string, trainable=True)
        
        # Build model
        model = tf.keras.Sequential([
            hub_layer,
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, activation='softmax')  # 5 document categories
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_document_classifier(texts, labels):
        """Train the document classifier"""
        
        model = build_document_classifier()
        
        # Prepare data
        train_texts = np.array(texts)
        train_labels = tf.keras.utils.to_categorical(labels)
        
        # Train model
        history = model.fit(
            train_texts, train_labels,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        return model, history
    
## Learning Objectives

By the end of this section, you should be able to:

### Core Competencies
- **Build models using both Sequential and Functional APIs** with appropriate choice based on architecture complexity
- **Create custom layers and models** for specialized architectures and novel approaches
- **Implement custom training loops** using GradientTape for fine-grained control over training process
- **Design efficient data pipelines** using tf.data API with proper preprocessing and optimization
- **Apply transfer learning techniques** using TensorFlow Hub for various domains (vision, NLP, audio)
- **Optimize model performance** through proper dataset handling, caching, and prefetching strategies

### Advanced Skills
- **Debug and monitor training** using callbacks, metrics, and visualization tools
- **Implement distributed training** strategies for large-scale model training
- **Handle multiple data modalities** (images, text, audio) in unified models
- **Apply advanced fine-tuning strategies** including gradual unfreezing and discriminative learning rates
- **Build production-ready pipelines** with proper error handling and resource management

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Create both Sequential and Functional API models from scratch  
□ Implement at least 2 custom layer types with proper weight initialization  
□ Write a complete custom training loop with validation and metrics  
□ Build an optimized data pipeline that handles 10GB+ datasets efficiently  
□ Successfully fine-tune a pre-trained model for a new task  
□ Implement proper callbacks for early stopping and learning rate scheduling  
□ Handle mixed data types (numerical, categorical, text, images) in one model  
□ Debug training issues using TensorBoard and other monitoring tools  
□ Apply transfer learning across different domains (vision ↔ NLP)  
□ Optimize dataset performance to achieve >90% GPU utilization  

## Practical Exercises

### Exercise 1: Multi-Modal Model
```python
# TODO: Build a model that processes both images and text descriptions
# Requirements:
# - Use pre-trained image encoder from TensorFlow Hub
# - Use pre-trained text encoder from TensorFlow Hub
# - Combine features and add classification head
# - Handle different input shapes properly

def build_multimodal_model():
    # Your implementation here
    pass

# Test with: image + text → category prediction
```

### Exercise 2: Custom Training Loop with Advanced Features
```python
# TODO: Implement a training loop with:
# - Custom loss function (e.g., focal loss)
# - Learning rate scheduling
# - Gradient clipping
# - Mixed precision training
# - Custom metrics (F1, precision, recall)

@tf.function
def advanced_train_step(model, optimizer, loss_fn, images, labels):
    # Your implementation here
    pass
```

### Exercise 3: Efficient Data Pipeline
```python
# TODO: Create a data pipeline that:
# - Loads images from 1000+ files
# - Applies random augmentations
# - Handles variable image sizes
# - Achieves <0.1s per batch loading time
# - Includes proper error handling

def create_optimized_pipeline(data_dir, batch_size=32):
    # Your implementation here
    pass
```

### Exercise 4: Transfer Learning Pipeline
```python
# TODO: Implement complete transfer learning workflow:
# - Load pre-trained model from TensorFlow Hub
# - Freeze/unfreeze layers strategically
# - Implement two-stage training
# - Add proper evaluation metrics
# - Save and load the fine-tuned model

def transfer_learning_workflow(base_model_url, dataset, num_classes):
    # Your implementation here
    pass
```

## Study Materials

### Essential Reading
- **Primary:** [TensorFlow 2.0 Complete Course](https://www.tensorflow.org/tutorials) - Official tutorials
- **Secondary:** "Hands-On Machine Learning" by Aurélien Géron - Chapters 10-16
- **Advanced:** [TensorFlow Hub Guide](https://www.tensorflow.org/hub/tutorials) - Transfer learning tutorials
- **Reference:** [tf.data Performance Guide](https://www.tensorflow.org/guide/data_performance)

### Video Resources
- **TensorFlow Official YouTube Channel** - Model building tutorials
- **DeepLearning.AI TensorFlow Specialization** - Comprehensive course series
- **Fast.ai Practical Deep Learning** - Transfer learning best practices
- **Google I/O TensorFlow Sessions** - Latest features and techniques

### Hands-on Labs
- **Lab 1:** Build an image classifier using Sequential API
- **Lab 2:** Create a text sentiment analyzer with Functional API
- **Lab 3:** Implement custom layers for attention mechanism
- **Lab 4:** Build end-to-end data pipeline for large dataset
- **Lab 5:** Fine-tune BERT for domain-specific classification
- **Lab 6:** Create multi-GPU training setup

### Documentation and References
- [Keras API Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [tf.data API Guide](https://www.tensorflow.org/guide/data)
- [TensorFlow Hub Models](https://tfhub.dev/)
- [Model Garden](https://github.com/tensorflow/models) - Official model implementations
- [TensorFlow Performance Guide](https://www.tensorflow.org/guide/gpu_performance_analysis)

### Practice Datasets
- **Image Classification:** CIFAR-10, ImageNet subset, Custom datasets
- **Text Classification:** IMDB reviews, Reuters news, Tweet sentiment
- **Multimodal:** MS-COCO (images + captions), VQA datasets
- **Time Series:** Stock prices, Weather data, IoT sensor data
- **Audio:** Speech commands, Music genre classification

### Development Environment Setup

**Required Installations:**
```bash
# Core TensorFlow
pip install tensorflow>=2.13.0
pip install tensorflow-hub
pip install tensorflow-datasets

# Visualization and analysis
pip install matplotlib seaborn
pip install tensorboard
pip install jupyter

# Additional tools
pip install pillow  # Image processing
pip install librosa  # Audio processing
pip install transformers  # Hugging Face models
```

**GPU Setup (Optional but Recommended):**
```bash
# For NVIDIA GPUs
pip install tensorflow[and-cuda]

# Verify GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Development Tools:**
```bash
# Code quality
pip install black isort flake8

# Experiment tracking
pip install wandb mlflow

# Model serving
pip install tensorflow-serving-api
```

### Common Issues and Solutions

**Performance Issues:**
- Use `tf.data.AUTOTUNE` for automatic optimization
- Implement proper caching with `.cache()`
- Use `.prefetch()` for pipeline parallelism
- Monitor GPU utilization with `nvidia-smi`

**Memory Issues:**
- Use mixed precision training: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- Implement gradient accumulation for large batch sizes
- Use dataset sharding for distributed training

**Training Issues:**
- Monitor gradients for vanishing/exploding problems
- Use proper weight initialization
- Implement learning rate scheduling
- Add proper regularization (dropout, batch norm)

### Assessment Questions

**Conceptual Questions:**
1. When would you choose Functional API over Sequential API? Provide 3 specific scenarios.
2. Explain the difference between `model.fit()` and custom training loops. When is each appropriate?
3. How does `tf.data.AUTOTUNE` optimize data pipeline performance?
4. What are the key considerations when fine-tuning a pre-trained model?
5. Explain the trade-offs between different transfer learning strategies.

**Technical Questions:**
6. How do you handle variable-length sequences in TensorFlow?
7. What's the difference between `trainable=False` and `training=False`?
8. How do you implement gradient accumulation in TensorFlow?
9. Explain the memory implications of different batching strategies.
10. How do you debug NaN losses in training?

**Coding Challenges:**
```python
# Challenge 1: Fix the performance bottleneck
def slow_data_pipeline(file_paths):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(lambda x: tf.py_function(slow_preprocess, [x], tf.float32))
    dataset = dataset.batch(32)
    return dataset

# Challenge 2: Implement memory-efficient training for large models
def train_large_model(model, dataset, steps_per_epoch):
    # Implement gradient accumulation and mixed precision
    pass

# Challenge 3: Create a custom layer with proper serialization
class AttentionLayer(tf.keras.layers.Layer):
    # Implement multi-head attention with save/load support
    pass
```

### Next Steps
- **Advanced Topics:** Custom training strategies, model optimization
- **Deployment:** TensorFlow Serving, TensorFlow Lite, TensorFlow.js
- **Research:** Latest architectures, SOTA techniques
- **Production:** MLOps, monitoring, A/B testing
