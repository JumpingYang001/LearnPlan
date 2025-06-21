# Google TPU and JAX

## Topics
- TPU architecture and programming model
- XLA (Accelerated Linear Algebra)
- JAX for TPU programming
- TPU-accelerated applications

### Example: JAX on TPU (Python)
```python
import jax
import jax.numpy as jnp
x = jnp.ones((3, 3))
print(jax.device_get(x))
```
