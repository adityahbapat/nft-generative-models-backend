## Creating NFT using generative models backend

## Generative Models used for image creation:
- Stable Diffusion (Text to Image)
- Neural Style Transfer (One Image to Another)
- Deep Dream (visualizes the patterns learned by a neural network)

## GPU required
- The GPU configuration uhhhhh, almost made me cry :)
- On windows run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted` 
- ``` pip list``` to see all packages
- Please make sure of your tensorflow version and if it supports Cuda
 `tf.test.is_built_with_cuda()`
- I uninstalled my previous tensorflow version, upgraded pip and reinstalled version 2.10.0 for python version 3.10.6
``` pip uninstall -y $(pip freeze | Select-String tensor | ForEach-Object { $_.ToString().Split('==')[0] }) ``` same for keras
`pip install tensorflow==2.10.0 `
- `pip cache purge` to clear pip cache
- Finally run code:
``` 
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
# Get the list of available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  for gpu in gpus:
    print("GPU detected:", gpu)
else:
  print("No GPU detected")

# Check if TensorFlow is using the GPU
print("Is TensorFlow using GPU?", tf.test.is_built_with_cuda())
print("Is GPU available for TensorFlow?", tf.test.is_gpu_available()) 
```

- with this tf version keras_cv version 0.0.25 is compatible, any other version will lead to error
