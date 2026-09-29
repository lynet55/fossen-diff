### Fossen Dynamics for UUV sim

- Differentiable Fossen dynamics for a UUV
- Single file implementations


Desing decitions i made:

- Prediciton model implemented in JAX for differentability, motivated by CBFs
- Plant is PyTorch based, mostly for possible mjlab extencion
- Plant and prediction model seperate, introduce some discrepency

TODO:
- Bridge godot and this control module. Ros2 topic? Sonar simulation, turned into occupancy grid for jax
- RosPublisher to send back to godot?