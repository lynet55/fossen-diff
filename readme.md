### Fossen Dynamics for UUV sim

- Differentiable Fossen dynamics for a UUV
- Single file implementations of filters and stuff such as mppi


Desing decitions i made:

- Prediciton model implemented in JAX for differentability, motivated by CBFs
- Plant is PyTorch based, mostly for possible mjlab extencion
- Plant and prediction model seperate, introduce some discrepency

Todo's:
- Fossen model for prediciton model. Real motivation is only for CBF?
- Write down the Fossen dynamics in your overleaf

- Figure out if there is anything to gain from using godot as the plant simulator, because if you are staying on gpu only you might aswell just use marine gym as the sim. If you care about sonar quality only at inference time, concsider using only godot sim at inference time, possibly exlcude sonar during policy training, quality mismatch? Use sonar only to inform something like pa mppi or a cbf.
- Bridge godot and this control module. Ros2 topic? Sonar simulation, turned into occupancy grid for jax
- RosPublisher to send back to godot?


Related repo's:

https://git.ntnu.no/ntnu-frl/SimEnvGoDot#

Fossen Dynamic's: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10215600

PyTorch/Isac Sim RL implementation of this UUV: https://marine-gym.com/, https://github.com/eather0056/bluerov-bt-autonomous/tree/main/docs
