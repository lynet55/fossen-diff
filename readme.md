### Fossen Dynamics for UUV sim

Repositroy is meant as a minimal playground for modifying algorithms such as mppi or other's filter like algorithms. Meant to be very fast to iterate, and has a delibererate focus on beeing simple for rapid iteration and keeping account across iterations managable.

The core contribution of the repo is:

- Differentiable Fossen dynamics for a UUV
- Single file implementations of filters and stuff such as mppi. These are deliberatly keept seperate from each other for the sake of rapid iteration.

Current rule is, there is one common ros interface for subscribing and publishing onto ros topics for the sake of easy integration with other simulators or deployment. Otherwise every runtime filter has a single file implementation with a option to run it as a node or step the algorithm once for testing purposes.

#### Running the ros node

Open rviz (from `fossen-diff/`). The `env -u` part is only needed in the VS Code snap terminal, where its GTK variables crash rviz2:

```bash
source /opt/ros/jazzy/setup.bash && env -u GTK_PATH -u GIO_MODULE_DIR -u GTK_EXE_PREFIX -u LOCPATH -u GSETTINGS_SCHEMA_DIR -u GTK_IM_MODULE_FILE rviz2 -d fossen.rviz
```

Run a filter node (from `fossen-diff/`), or start both at once. Each filter is a
single self-contained file that runs as a node; swap in a different one here:

```bash
source /opt/ros/jazzy/setup.bash && export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages && (env -u GTK_PATH -u GIO_MODULE_DIR -u GTK_EXE_PREFIX -u LOCPATH -u GSETTINGS_SCHEMA_DIR -u GTK_IM_MODULE_FILE rviz2 -d fossen.rviz &) && uv run python modifiers/vanilla_mppi.py
```

#### Closed loop with MarineGym

Run `./run.sh` from the parent repo. Controllers in `modifiers/` only map `ros_io.Observation -> ros_io.Command`;
`ros_io.py` owns the ROS inputs, the output sinks (wrench, cmd_vel, rollout markers) and MarineGym's UDP action
port (`{"action": [u, v, w, r]}`, body FLU / `VEL_MAX`, UDP:15000). Add an input/output there, not in a controller.
`VEL_MAX` must match the sim's `task.controller.{u,v,w,r}_max`.

More details: [docs/implementation_details.md](docs/implementation_details.md)

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
