import jax.numpy as jnp
import numpy as np
import rerun as rr

rr.init("jax_demo", spawn=True)   # spawn=True opens the viewer window

for step in range(100):
    rr.set_time("step", sequence=step)
    x = jnp.sin(step / 10.0)
    rr.log("signal/sin", rr.Scalars(float(x)))

# arrays: convert jax -> numpy before logging
img = np.asarray(jnp.ones((64, 64, 3)) * 255, dtype=np.uint8)
rr.log("image", rr.Image(img))