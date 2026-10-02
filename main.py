import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from jax.scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry

from models.fossen_diff import fossen_rollout_jit

DT = 0.05
ODOM_TOPIC = "/fossen/odom"  # published AND subscribed -> closed-loop rollout
INPUT_TOPIC = "/fossen/modified_input"  # tau = [Fx, Fy, Fz, Tx, Ty, Tz], same input as the dynamics

rclpy.init()
node = rclpy.create_node("fossen_propagator")
odom_pub = node.create_publisher(Odometry, ODOM_TOPIC, 10)
odom = None
tau = np.zeros(6)


def on_odom(msg):
    global odom
    odom = msg


def on_input(msg):
    global tau
    f, t = msg.wrench.force, msg.wrench.torque
    tau = np.array([f.x, f.y, f.z, t.x, t.y, t.z])


def on_timer():
    if odom is None:
        return
    p, o, v, w = odom.pose.pose.position, odom.pose.pose.orientation, odom.twist.twist.linear, odom.twist.twist.angular

    # One-step propagation from the latest state.
    out = fossen_rollout_jit([v.x, v.y, v.z, w.x, w.y, w.z], tau[None], DT,
                             init_pos=[p.x, p.y, p.z],
                             init_R=Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix())
    nu, pos, q = (np.asarray(x) for x in (out["nu"][0], out["pos"][0], Rotation.from_matrix(out["R"][0]).as_quat()))

    msg = Odometry(header=odom.header)
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, pos)
    msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = map(float, q)
    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = map(float, nu[0:3])
    msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z = map(float, nu[3:6])
    odom_pub.publish(msg)


node.create_subscription(Odometry, ODOM_TOPIC, on_odom, 10)
node.create_subscription(WrenchStamped, INPUT_TOPIC, on_input, 10)
node.create_timer(DT, on_timer)

# Seed the loop: origin, level, moving forward at 0.5 m/s.
odom = Odometry()
odom.header.frame_id = "odom"
odom.pose.pose.orientation.w = 1.0
odom.twist.twist.linear.x = 0.5

try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
