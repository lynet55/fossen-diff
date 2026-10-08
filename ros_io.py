"""Shared ROS / sim I/O for the single-file controllers in modifiers/.

A controller is any callable `Observation -> Command`. This file does everything around it:

    ROS topics --Inputs--> Observation --controller--> Command --sinks--> ROS topics / sim / RViz

Everything here is in the fossen convention (NED world, FRD body); the sim's Z-up/FLU frames
are converted at the edges with F = diag(1, -1, -1):  p_ned = F p,  R_ned = F R F,  nu_frd = F nu_flu.

Extending:
    new input   add a field to Observation, a subscription in Inputs.__init__, fill it in observe()
    new output  write a class with __init__(self, node) and send(self, cmd), add it to a sink list
    new sim     write its action sink (like MarineGymUdp) and a *_sinks() list next to marinegym_sinks()

Run a controller:  ros_io.run(controller, dt, sinks=ros_io.marinegym_sinks(vel_max))
"""

import json
import socket
import time
from dataclasses import dataclass

import numpy as np

F = np.diag([1.0, -1.0, -1.0])  # Z-up/FLU <-> NED/FRD, its own inverse

# Topics of the MarineGym telemetry bridge (uuv-rl-env/scripts/ros2_udp_telemetry_publisher.py).
STATE_TOPIC = "/bluerov/odom"              # nav_msgs/Odometry: pose in `map` (Z-up), twist in WORLD frame
SONAR_TOPIC = "/bluerov/sonar/scan"        # sensor_msgs/LaserScan: planar, in bluerov_base_link (FLU)
MAP_TOPIC = "/bluerov/map"                 # nav_msgs/OccupancyGrid: 2D, in `map`, 0..100
GOAL_TOPIC = "/bluerov/goal"               # geometry_msgs/PoseStamped: in `map`
NOMINAL_TOPIC = "/fossen/nominal_cmd_vel"  # geometry_msgs/Twist: optional pilot/policy command, body FLU
NOMINAL_TIMEOUT = 0.5                      # [s] older nominal commands are dropped


# --------------------------------------------------------------------------- #
# Data passed to and from a controller
# --------------------------------------------------------------------------- #
@dataclass
class Grid:
    occ: np.ndarray     # (H, W) 0..100, rows along map y, cols along map x
    origin: np.ndarray  # (2,) map xy of cell (0, 0)
    res: float          # [m] cell size
    version: int        # bumps on every new map, so controllers can cache derived fields


@dataclass
class Observation:
    t: float                         # [s] node clock
    x: np.ndarray                    # (18,) [nu FRD (6), pos NED (3), R body->NED (9)]
    goal: np.ndarray | None = None   # (3,) NED
    nominal: np.ndarray | None = None  # (4,) [u, v, w, r] FRD, only while fresh
    scan: np.ndarray | None = None   # (N, 2) NED xy of sonar hits (variable N)
    grid: Grid | None = None


@dataclass
class Command:
    tau: np.ndarray                     # (6,) [Fx, Fy, Fz, Tx, Ty, Tz] body FRD
    nu_next: np.ndarray | None = None   # (6,) body FRD velocity the controller wants next
    rollouts: np.ndarray | None = None  # (K, T, 3) NED positions to draw
    scores: np.ndarray | None = None    # (K,) in [0, 1], 1 = best (drawn green)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class Inputs:
    """Subscribes to the sim topics and packs the latest messages into an Observation."""

    def __init__(self, node):
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import OccupancyGrid, Odometry
        from sensor_msgs.msg import LaserScan

        self.node = node
        self.msgs = {}
        self.nominal_t = -1e9
        self.grid = None

        def keep(name):
            return lambda m: self.msgs.__setitem__(name, m)

        def on_nominal(m):
            self.msgs["nominal"], self.nominal_t = m, self.now()

        def on_map(m):
            occ = np.asarray(m.data, np.int16).reshape(m.info.height, m.info.width)
            origin = np.array([m.info.origin.position.x, m.info.origin.position.y])
            self.grid = Grid(occ, origin, float(m.info.resolution), 0 if self.grid is None else self.grid.version + 1)

        node.create_subscription(Odometry, STATE_TOPIC, keep("odom"), 10)
        node.create_subscription(LaserScan, SONAR_TOPIC, keep("scan"), 10)
        node.create_subscription(PoseStamped, GOAL_TOPIC, keep("goal"), 10)
        node.create_subscription(Twist, NOMINAL_TOPIC, on_nominal, 10)
        node.create_subscription(OccupancyGrid, MAP_TOPIC, on_map, 1)

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def observe(self):
        """Latest Observation, or None until the first state arrives."""
        from scipy.spatial.transform import Rotation

        o = self.msgs.get("odom")
        if o is None:
            return None

        # State: Z-up map pose + WORLD-frame twist -> NED pose, FRD body velocity.
        p, q = o.pose.pose.position, o.pose.pose.orientation
        v, w = o.twist.twist.linear, o.twist.twist.angular
        R_zup = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()  # body FLU -> map
        R = F @ R_zup @ F
        nu = np.concatenate([F @ R_zup.T @ [v.x, v.y, v.z], F @ R_zup.T @ [w.x, w.y, w.z]])
        pos = F @ [p.x, p.y, p.z]
        obs = Observation(t=self.now(), x=np.concatenate([nu, pos, R.ravel()]), grid=self.grid)

        g = self.msgs.get("goal")
        if g is not None:
            obs.goal = F @ [g.pose.position.x, g.pose.position.y, g.pose.position.z]

        n = self.msgs.get("nominal")
        if n is not None and obs.t - self.nominal_t < NOMINAL_TIMEOUT:
            obs.nominal = np.array([n.linear.x, -n.linear.y, -n.linear.z, -n.angular.z])  # FLU -> FRD

        s = self.msgs.get("scan")
        if s is not None:  # planar ranges in body FLU -> NED xy hit points
            r = np.asarray(s.ranges, np.float64)
            a = s.angle_min + s.angle_increment * np.arange(len(r))
            hit = np.isfinite(r) & (r >= s.range_min) & (r <= s.range_max)
            body = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], -1)[hit]
            obs.scan = (pos + (R @ F @ body.T).T)[:, :2]
        return obs


# --------------------------------------------------------------------------- #
# Sinks: each takes a Command and sends one thing somewhere
# --------------------------------------------------------------------------- #
class WrenchOut:
    """tau as geometry_msgs/WrenchStamped, body FRD (fossen convention)."""
    topic = "/fossen/modified_input"

    def __init__(self, node):
        from geometry_msgs.msg import WrenchStamped
        self.node, self.Msg = node, WrenchStamped
        self.pub = node.create_publisher(WrenchStamped, self.topic, 10)

    def send(self, cmd):
        m = self.Msg()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.wrench.force.x, m.wrench.force.y, m.wrench.force.z = map(float, cmd.tau[0:3])
        m.wrench.torque.x, m.wrench.torque.y, m.wrench.torque.z = map(float, cmd.tau[3:6])
        self.pub.publish(m)


def nu_to_flu4(nu):
    """(6,) body FRD velocity -> [u, v, w, r] body FLU."""
    return np.array([nu[0], -nu[1], -nu[2], -nu[5]])


class CmdVelOut:
    """nu_next as geometry_msgs/Twist, body FLU."""
    topic = "/fossen/modified_cmd_vel"

    def __init__(self, node):
        from geometry_msgs.msg import Twist
        self.Msg = Twist
        self.pub = node.create_publisher(Twist, self.topic, 10)

    def send(self, cmd):
        if cmd.nu_next is None:
            return
        u, v, w, r = map(float, nu_to_flu4(cmd.nu_next))
        m = self.Msg()
        m.linear.x, m.linear.y, m.linear.z, m.angular.z = u, v, w, r
        self.pub.publish(m)


class RolloutsOut:
    """Rollouts as one visualization_msgs/Marker LINE_LIST in `map`, colored red (0) -> green (1)."""
    topic = "/mppi_rollouts"

    def __init__(self, node):
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker
        self.node, self.Point, self.Color, self.Marker = node, Point, ColorRGBA, Marker
        self.pub = node.create_publisher(Marker, self.topic, 1)

    def send(self, cmd):
        if cmd.rollouts is None:
            return
        pts = np.asarray(cmd.rollouts) @ F  # NED -> map
        c = np.ones(len(pts)) if cmd.scores is None else np.clip(np.asarray(cmd.scores), 0.0, 1.0)
        m = self.Marker()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.ns, m.id, m.type, m.action = "rollouts", 0, self.Marker.LINE_LIST, self.Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.01
        seg = np.stack([pts[:, :-1], pts[:, 1:]], axis=2).reshape(-1, 3).tolist()  # a0, b0, a1, b1, ...
        m.points = [self.Point(x=x, y=y, z=z) for x, y, z in seg]
        colors = [self.Color(r=float(1 - ck), g=float(ck), b=0.0, a=float(0.15 + 0.85 * ck)) for ck in c]
        m.colors = [colors[k] for k in range(len(c)) for _ in range(2 * (pts.shape[1] - 1))]
        self.pub.publish(m)


class MarineGymUdp:
    """nu_next as MarineGym's physics input: UDP JSON {"action": [u, v, w, r]}, body FLU / vel_max,
    clipped to [-1, 1]. Same packet as uuv-rl-env/scripts/ros2_joy_to_udp.py (action_mode=ros2_joy).
    vel_max must match the sim's task.controller.{u,v,w,r}_max."""

    def __init__(self, node, vel_max, host="127.0.0.1", port=15000):
        self.vel_max, self.target = np.asarray(vel_max, np.float64), (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, cmd):
        if cmd.nu_next is None:
            return
        action = np.clip(nu_to_flu4(cmd.nu_next) / self.vel_max, -1.0, 1.0).tolist()
        self.sock.sendto(json.dumps({"stamp": time.time(), "action": action}).encode(), self.target)


def ros_sinks():
    """Topics + RViz only (no sim)."""
    return [WrenchOut, CmdVelOut, RolloutsOut]


def marinegym_sinks(vel_max, **udp):
    """ros_sinks() plus the MarineGym action port."""
    return ros_sinks() + [lambda node: MarineGymUdp(node, vel_max, **udp)]


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
def run(controller, dt, sinks=None, name="controller"):
    """Spin a node that calls `controller(obs) -> Command` every dt and fans it out to the sinks.
    `sinks` is a list of factories `node -> sink`; defaults to ros_sinks()."""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = rclpy.create_node(name)
    inputs = Inputs(node)
    outs = [make(node) for make in (ros_sinks() if sinks is None else sinks)]

    def tick():
        if not rclpy.ok():  # SIGINT/SIGTERM shut the context down mid-spin
            return
        obs = inputs.observe()
        if obs is None:
            return
        cmd = controller(obs)
        try:
            for out in outs:
                out.send(cmd)
        except Exception:
            if rclpy.ok():
                raise  # a real error; otherwise the context was shut down mid-send

    node.create_timer(dt, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():  # Ctrl-C already shut the context down
            rclpy.shutdown()
