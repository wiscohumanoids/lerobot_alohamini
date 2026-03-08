import os
import zmq
import sys
import cv2
import json
import time
import math
import random
import base64
import argparse
import numpy as np
from isaacsim import SimulationApp

def log(msg: str):
    print(f"\033[1;36m{msg}\033[0m")

def error(msg: str):
    print(f"\033[1;31m[ERROR] {msg}\033[0m")

log("[ENV] AlohaMini sim startup")


PORT_OBS = 5556
PORT_CMD = 5555

CAM_RESOLUTION = (640, 480)
FPS = 30

SIM_CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": 3286,  # Show Grid
}

# Immediately start simulation - required by IssacSim (...?)
simulation_app = SimulationApp(SIM_CONFIG)


# Imports after SimulationApp
import isaacsim.core.utils.prims as prim_utils  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import omni.usd
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.articulations import Articulation, ArticulationSubset  # noqa: E402
from omni.isaac.core.controllers import ArticulationController
from isaacsim.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot  # noqa: E402
from omni.kit.commands import execute # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: F401


class IsaacWorld:
    def __init__(self, usd_path: str, aloha_prim_path: str, verbose=False):
        log("[World] Creating IsaacSim world...")
        self.isaacWorld = World(stage_units_in_meters=1.0)

        add_reference_to_stage(usd_path=usd_path, prim_path="/World")

        log(f"[World] Loaded USD asset from {usd_path}")

        self.aloha = AlohaSim(self.isaacWorld, aloha_prim_path, verbose=verbose)

        #for prim_path in other_prim_paths:
        #    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        #    log(f"[World] Loaded additional USD asset from {usd_path} at prim path {prim_path}")



class AlohaSim:
    def __init__(self, world, prim_path: str, verbose=False):
        self.prim_path = prim_path
        self.cameras = {}
        self.verbose = verbose

        # Add Articulation to robot prim
        log("[Aloha] Setting up sim robot w/ articulation...")


        self.robot = Articulation(prim_path=self.prim_path, name="/World/Scene/AlohaMini")
               
        # Add cameras
        self._add_camera("camera_front", "/World/Scene/AlohaMini/base_link/camera_front")   # figure out actual placements & base orientations
        self._add_camera("camera_top", "/World/Scene/AlohaMini/base_link/camera_top")
        self._add_camera("camera_back", "/World/Scene/AlohaMini/base_link/camera_back")
        self._add_camera("camera_left", "/World/Scene/AlohaMini/left_Wrist_Pitch_Roll/camera_left")
        self._add_camera("camera_right", "/World/Scene/AlohaMini/right_Wrist_Pitch_Roll/camera_right")

        world.scene.add(self.robot)
        world.reset()

        self.robot.initialize()


        log("[Aloha] Model setup complete, robot and cameras initialized.")

        # Add dof names mapping
        self.dof_names = self.robot.dof_names   # Note: these names should be different, but the order in which they're listed remains consistent and should be good for now -- TODO: address me in the future
        
        self.dof_indices = {}
        for _, name in enumerate(self.dof_names):
            self.dof_indices[name] = self.robot.get_dof_index(name) # just to be extra sure about the order here

        log("[Aloha] DoF/joint map: " + str(self.dof_indices))

        self.wheel_indices = []
        try:
            self.wheel_indices = [self.dof_indices["wheel1_joint"], self.dof_indices["wheel2_joint"], self.dof_indices["wheel3_joint"]]
        except Exception as e:
            error(f"[Aloha] Error identifying wheel joint indices: {e}")
        self.virtual_base_indices = [self.dof_indices["root_x_axis_joint"], self.dof_indices["root_y_axis_joint"], self.dof_indices["root_z_rotation_joint"]]
        self.joint_indices = [idx for idx in range(len(self.dof_names)) if idx not in self.virtual_base_indices and idx not in self.wheel_indices]

        log("[Aloha] Joint indices indices: " + str(self.joint_indices))
        log("[Aloha] Virtual base indices: " + str(self.virtual_base_indices))
        log("[Aloha] Wheel joint indices: " + str(self.wheel_indices))


    def _add_camera(self, name, prim_path):

        # Apply domain randomization to camera position
        # Small random perturbation to translation (+- 2cm) and rotation (+- 2 deg)
        # This helps the model become robust to slight calibration errors in the real world
        #translation += np.random.uniform(-0.02, 0.02, size=3)
        #rotation_euler_deg = rotation_euler_deg.astype(np.float64) + np.random.uniform(-2, 2, size=3)
        
        # rotation in sim is usually quaternion
        # rotation_euler_deg: [x, y, z]
        #rot_quat = euler_angles_to_quat(np.radians(rotation_euler_deg))
        
        camera = Camera(
            prim_path=prim_path,
            #position=translation,
            frequency=FPS,
            resolution=CAM_RESOLUTION,
            #orientation=rot_quat
        )
        camera.initialize()
        self.cameras[name] = camera

        #log(f"[Aloha] Adding camera '{name}' at {prim_path} with translation {translation} and rotation (deg) {rotation_euler_deg}")
        log(f"[Aloha] Adding camera '{name}' at {prim_path}")

    def _calc_wheel_vels(self, vX, vY, vTheta):
        return {
            "root_x_axis_joint": vX,
            "root_y_axis_joint": vY,
            "root_z_rotation_joint": vTheta
        }

        WHEEL_RADIUS = 0.05  # meters
        BASE_RADIUS = 0.125  # meters

        theta_rad = vTheta * (np.pi / 180.0)
        velocity_vector = np.array([vX, vY, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]))
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        #m = np.array([[np.cos(a), np.sin(a), BASE_RADIUS] for a in angles])

        # For each wheel, the rolling direction is perpendicular to the radial vector
        # Wheel speed = -sin(alpha)*Vx + cos(alpha)*Vy + R*vTheta
        m = np.array([[-np.sin(a), np.cos(a), BASE_RADIUS] for a in angles])


        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / WHEEL_RADIUS

        return {
            "base_left_wheel": wheel_angular_speeds[0],
            "base_back_wheel": wheel_angular_speeds[1],
            "base_right_wheel": wheel_angular_speeds[2],
        }


    def _set_articulation(self, joint_positions: dict, vX, vY, vTheta):
        if not joint_positions:
            if self.verbose:
                log("[Aloha] No joint positions provided in command, skipping update.")
            return
        
        # Refer to: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.prims/docs/index.html#isaacsim.core.prims.SingleArticulation.get_joint_velocities

        # inefficient, but should be fine 
        current_pos = self.robot.get_joint_positions()
        current_vel = self.robot.get_joint_velocities()

        target_pos = current_pos.copy()
        target_vel = current_vel.copy()

        #target_pos[self.wheel_indices] = None  # ignore position control for base joints
        target_pos[self.virtual_base_indices] = None  # ignore position control for base joints
        target_vel[self.joint_indices] = None  # ignore velocity control for arm joints

        for name, pos in joint_positions.items():
            if name in self.dof_indices:
                idx = self.dof_indices[name]
                target_pos[idx] = pos

        wheel_cmds = self._calc_wheel_vels(vX, vY, vTheta)
        for name, vel in wheel_cmds.items():
            if name in self.dof_indices:
                idx = self.dof_indices[name]
                target_vel[idx] = vel
                
        action = ArticulationAction(joint_positions=target_pos, joint_velocities=target_vel)
        self.robot.apply_action(action)

        if self.verbose:
            log(f"[Aloha] Current joint positions: {current_pos}")
            log(f"[Aloha] Current joint velocities: {current_vel}")

            log(f"[Aloha] Received joint position commands: {joint_positions}")
            log(f"[Aloha] Received velocity commands: vX={vX}, vY={vY}, vTheta={vTheta}")

            log(f"[Aloha] Target joint positions: {target_pos}")
            log(f"[Aloha] Target joint velocities: {target_vel}")


    def send_action(self, action: dict) -> dict:
        if self.verbose:
            log(f"[Aloha] Received action command: {action}")

        vX = vY = vTheta = 0.0
        joint_cmds = {}

        for k, v in action.items():
            if k == "reset" and v is True:
                log("[HOST] Resetting Aloha environment...")
                self.robot.reset()
                joint_cmds = {name: 0.0 for name in self.dof_names}
                self._set_joint_positions(joint_cmds)
                continue
            if k.endswith(".pos"):
                joint_name = k.replace(".pos", "").removeprefix("arm_")
                if joint_name == "left_gripper" or joint_name == "right_gripper":
                    v = (v - 45) * (63 / 45)
                v = v * math.pi / 180.0

                joint_cmds[joint_name] = v
            elif k in ["x.vel", "y.vel", "theta.vel"]:
                if k == "x.vel":
                    vX = v
                elif k == "y.vel":
                    vY = v
                elif k == "theta.vel":
                    vTheta = v
            elif k == "lift_axis.height_mm":
                joint_cmds["vertical_move"] = (v / 1000.0) - 0.522  # mm -> m, with offset

        if self.verbose:
            log(f"[Aloha] Parsed action - vX: {vX}, vY: {vY}, vTheta: {vTheta}, joint_cmds: {joint_cmds}")

        self._set_articulation(joint_cmds, vX, vY, vTheta)

    def get_observation(self) -> dict:
        obs = {}
        
        # Joints
        joint_pos = self.robot.get_joint_positions()
        
        for i, name in enumerate(self.dof_names):
            if i == self.dof_indices.get("lift_axis", -1):
                # Convert lift height to mm for observation
                obs[f"{name}.pos"] = float(joint_pos[i] * 1000.0)  # m -> mm
            else:
                obs[f"{name}.pos"] = float(joint_pos[i])
            
        # Base (Ground Truth for now)
        pose = self.robot.get_world_pose()
        obs["x_pos"] = float(pose[0][0])
        obs["y_pos"] = float(pose[0][1])
        # Theta from quaternion ...
        
        # Cameras
        for name, cam in self.cameras.items():
            rgb = cam.get_rgb()
            if rgb is None:
                error(f"[Aloha] Camera {name} returned no image!")
                continue
            
            # Convert to BGR for compatibility with cv2/existing pipeline if needed, 
            # but existing pipeline seems to just encode to jpg.
            # cv2 uses BGR, isaac returns RGB.
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            obs[name] = bgr
                
        return obs



# --- Host Logic ---

def main():
    # Command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--editor_only", action="store_true", help="Run without loading or starting the simulation (for testing)")
    parser.add_argument("--usd_name", type=str, default="alohamini.usd", help="Name of the USD file to load from assets/")
    
    args = parser.parse_args()

    if args.editor_only:
        log("[HOST] Running in no-sim mode. Will idle until interrupted, without starting simulation or ZMQ sockets.")
        try:
            while simulation_app.is_running():
                start_time = time.perf_counter()
                simulation_app.update()
                
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, (1.0 / FPS) - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            log("[HOST] Keyboard interrupt received. Exiting...")
            simulation_app.close()
        return

    # ZMQ Setup
    context = zmq.Context()
    socket_pub = context.socket(zmq.PUB)
    socket_pub.setsockopt(zmq.CONFLATE, 1)      # TODO: might need to comment out, revisit later
    socket_pub.bind(f"tcp://0.0.0.0:{PORT_OBS}")
    
    socket_sub = context.socket(zmq.PULL)
    socket_sub.setsockopt(zmq.CONFLATE, 1)
    socket_sub.bind(f"tcp://0.0.0.0:{PORT_CMD}")

    log(f"[HOST] Loading simulation with USD asset '{args.usd_name}'...")
    sim = IsaacWorld(f"./assets/{args.usd_name}", "/World/Scene/AlohaMini", verbose=args.verbose)
    log(f"[HOST] Simulator running on ports: OBS={PORT_OBS}, CMD={PORT_CMD}")

    try:
        while simulation_app.is_running():
            sim.isaacWorld.step(render=True)
            
            if not sim.isaacWorld.is_playing():
                log("[HOST] Simulation is paused. Stepping until resumed...")
                while not sim.isaacWorld.is_playing():
                    sim.isaacWorld.step(render=True)
                    elapsed = time.perf_counter() - start_time
                    sleep_time = max(0, (1.0 / FPS) - elapsed)
                    time.sleep(sleep_time)

                log("[HOST] Simulation resumed.")
                continue

            start_time = time.perf_counter()
            
            # 1. Process Commands (non-blocking)
            try:
                msg = socket_sub.recv_string(zmq.NOBLOCK)
                action = json.loads(msg)
                sim.aloha.send_action(action)
                if args.verbose:
                    log(f"[HOST] Processed action command: {action}")
            except zmq.Again:
                pass
            except Exception as e:
                log(f"[HOST] Error receiving command: {e}")

            # 2. Get Observation
            obs = sim.aloha.get_observation()
            
            # 3. Encode Images
            encoded_obs = obs.copy()
            
            # Remove raw image data from encoded_obs to save bandwidth/processing if not needed, 
            # but we need to encode it first.
            
            # Special handling for detections: keep them as object
            # (they are already json serializable)
            
            for cam in sim.aloha.cameras.keys():
                if cam in obs:
                    ret, buffer = cv2.imencode(".jpg", obs[cam], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ret:
                        encoded_obs[cam] = base64.b64encode(buffer).decode("utf-8")
                    else:
                        encoded_obs[cam] = ""
                        
            # 4. Publish
            socket_pub.send_string(json.dumps(encoded_obs))

            # 5. Sleep
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, (1.0 / FPS) - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        log("[HOST] Keyboard interrupt received. Stopping...")
    finally:
        log("[HOST] Shutting down simulation...")
        socket_pub.close()
        socket_sub.close()
        context.term()
        simulation_app.close()
        
if __name__ == "__main__":
    main()