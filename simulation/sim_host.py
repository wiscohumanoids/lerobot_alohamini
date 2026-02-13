import os
import zmq
import sys
import cv2
import json
import time
import random
import base64
import argparse
import numpy as np
from isaacsim import SimulationApp

def log(msg: str):
    print(f"\033[1;36m {msg}\033[0m")

def error(msg: str):
    print(f"\033[1;31m [ERROR] {msg}\033[0m")

log("[ENV] AlohaMini sim startup")


PORT_OBS = 5556
PORT_CMD = 5555

CAM_RESOLUTION = (640, 480)
FPS = 30

SIM_CONFIG = {
    "width": 100,#1280,
    "height": 50,#720,
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
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.articulations import Articulation, ArticulationSubset  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot  # noqa: E402
from omni.kit.commands import execute # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: F401



class IsaacWorld:
    def __init__(self, urdf_path: str, verbose=False):
        log("Creating IsaacSim world...")
        self.isaacWorld = World(stage_units_in_meters=1.0)
        self.isaacWorld.scene.add_default_ground_plane()

        self.aloha = AlohaSim(self.isaacWorld, urdf_path, verbose=verbose)



class AlohaSim:
    def __init__(self, world, urdf_path: str, verbose=False):
        self.urdf_path = urdf_path
        self.cameras = {}
        self.verbose = verbose

        success, import_config = execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False        # TODO: look into exactly what these are
        import_config.fix_base = False
        import_config.make_default_prim = False
        import_config.create_physics_scene = True

        if not success:
            error("Failed to create import URDF import config! Check installation for proper extensions.")
            sys.exit(1)

        success, self.prim_path = execute(
            "URDFParseAndImportFile",
            urdf_path=self.urdf_path,
            import_config=import_config
        )
        
        if success:
            log(f"[Aloha] Imported URDF to prim path: {self.prim_path}")
            # Assume correct imports and such
            pass
        else:
            error(f"[Aloha] Failed to import URDF from {self.urdf_path}, exiting...")
            sys.exit(1)

        # Add Articulation to robot prim
        log("[Aloha] Setting up sim robot w/ articulation...")

        self.robot = Articulation(prim_path=self.prim_path, name="Aloha")
        world.scene.add(self.robot)
        
        # Add cameras
        self._add_camera("head_front", "/Aloha/base_link/front_cam", np.array([0.2, 0, 0.2]), np.array([0, 0, 0]))
        self._add_camera("head_top", "/Aloha/base_link/top_cam", np.array([0, 0, 0.5]), np.array([0, 90, 0]))

        world.reset()
        self.robot.initialize()
        log("[Aloha] Model setup complete, robot and cameras initialized.")

        # Add dof names mapping
        self.dof_names = self.robot.dof_names   # Note: these names should be different, but the order in which they're listed remains consistent and should be good for now -- TODO: address me in the future
        self.dof_indices = {name: i for i, name in enumerate(self.dof_names)}

        log("[Aloha] DoF/joint map: " + str(self.dof_indices))


    def _add_camera(self, name, prim_path, translation, rotation_euler_deg):
        # Apply domain randomization to camera position
        # Small random perturbation to translation (+- 2cm) and rotation (+- 2 deg)
        # This helps the model become robust to slight calibration errors in the real world
        translation += np.random.uniform(-0.02, 0.02, size=3)
        rotation_euler_deg = rotation_euler_deg.astype(np.float64) + np.random.uniform(-2, 2, size=3)
        
        # rotation in sim is usually quaternion
        # rotation_euler_deg: [x, y, z]
        rot_quat = euler_angles_to_quat(np.radians(rotation_euler_deg))
        
        camera = Camera(
            prim_path=prim_path,
            position=translation,
            frequency=FPS,
            resolution=CAM_RESOLUTION,
            orientation=rot_quat
        )
        camera.initialize()
        self.cameras[name] = camera

        log(f"[Aloha] Adding camera '{name}' at {prim_path} with translation {translation} and rotation (deg) {rotation_euler_deg}")




    def _set_joint_positions(self, joint_positions: dict):
        if not joint_positions:
            if self.verbose:
                log("[Aloha] No joint positions provided in command, skipping joint update.")
            return
        
        # joint_positions: dict of joint_name -> position
        # We need to map this to the robot's dof indices or names
        # For simplicity, we can use the high level Articulation API if names match
        
        # Note: self.robot.set_joint_positions takes numpy array and indices is optional
        # We need to find indices for names
        
        current_joint_pos = self.robot.get_joint_positions()
            
        # Construct target array
        # Start with current to keep uncommanded joints steady
        target_pos = current_joint_pos.copy()
        
        for name, pos in joint_positions.items():
            if name in self.dof_indices:
                idx = self.dof_indices[name]
                target_pos[idx] = pos
                
        controller = self.robot.get_articulation_controller()
        action = ArticulationAction(joint_positions=target_pos)
        controller.apply_action(action)

        if self.verbose:
            log(f"[Aloha] Updated joint target positions with command: {joint_positions}")

    def _set_base_velocity(self, vx, vy, vtheta):
        # Set root velocity
        # chassis frame: x forward, y left
        self.robot.set_linear_velocity(np.array([vx, vy, 0]))
        self.robot.set_angular_velocity(np.array([0, 0, vtheta]))
        if self.verbose:
            log(f"[Aloha] Updated base velocity to vx: {vx}, vy: {vy}, vtheta: {vtheta}")


    def send_action(self, action: dict) -> dict:
        if self.verbose:
            log(f"[Aloha] Received action command: {action}")

        vX = vY = vTheta = 0.0
        joint_cmds = {}

        for k, v in action.items():
            if k == "reset" and v is True:
                # Reset -- might be an informal addon
                log("[HOST] Resetting Aloha environment...")
                self.robot.reset()
                joint_cmds = {name: 0.0 for name in self.dof_names}
                self._set_joint_positions(joint_cmds)
                continue
            if k.endswith(".pos"):
                joint_name = k.replace(".pos", "")
                joint_cmds[joint_name] = v
            elif k in ["x.vel", "y.vel", "theta.vel"]:
                if k == "x.vel":
                    vX = v
                elif k == "y.vel":
                    vY = v
                elif k == "theta.vel":
                    vTheta = v
            elif k == "lift_axis.height_mm":
                joint_cmds["lift_axis"] = v

        if self.verbose:
            log(f"[Aloha] Parsed action - vX: {vX}, vY: {vY}, vTheta: {vTheta}, joint_cmds: {joint_cmds}")

        self._set_base_velocity(vX, vY, vTheta)
        self._set_joint_positions(joint_cmds)

    def get_observation(self) -> dict:
        obs = {}
        
        # Joints
        joint_pos = self.robot.get_joint_positions()
        
        for i, name in enumerate(self.dof_names):
            obs[f"{name}.pos"] = float(joint_pos[i])
            
        # Base (Ground Truth for now)
        pose = self.robot.get_world_pose()
        obs["x_pos"] = float(pose[0][0])
        obs["y_pos"] = float(pose[0][1])
        # Theta from quaternion ...

        if self.verbose:
            log(f"[Aloha] Current pose: x={obs['x_pos']:.2f}, y={obs['y_pos']:.2f}, theta={obs.get('theta', 0):.2f}")
            log(f"[Aloha] Current joint positions: " + ", ".join([f"{name}={obs[f'{name}.pos']:.2f}" for name in self.dof_names]))
        
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
    parser.add_argument("--no_sim", action="store_true", help="Run without loading or starting the simulation (for testing)")
    args = parser.parse_args()

    if args.no_sim:
        log("[HOST] Running in no-sim mode. Will idle until interrupted, without starting simulation or ZMQ sockets.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
                time.sleep(0.1)
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

    sim = IsaacWorld("./model/Aloha/urdf/Aloha - Copy.urdf", args.verbose)
    #sim = IsaacWorld("./model/am/aloha_mini.urdf", args.verbose)
    log(f"[HOST] Simulator running on ports: OBS={PORT_OBS}, CMD={PORT_CMD}")

    try:
        while simulation_app.is_running():
            sim.isaacWorld.step(render=True)
            if not sim.isaacWorld.is_playing():
                if args.verbose:
                    log("[HOST] Simulation paused, skipping step.")
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
            if args.verbose:
                log(f"[HOST] Published observation with keys: {list(encoded_obs.keys())}")

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