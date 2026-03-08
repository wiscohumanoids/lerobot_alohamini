import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


JOINT_NAMES = [
    "root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint", "vertical_move",
    "arm_left_shoulder_pan", "arm_left_shoulder_lift", "arm_left_elbow_flex",
    "arm_left_wrist_flex", "arm_left_wrist_roll", "arm_left_gripper",
    "arm_right_shoulder_pan", "arm_right_shoulder_lift", "arm_right_elbow_flex",
    "arm_right_wrist_flex", "arm_right_wrist_roll", "arm_right_gripper",
]

def get_default_joint_pos():
    pos = {}
    for name in JOINT_NAMES:
        pos[name] = 0.0
    pos["vertical_move"] = 300.0
    return pos



@configclass
class AlohaMiniActionManagerCfg:
    """Action configuration mapping raw joint inputs to the robot."""
    
    # -- BASE: Velocity Control
    # (Assuming you have a mobile base with left/right wheels or omni-wheels)
    virtual_base = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"], # Update with your exact URDF joint names
        scale=1.0, # Multiplier for your keyboard input
    )

    vertical_move = mdp.JointPositionActionCfg( #lift axis
        asset_name="robot",
        joint_names=["vertical_move"],
        scale=0.1, # Keep this small! A full 1.0 step on a keyboard press will violently snap the arm.
        use_default_offset=True, # Actions act as deltas (+/-) from the robot's default resting pose
    )

    # -- LEFT ARM: Direct Joint Position Control
    left_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "arm_left_shoulder_pan", "arm_left_shoulder_lift", "arm_left_elbow_flex",
            "arm_left_wrist_flex", "arm_left_wrist_roll", "arm_left_gripper",
        ],
        scale=0.1, # Keep this small! A full 1.0 step on a keyboard press will violently snap the arm.
        use_default_offset=True, # Actions act as deltas (+/-) from the robot's default resting pose
    )
    
    # -- RIGHT ARM: Direct Joint Position Control
    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "arm_right_shoulder_pan", "arm_right_shoulder_lift", "arm_right_elbow_flex",
            "arm_right_wrist_flex", "arm_right_wrist_roll", "arm_right_gripper",
        ],
        scale=0.1,
        use_default_offset=True,
    )


@configclass
class AlohaMiniSceneCfg(InteractiveSceneCfg):
    """Configuration for the ALOHA Mini workspace."""


    # 1. Add a basic ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg()
    )

    # 2. Load the ALOHA Mini robot (Replace with your actual USD path)
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Aloha",
        spawn=UsdFileCfg(
            usd_path="./../assets/test.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Define safe resting joint angles for all motors
            joint_pos=get_default_joint_pos()
            
        ),
        actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
            damping={"slider_to_cart": 10.0, "cart_to_pole": 0.0},
        ),
        },
    )


    camera_front = TiledCameraCfg(
        # 1. Point this exactly to the path of the camera inside your USD
        # The `/World/envs/env_.*/` ensures it finds the camera in every parallel environment
        prim_path="/World/envs/env_.*/Aloha/base_link/camera_front", 
        
        # 2. THE MAGIC KEY: Tell Isaac Lab NOT to create a new camera
        spawn=None, 
        
        # 3. Define the output format you want to extract from this camera
        width=640,
        height=480,
        data_types=["rgb"],
        update_period=0.0,
    )

    camera_top = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Aloha/base_link/camera_top", 
        spawn=None, 
        width=640,
        height=480,
        data_types=["rgb"],
        update_period=0.0,
    )

    camera_back = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Aloha/base_link/camera_back", 
        spawn=None, 
        width=640,
        height=480,
        data_types=["rgb"],
        update_period=0.0,
    )

    camera_left = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Aloha/left_Wrist_Pitch_Roll/camera_left", 
        spawn=None, 
        width=640,
        height=480,
        data_types=["rgb"],
        update_period=0.0,
    )

    camera_right = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Aloha/right_Wrist_Pitch_Roll/camera_right", 
        spawn=None, 
        width=640,
        height=480,
        data_types=["rgb"],
        update_period=0.0,
    )

    # 3. Add objects to manipulate
    # ... (e.g., a cube, a sponge, or a table)


@configclass
class AlohaMiniObservationManagerCfg:
    """Observation configuration tailored for an ACT Policy."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        # ==========================================
        # 1. PROPRIOCEPTION (Robot's internal state)
        # ==========================================
        # ACT needs to know where its joints currently are.
        # This will grab the positions of the base, arms, and grippers.
        joint_pos = ObsTerm(
            func=mdp.joint_pos, # Note: using absolute joint positions, not relative, is standard for ACT
            params={"asset_cfg_name": "robot"}
        )
        
        # (Optional) ACT sometimes uses joint velocities, though it's less critical than position.
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg_name": "robot"}
        )

        # ==========================================
        # 2. VISUALS (Tiled Camera Feeds)
        # ==========================================
        # ACT relies heavily on multi-view RGB data. 
        # Standard ALOHA setups use 3 to 4 cameras.
        
        # Global / Top-down view
        image_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg_name": "top_camera", "data_type": "rgb"}
        )
        
        # Left wrist view
        image_left_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg_name": "left_wrist_camera", "data_type": "rgb"}
        )
        
        # Right wrist view
        image_right_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg_name": "right_wrist_camera", "data_type": "rgb"}
        )

    # Register the group so the environment computes it every step
    policy: PolicyCfg = PolicyCfg()