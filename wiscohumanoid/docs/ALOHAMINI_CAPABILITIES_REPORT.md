# AlohaMini Capabilities Report

A concise technical overview of the current system state and future opportunities.

---

## Currently Implemented

AlohaMini is a bimanual mobile manipulation robot with a mature **teleoperation system** and infrastructure for **imitation learning data collection**. No autonomous policies have been trained yet.

### Teleoperation System

**Multi-Modal Input Integration**
- Leader arms provide joint positions for follower arms (7 DOF each)
- Keyboard controls for mobile base (forward/backward/rotate)
- Voice commands via DashScope ASR for high-level actions (lift height, emergency stop)
- All inputs merge seamlessly with later sources overriding earlier ones

**Network Architecture**
- ZMQ client-server model with separate command (port 5555) and observation (port 5556) channels
- Runs at 30 Hz control frequency with non-blocking sockets
- Observation streaming includes joint states and JPEG-compressed camera feeds
- Supports both LAN and wireless operation

**Safety Features**
- Overcurrent protection with 20-frame debouncing to prevent false triggers
- Watchdog timeout stops base if no command received for >1.5 seconds
- Max relative target caps joint velocity to prevent sudden movements
- Emergency stop via voice command takes priority over all other inputs

### LeKiwi Robot Implementation

**Hardware Control**
- Feetech serial bus servos for arms (7 DOF per arm)
- Lift axis with multi-turn absolute encoder and P-controller
- Differential drive base with velocity control
- Serial communication at 1Mbps baud rate

**Motor Management**
- Position and velocity reading with configurable update rates
- Torque enable/disable for safety
- Calibration routines for lift axis homing
- Present position reading with multi-turn tracking

### Data Collection Infrastructure

**HuggingFace Integration**
- Dataset recording compatible with LeRobot format
- Direct upload to HuggingFace Hub for sharing and training
- Episode-based recording with automatic metadata

**Recording Capabilities**
- Synchronized joint state and action recording
- Camera frame capture (when cameras are enabled)
- Timestamp tracking for all observations
- Configurable episode length and frequency

### Voice Control System

**DashScope Integration**
- Real-time ASR (Automatic Speech Recognition)
- Chinese and English number normalization
- Command parsing with regex patterns

**Supported Commands**
- Lift height control: "raise to 10cm", "lower to 5cm"
- Base movement: "forward 2 seconds", "backward"
- Emergency stop: "stop"
- Sticky targets that persist until reached

### Current Limitations

**Cameras**: Camera code exists but is currently disabled by default in configuration. No vision processing or visual features are used.

**Policies**: No trained policies exist yet. The evaluation script can load and run ACT/Diffusion policies, but no models have been trained on AlohaMini data.

**Planning**: No motion planning, trajectory optimization, or obstacle avoidance. All movement is direct teleoperation or policy replay.

**Perception**: No object detection, depth sensing, or semantic understanding. System is blind except for raw camera feeds.

---

## What Can Be Implemented/Improved

### 1. Enable Vision System

**Camera Configuration**
- Code exists for OpenCV cameras in [`lerobot_alohamini/src/lerobot/robots/alohamini/config_lekiwi.py`](lerobot_alohamini/src/lerobot/robots/alohamini/config_lekiwi.py)
- Currently disabled with `cameras={}` - needs device paths and calibration
- Enable by adding camera configurations (device IDs, resolution, FPS)

**What This Enables**
- Visual observation recording during teleoperation
- Visual features for policy training
- Real-time camera feed streaming to operator
- Foundation for vision-based policies

### 2. Train Vision-Based Policies

**Complete Workflow** (cameras → data → policy → deployment):

1. **Collect Teleoperation Demonstrations**
   - Enable cameras in configuration
   - Use existing `record_bi.py` script to record episodes
   - Collect 50-200 demonstrations of target task
   - Each demo includes: joint states, actions, camera RGB frames, timestamps

2. **Upload to HuggingFace Hub**
   - Use LeRobot's `push_dataset_to_hub` tools in [`lerobot_alohamini/src/lerobot/datasets/`](lerobot_alohamini/src/lerobot/datasets/)
   - Dataset format is already compatible with LeRobot training
   - Include episode metadata and task description

3. **Train Policy with LeRobot**
   - **ACT (Action Chunking Transformer)**: Good for precise manipulation
     - Encoder-decoder architecture with visual features
     - Predicts action chunks (sequences of actions)
     - Recommended for fine-grained arm control

   - **Diffusion Policy**: Better for smooth, compliant motion
     - Denoising diffusion process for action generation
     - More robust to distribution shift
     - Recommended for mobile base + arm coordination

   - Use training scripts in [`lerobot_alohamini/examples/training/`](lerobot_alohamini/examples/training/)
   - Training typically requires 50K-200K gradient steps
   - Monitor validation loss and success rate on held-out episodes

4. **Deploy Trained Model Back to Robot**
   - Use `evaluate_bi.py` to load checkpoint and run policy
   - Policy takes current observations (joint states + camera frames) → outputs actions
   - Execute in closed loop at 10-30 Hz depending on policy
   - Monitor for distribution shift and failure modes

**Policy Training Considerations**
- **Data quality matters more than quantity**: Clean, consistent demos beat noisy large datasets
- **Task definition**: Clearly defined success criteria and termination conditions
- **Sim-to-real gap**: Policies trained on real robot data generalize better
- **Action representation**: Choose between joint positions, velocities, or end-effector poses

### 3. Autonomous Failure Recovery

**Current State**: When policy fails or gets stuck, system requires manual intervention

**Opportunities**
- Anomaly detection to recognize when policy is struggling
- Fallback behaviors (return to home position, request help)
- Hierarchical control: high-level task planner + low-level policy execution
- Human-in-the-loop recovery: operator takes over temporarily then policy resumes

### 4. Advanced Motion Planning

**Path Planning**
- RRT/RRT* for collision-free trajectories
- Integrate with placo kinematics library (already in dependencies)
- Cartesian space planning for end-effector goals

**Obstacle Avoidance**
- Depth camera integration for 3D scene understanding
- Dynamic replanning when environment changes
- Safety constraints in optimization

### 5. Multi-Robot Coordination

**Potential Applications**
- Multiple AlohaMini robots collaborating on large objects
- Leader-follower formations for navigation
- Task allocation and coordination protocols

**Requirements**
- Shared world model and communication protocol
- Distributed planning and control
- Conflict resolution for shared resources

### 6. Sim-to-Real Transfer

**Simulation Environment**
- MuJoCo models exist in `gym-aloha` package (in dependencies)
- Train policies in simulation with domain randomization
- Fine-tune on real robot with minimal real-world data

**Benefits**
- Faster iteration during policy development
- Safer exploration of failure modes
- Cheaper data collection at scale

### 7. Additional Improvements

**Teleoperation Enhancements**
- Haptic feedback from follower to leader arms
- Motion smoothing and acceleration limiting
- Latency prediction and compensation
- Customizable watchdog timeouts per task

**Perception Additions**
- Object detection and tracking
- Depth estimation from stereo or monocular
- Semantic segmentation for scene understanding
- Tactile sensing for grasp quality

**Control Improvements**
- Impedance control for compliant manipulation
- Admittance control for force-sensitive tasks
- Cartesian velocity control for end-effector
- Joint torque control for delicate operations

---

## Getting Started with Policy Training

For teams ready to start training policies:

1. **Enable cameras** in [lerobot_alohamini/src/lerobot/robots/alohamini/config_lekiwi.py](lerobot_alohamini/src/lerobot/robots/alohamini/config_lekiwi.py)
2. **Collect 50-100 demos** of your target task using `teleoperate_bi_voice.py` with recording
3. **Upload dataset** to HuggingFace Hub using LeRobot tools
4. **Train ACT or Diffusion policy** on your dataset
5. **Evaluate** on robot and iterate

See [LeRobot documentation](https://huggingface.co/docs/lerobot) for detailed training guides and hyperparameter recommendations.
