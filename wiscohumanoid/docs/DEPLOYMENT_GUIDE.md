# AlohaMini Hardware Deployment Guide

This guide covers deploying AlohaMini to physical hardware (Raspberry Pi 4/5 or NVIDIA Jetson) with proper networking, SSH access, and device configuration.

---

## Hardware Requirements

### Supported Platforms

| Platform | CPU | RAM | Storage | Notes |
|----------|-----|-----|---------|-------|
| **Raspberry Pi 5** | ARM Cortex-A76 (4 cores) | 8 GB | 64 GB+ SD card | ✅ Recommended |
| **Raspberry Pi 4** | ARM Cortex-A72 (4 cores) | 4-8 GB | 64 GB+ SD card | ✅ Supported |
| **NVIDIA Jetson Nano** | ARM Cortex-A57 (4 cores) + 128-core GPU | 4 GB | 64 GB+ SD card | ✅ Supported (GPU for inference) |
| **NVIDIA Jetson Orin Nano** | ARM Cortex-A78AE (6 cores) + 1024-core GPU | 8 GB | 128 GB+ NVMe | ✅ Best performance |

### Peripherals Required

- **Motor Controllers**: 2x USB-to-serial adapters for Feetech motor buses
  - Left bus: `/dev/ttyUSB0` (or `/dev/am_arm_follower_left`)
  - Right bus: `/dev/ttyUSB1` (or `/dev/am_arm_follower_right`)

- **Cameras** (optional, disabled by default):
  - 5x USB webcams or RealSense cameras
  - See `config_lekiwi.py` lines 23-40 to enable

- **Network**: Ethernet or WiFi connection to LAN

- **Power Supply**: 5V 3A+ (Pi), 12V+ (Jetson)

---

## OS Installation & Initial Setup

### Raspberry Pi Setup

1. **Flash OS (Ubuntu 22.04 LTS)**
   ```bash
   # Use Raspberry Pi Imager: https://www.raspberrypi.com/software/
   # Select: Ubuntu Server 22.04 LTS (64-bit)
   # Configure WiFi/hostname during imaging (optional)
   ```

2. **Initial Boot**
   - Insert SD card, connect Ethernet/HDMI/keyboard
   - Default credentials: `ubuntu` / `ubuntu` (will prompt password change)

3. **Update System**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo reboot
   ```

### Jetson Setup

1. **Flash JetPack**
   - Use NVIDIA SDK Manager: [developer.nvidia.com/sdk-manager](https://developer.nvidia.com/sdk-manager)
   - Install JetPack 5.1+ (includes Ubuntu 20.04/22.04)

2. **Update System**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo reboot
   ```

---

## Network Configuration

### Option A: Static IP (Recommended for Production)

1. **Find Current IP**
   ```bash
   ip addr show
   # Note your interface name (e.g., eth0, wlan0)
   ```

2. **Configure Static IP (Ubuntu 22.04 Netplan)**
   ```bash
   sudo nano /etc/netplan/50-cloud-init.yaml
   ```

   ```yaml
   network:
     version: 2
     ethernets:
       eth0:  # Change to your interface name
         dhcp4: false
         addresses:
           - 192.168.1.100/24  # Set your desired static IP
         routes:
           - to: default
             via: 192.168.1.1  # Your router IP
         nameservers:
           addresses: [8.8.8.8, 8.8.4.4]
   ```

3. **Apply Configuration**
   ```bash
   sudo netplan apply
   ip addr show eth0  # Verify new IP
   ```

### Option B: Find Dynamic IP

```bash
# On robot
hostname -I

# Or from laptop (if mDNS/Avahi enabled)
ping ubuntu.local
```

---

## SSH Configuration

### Enable SSH (Usually enabled by default on Ubuntu)

```bash
# Check SSH status
sudo systemctl status ssh

# If not running, enable it
sudo systemctl enable ssh
sudo systemctl start ssh
```

### Connect from Laptop

```bash
# Replace with your robot's IP
ssh ubuntu@192.168.1.100

# Or using hostname (if mDNS works)
ssh ubuntu@ubuntu.local
```

### SSH Key Authentication (Recommended)

**On your laptop**:
```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key to robot
ssh-copy-id ubuntu@192.168.1.100

# Now you can SSH without password
ssh ubuntu@192.168.1.100
```

---

## Installing Dependencies on Robot

### Option 1: Docker Installation (Recommended)

**Pros**: Isolated environment, easy updates, same as development
**Cons**: Slightly higher overhead

1. **Install Docker**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Add user to docker group (no sudo needed)
   sudo usermod -aG docker $USER
   newgrp docker

   # Verify
   docker --version
   ```

2. **Clone Repository**
   ```bash
   cd ~
   git clone https://github.com/your-org/lerobot_alohamini.git
   cd lerobot_alohamini
   ```

3. **Build Image**
   ```bash
   chmod +x docker/build.sh docker/run.sh
   docker/build.sh
   # Takes 20-40 minutes on Pi, 10-20 minutes on Jetson
   ```

### Option 2: Native Installation (Direct on Host)

**Pros**: Lower latency, direct hardware access
**Cons**: Dependency conflicts possible

1. **Install System Dependencies**
   ```bash
   sudo apt update
   sudo apt install -y \
       build-essential \
       cmake \
       git \
       python3.10 \
       python3.10-dev \
       python3-pip \
       libzmq3-dev \
       libopencv-dev \
       libgl1-mesa-glx \
       portaudio19-dev \
       libusb-1.0-0-dev \
       libudev-dev \
       udev
   ```

2. **Clone Repository**
   ```bash
   cd ~
   git clone https://github.com/your-org/lerobot_alohamini.git
   cd lerobot_alohamini
   ```

3. **Install Python Dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements-ubuntu.txt
   ```

---

## USB Device Passthrough & Permissions

### 1. Identify Motor Controllers

```bash
# List USB serial devices
ls -l /dev/ttyUSB*

# Should show:
# /dev/ttyUSB0  (left bus: arm + base + lift)
# /dev/ttyUSB1  (right bus: arm only)
```

### 2. Create Udev Rules (Persistent Device Names)

```bash
# Find device serial numbers
udevadm info --name=/dev/ttyUSB0 | grep SERIAL

# Create udev rule
sudo nano /etc/udev/rules.d/99-alohamini.rules
```

Add:
```udev
# AlohaMini Motor Controllers
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", ATTRS{serial}=="SERIAL_LEFT", SYMLINK+="am_arm_follower_left", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", ATTRS{serial}=="SERIAL_RIGHT", SYMLINK+="am_arm_follower_right", MODE="0666"

# Cameras (if enabled)
SUBSYSTEM=="video4linux", ATTRS{serial}=="CAM_HEAD_TOP_SERIAL", SYMLINK+="am_camera_head_top", MODE="0666"
SUBSYSTEM=="video4linux", ATTRS{serial}=="CAM_HEAD_BACK_SERIAL", SYMLINK+="am_camera_head_back", MODE="0666"
SUBSYSTEM=="video4linux", ATTRS{serial}=="CAM_HEAD_FRONT_SERIAL", SYMLINK+="am_camera_head_front", MODE="0666"
SUBSYSTEM=="video4linux", ATTRS{serial}=="CAM_WRIST_LEFT_SERIAL", SYMLINK+="am_camera_wrist_left", MODE="0666"
SUBSYSTEM=="video4linux", ATTRS{serial}=="CAM_WRIST_RIGHT_SERIAL", SYMLINK+="am_camera_wrist_right", MODE="0666"
```

Replace `SERIAL_LEFT`, `SERIAL_RIGHT`, and `CAM_*_SERIAL` with actual serial numbers.

**Reload udev rules**:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**Verify**:
```bash
ls -l /dev/am_*
# Should show symbolic links to /dev/ttyUSB* and /dev/video*
```

### 3. Add User to Dialout Group (Serial Access)

```bash
sudo usermod -aG dialout $USER
newgrp dialout  # Apply immediately

# Verify
groups
# Should include "dialout"
```

### 4. Docker Device Passthrough

**In `docker/run.sh`**, devices are automatically passed through:
```bash
--device=/dev/ttyUSB0 \
--device=/dev/ttyUSB1 \
--device=/dev/video0 \
# etc.
```

**Or use `--privileged` for full device access** (less secure):
```bash
docker run --privileged ...
```

---

## Running AlohaMini on Robot

### Architecture Overview

AlohaMini uses a **client-server architecture**:
- **Server (Host)**: Runs on robot hardware, controls motors/cameras
- **Client**: Runs on laptop, sends commands, receives observations

```
Laptop (Client)           Network (ZMQ)        Robot (Server)
-----------------         ---------------      ----------------
lekiwi_client.py    -->   Port 5555 (CMD)  --> lekiwi_host.py
                    <--   Port 5556 (OBS)  <--   |
                                                  v
                                          Motors + Cameras
```

### Step 1: Start Server on Robot

**Using Docker**:
```bash
# SSH into robot
ssh ubuntu@192.168.1.100

# Navigate to repo
cd ~/lerobot_alohamini

# Run container with device passthrough
docker/run.sh

# Inside container, start host server
python lerobot_alohamini/examples/alohamini/lekiwi_host.py
```

**Using Native Installation**:
```bash
# SSH into robot
ssh ubuntu@192.168.1.100

# Navigate to repo
cd ~/lerobot_alohamini

# Start host server
python lerobot_alohamini/examples/alohamini/lekiwi_host.py
```

**Expected Output**:
```
[INFO] LeKiwi Host starting...
[INFO] Binding ZMQ sockets on ports 5555, 5556
[INFO] Initializing motors on /dev/am_arm_follower_left
[INFO] Initializing motors on /dev/am_arm_follower_right
[INFO] Motors ready. Waiting for commands...
[INFO] Watchdog timeout: 1500 ms
```

**Troubleshooting**:
- **"Permission denied" on `/dev/ttyUSB*`**: Run `sudo usermod -aG dialout $USER`, log out and back in
- **"Device not found"**: Check `ls /dev/ttyUSB*`, verify udev rules, reconnect USB cables
- **"Port already in use"**: Kill existing process: `sudo lsof -i :5555`, `kill -9 <PID>`

### Step 2: Connect from Laptop (Client)

**On your laptop** (NOT in Docker, unless you configure network bridge):

```bash
# Navigate to repo
cd /path/to/lerobot_alohamini

# Activate Python environment (if using venv)
source venv/bin/activate

# Run client
python lerobot_alohamini/examples/alohamini/lekiwi_client.py --remote-ip 192.168.1.100
```

**Or use BiSO100 teleoperation**:
```bash
python lerobot_alohamini/examples/alohamini/teleoperate_bi.py --remote-ip 192.168.1.100
```

**Or use voice control**:
```bash
python lerobot_alohamini/examples/alohamini/teleoperate_bi_voice.py --remote-ip 192.168.1.100
```

**Expected Output**:
```
[INFO] Connecting to robot at 192.168.1.100:5555...
[INFO] Connected successfully
[INFO] Teleoperation active. Press 'q' to quit.

Controls:
  w/s - forward/backward
  a/d - rotate left/right
  z/x - strafe left/right
  u/j - lift up/down
  r/f - speed up/down
```

---

## Running Autonomous Policies

### Step 1: Load Pretrained Policy

**On robot** (or run policy server on laptop):

```bash
# Example: Evaluate ACT policy
python lerobot_alohamini/examples/alohamini/evaluate_bi.py \
    --robot-path lekiwi \
    --policy-path path/to/trained_policy \
    --remote-ip 192.168.1.100
```

**Note**: No pretrained policies exist yet. You must:
1. Collect demonstrations using `record_bi.py`
2. Train policy using `examples/training/train_policy.py`
3. Evaluate using `evaluate_bi.py`

### Step 2: Record Demonstrations

```bash
# On laptop (connected to robot server)
python lerobot_alohamini/examples/alohamini/record_bi.py \
    --robot-path lekiwi_client \
    --remote-ip 192.168.1.100 \
    --repo-id your-username/alohamini-dataset \
    --num-episodes 50
```

### Step 3: Train Policy

```bash
# On laptop or Jetson (requires GPU)
python lerobot_alohamini/examples/training/train_policy.py \
    --policy act \
    --dataset-repo-id your-username/alohamini-dataset \
    --output-dir outputs/act_policy \
    --num-epochs 5000
```

### Step 4: Deploy Trained Policy

```bash
# Evaluate on hardware
python lerobot_alohamini/examples/alohamini/evaluate_bi.py \
    --robot-path lekiwi \
    --policy-path outputs/act_policy \
    --remote-ip 192.168.1.100
```

---

## Environment Variables

Set these in `~/.bashrc` or pass via Docker `-e` flag:

```bash
# ZMQ Configuration
export LEKIWI_HOST_PORT_CMD=5555
export LEKIWI_HOST_PORT_OBS=5556

# Watchdog timeout (milliseconds)
export LEKIWI_WATCHDOG_TIMEOUT_MS=1500

# Control loop frequency (Hz)
export LEKIWI_LOOP_FREQ_HZ=30

# Device paths (override defaults)
export LEKIWI_LEFT_PORT=/dev/am_arm_follower_left
export LEKIWI_RIGHT_PORT=/dev/am_arm_follower_right

# Lift axis parameters
export LIFT_AXIS_KP=300
export LIFT_AXIS_MAX_VEL=1000

# Camera enable (set to 1 to enable)
export ENABLE_CAMERAS=0  # Disabled by default

# Apply changes
source ~/.bashrc
```

---

## Performance Tuning

### CPU Governor (Increase Performance)

```bash
# Check current governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make persistent (add to /etc/rc.local)
sudo nano /etc/rc.local
# Add before "exit 0":
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Increase Swap (If RAM < 8 GB)

```bash
# Create 4 GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make persistent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Reduce Latency

```bash
# Disable WiFi power management (if using WiFi)
sudo iw dev wlan0 set power_save off

# Set process priority
sudo nice -n -10 python lerobot_alohamini/examples/alohamini/lekiwi_host.py
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'lerobot'"

**Solution**:
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/workspace:$PYTHONPATH  # Docker
export PYTHONPATH=~/lerobot_alohamini:$PYTHONPATH  # Native

# Or install in editable mode
pip install -e .
```

### Problem: High Latency (>100 ms loop time)

**Causes**:
- Network congestion
- CPU throttling
- Insufficient resources

**Solutions**:
```bash
# Check CPU usage
htop

# Check network latency
ping 192.168.1.100

# Monitor loop timing (add logging in lekiwi_host.py)
# Reduce control frequency if needed (30 Hz → 20 Hz)
```

### Problem: Motors Not Responding

**Check**:
```bash
# Verify USB devices
ls -l /dev/ttyUSB*

# Check permissions
groups  # Should include "dialout"

# Test serial connection
python -m lerobot.scripts.lerobot_find_port
```

**Solution**:
```bash
# Replug USB cables
# Check power supply to motors
# Verify motor IDs match configuration
```

### Problem: Cameras Not Working

**Note**: Cameras are disabled by default in `config_lekiwi.py`.

**Enable cameras**:
```bash
nano lerobot_alohamini/src/lerobot/robots/alohamini/config_lekiwi.py
# Uncomment lines 23-40
```

**Test cameras**:
```bash
python -m lerobot.scripts.lerobot_find_cameras
```

### Problem: Watchdog Timeout

**Cause**: No commands received for >1.5 seconds

**Solution**:
- Check network connection stability
- Increase watchdog timeout in `config_lekiwi.py`
- Verify client is sending commands regularly

### Problem: Overcurrent Protection Triggered

**Cause**: Motor current exceeded threshold

**Check**:
- Motor wiring and connections
- Mechanical obstructions
- Load on arms (too heavy?)

**Adjust threshold** (if safe):
```python
# In lekiwi.py
self.overcurrent_threshold = 1500  # Increase from default (use with caution!)
```

---

## Remote Access (Optional)

### VNC for GUI Access

1. **Install VNC Server on Robot**
   ```bash
   sudo apt install tightvncserver
   vncserver :1
   ```

2. **Connect from Laptop**
   ```bash
   # SSH tunnel
   ssh -L 5901:localhost:5901 ubuntu@192.168.1.100

   # Open VNC client to localhost:5901
   ```

### Tmux for Persistent Sessions

```bash
# Start tmux session
tmux new -s robot

# Run host server
python lerobot_alohamini/examples/alohamini/lekiwi_host.py

# Detach: Ctrl+b, then d

# Reattach later
tmux attach -t robot
```

---

## Production Deployment Checklist

- [ ] OS updated (`sudo apt update && sudo apt upgrade`)
- [ ] Static IP configured
- [ ] SSH key authentication set up
- [ ] Udev rules created for persistent device names
- [ ] User added to `dialout` group
- [ ] Dependencies installed (Docker or native)
- [ ] USB devices verified (`ls /dev/ttyUSB*`, `/dev/video*`)
- [ ] Host server starts without errors
- [ ] Client can connect from laptop
- [ ] Teleoperation tested (keyboard, BiSO100, voice)
- [ ] Watchdog timeout tested (disconnect client)
- [ ] Performance tuning applied (CPU governor, swap)
- [ ] Backup calibration files (`calibration.json`)

---

## Safety Reminders

⚠️ **ALWAYS**:
- Keep emergency stop button accessible
- Test in open space without obstacles
- Monitor motor currents for overheating
- Disable torque when not in use (`disable_torque_on_disconnect: true`)

⚠️ **NEVER**:
- Run at full speed near people or fragile objects
- Bypass overcurrent protection
- Operate with damaged cables or loose connections
- Leave robot unattended during teleoperation

---

## Next Steps

1. **Collect Demonstrations**: Use `record_bi.py` with BiSO100 leader arms
2. **Train Policies**: Use collected data to train ACT or Diffusion policies
3. **Enable Cameras**: Uncomment camera configs, test with `lerobot_find_cameras`
4. **Implement Improvements**: Follow roadmap in `docs/ALOHAMINI_CAPABILITIES_REPORT.md`

---

## Additional Resources

- **LeRobot Documentation**: [huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot)
- **Raspberry Pi Forums**: [forums.raspberrypi.com](https://forums.raspberrypi.com)
- **NVIDIA Jetson Forums**: [forums.developer.nvidia.com](https://forums.developer.nvidia.com)
- **Docker Documentation**: [docs.docker.com](https://docs.docker.com)

---

**Good luck with your deployment!** 🤖
