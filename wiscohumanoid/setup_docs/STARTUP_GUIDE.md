# AlohaMini Startup Guide

This guide helps new team members get up and running with the AlohaMini codebase using Docker. No physical hardware required for exploration.

---

## Prerequisites

### Required Software

1. **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
   - Windows/Mac: Download from [docker.com/get-started](https://www.docker.com/get-started)
   - Linux: Follow [docs.docker.com/engine/install](https://docs.docker.com/engine/install/)
   - Verify installation:
     ```bash
     docker --version
     # Should output: Docker version 20.x.x or higher
     ```

2. **Git** (for cloning repository)
   - Download from [git-scm.com](https://git-scm.com/)
   - Verify installation:
     ```bash
     git --version
     ```

3. **Minimum System Requirements**
   - 8 GB RAM (16 GB recommended)
   - 20 GB free disk space
   - Modern CPU (multi-core recommended)

---

## Quick Start (5 Steps)

### Step 1: Clone the Repository

```bash
# Clone the AlohaMini repository
git clone https://github.com/wiscohumanoids/lerobot_alohamini.git
cd alohamini

# Checkout to our club's branch
git checkout wiscohumanoids
cd wiscohumanoids
```

### Step 2: Build Docker Image

```bash
# Make build script executable (Linux/Mac only)
chmod +x docker/build.sh docker/run.sh

# Build the Docker image (takes 10-20 minutes on first run)
docker/build.sh
```

**Windows users**: If `docker/build.sh` doesn't work, use:
```powershell
bash docker/build.sh
```

**What's happening**: Docker is creating an Ubuntu 22.04 container with Python 3.10 and all LeRobot dependencies (PyTorch, NumPy, OpenCV, ZMQ, etc.).

### Step 3: Run the Container

**Interactive Terminal** (recommended for exploration):
```bash
# Script is already executable from Step 2

# Run container
docker/run.sh
cd wiscohumanoids
```

**Windows users**:
```powershell
bash docker/run.sh
```

### Step 4: Access Documentation

Once inside the container, explore the documentation:

```bash
# View architecture documentation (Step 1)
cat docs/ALOHAMINI_ARCHITECTURE.md | less
# Press 'q' to exit

# View capabilities report (Step 3)
cat docs/ALOHAMINI_CAPABILITIES_REPORT.md | less

# List all documentation files
ls -lh docs/*.md
```

### Step 5: Run Interactive Notebook

**Start with `docker/run.sh`** (interactive terminal):
```bash
# go into root directory
cd /lerobot_alohamini/wiscohumanoids
# Start Jupyter from inside container
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

# Then open browser to http://localhost:8888
```

---

## Safe Exploration (No Hardware Required)

The Docker container allows you to explore the codebase safely without physical hardware:

### 1. Browse Source Code

```bash
# Navigate to robot implementation
cd /workspace/src/lerobot/robots/alohamini
ls -lh

# View core robot controller
cat lekiwi.py | less

# View configuration
cat config_lekiwi.py
```

### 2. Inspect Motor Configurations

```bash
# Use Python REPL to explore configs
python
```

```python
from lerobot.robots.alohamini.config_lekiwi import LeKiwiConfig

# Load configuration
config = LeKiwiConfig()

# Inspect settings
print(f"Left bus port: {config.left_port}")
print(f"Right bus port: {config.right_port}")
print(f"Cameras configured: {list(config.cameras.keys())}")  # Will be empty (disabled by default)

exit()
```

### 3. Run Mock Examples

```bash
# Navigate to examples
cd /workspace/examples/alohamini

# List available examples
ls -lh

# View teleoperation code (don't run without hardware!)
cat teleoperate_bi_voice.py | less

# View voice control implementation
cat voice_exec.py | less
```

### 4. Explore Dataset Tools

```bash
# Navigate to dataset examples
cd /workspace/examples/dataset

# View dataset loading example
cat load_lerobot_dataset.py

# Inspect dataset utilities
cat use_dataset_tools.py
```

### 5. Check Available Scripts

```bash
# List all lerobot CLI scripts
ls -lh /workspace/src/lerobot/scripts/

# View help for dataset visualization
python -m lerobot.scripts.lerobot_dataset_viz --help

# View help for motor calibration
python -m lerobot.scripts.lerobot_calibrate --help
```

---

## Accessing Previous Documentation

All documentation from Steps 1-3 is available in the container:

| Document | Description | How to View |
|----------|-------------|-------------|
| `docs/ALOHAMINI_ARCHITECTURE.md` | Complete system architecture (Step 1) | `cat docs/ALOHAMINI_ARCHITECTURE.md \| less` |
| `docs/AlohaMini_Walkthrough.ipynb` | Interactive exploration notebook (Step 2) | Open in Jupyter at `http://localhost:8888` |
| `docs/ALOHAMINI_CAPABILITIES_REPORT.md` | Capabilities analysis & roadmap (Step 3) | `cat docs/ALOHAMINI_CAPABILITIES_REPORT.md \| less` |

**Tip**: Use `less` to read long files (press `q` to exit, arrow keys to scroll)

---

## Common Tasks

### View Git Status
```bash
git status
git log --oneline -10  # View last 10 commits
```

### Search for Code Patterns
```bash
# Find all motor configuration files
find . -name "*config*.py" -type f

# Search for voice control references
grep -r "DashScope" --include="*.py"

# Find camera initialization code
grep -r "OpenCVCamera" --include="*.py"
```

### Install Additional Python Packages (if needed)
```bash
pip install <package-name>
# Example: pip install matplotlib==3.8.0
```

### Exit Container
```bash
exit
# Container will be automatically removed (--rm flag)
```

### Stop Jupyter Server
Press `Ctrl+C` twice in the terminal where Jupyter is running.

---

## Understanding Container Behavior

### What's Mounted
- **Source code**: Your local repository is mounted to `/workspace` in the container
- **Changes persist**: Any file edits in the container will reflect on your host machine
- **Logs/outputs**: Files created in `/workspace` will persist after container exits

### Network Ports
| Port | Purpose | Access |
|------|---------|--------|
| 8888 | Jupyter Notebook/Lab | `http://localhost:8888` |
| 5555 | ZMQ command port | For robot networking |
| 5556 | ZMQ observation port | For robot networking |

### Device Passthrough (Auto-detected)
If you connect physical hardware (motors, cameras), the `docker/run.sh` script automatically detects and mounts:
- `/dev/ttyUSB*` → Motor controllers
- `/dev/video*` → Cameras
- `/dev/snd` → Audio devices

**Note**: For full hardware deployment, see `docs/DEPLOYMENT_GUIDE.md`

---

## Troubleshooting

### Build Fails with "No space left on device"
```bash
# Clean up Docker
docker system prune -a
docker volume prune

# Then rebuild
docker/build.sh
```

### Permission Denied on `docker/build.sh`
```bash
# Make executable (Linux/Mac)
chmod +x docker/build.sh docker/run.sh

# Or use bash explicitly
bash docker/build.sh
```

### Line Ending Issues on Windows/WSL
**Note**: This repository includes a `.gitattributes` file that automatically handles line endings. Shell scripts (`.sh`) and Dockerfiles are always checked out with Unix (LF) line endings, even on Windows.

If you cloned the repository before this fix was added, you may see errors like:
```
/bin/bash^M: bad interpreter
```

To fix existing files:
```bash
# If you have dos2unix installed:
dos2unix docker/build.sh docker/run.sh

# Or use git to re-checkout with correct line endings:
git rm --cached -r .
git reset --hard
```

New clones after this fix will work automatically without any manual conversion.

### "Image not found" when running `docker/run.sh`
```bash
# Ensure image was built successfully
docker images | grep alohamini

# If not found, rebuild
docker/build.sh
```

### Jupyter Won't Open
```bash
# Check if port 8888 is already in use
# Linux/Mac:
lsof -i :8888

# Windows:
netstat -ano | findstr :8888

# Kill the process or use a different port:
docker run ... -p 8889:8888 ...  # Use port 8889 instead
```

### Slow Container Performance
```bash
# Check Docker resource allocation
# Docker Desktop → Settings → Resources
# Increase CPU cores and memory allocation

# Recommended: 4+ CPU cores, 8+ GB RAM
```

### Cannot Access Container Network from Host
```bash
# Verify ports are exposed
docker ps
# Should show "0.0.0.0:8888->8888/tcp"

# Check firewall settings (may block Docker ports)
```

---

## Next Steps

After exploring the codebase in Docker:

1. **Review Architecture** (Step 1): Read `docs/ALOHAMINI_ARCHITECTURE.md` to understand system design
2. **Complete Walkthrough** (Step 2): Run `docs/AlohaMini_Walkthrough.ipynb` interactively
3. **Study Capabilities** (Step 3): Read `docs/ALOHAMINI_CAPABILITIES_REPORT.md` for roadmap
4. **Deploy to Hardware**: Follow `docs/DEPLOYMENT_GUIDE.md` for Raspberry Pi/Jetson setup

---

## Useful Commands Reference

```bash
# Container Management
docker/build.sh                     # Build image
docker/run.sh                       # Run interactive shell
docker ps                           # List running containers
docker images                       # List images
docker system prune -a              # Clean up Docker

# Inside Container
cd /workspace                       # Go to repository root
ls -lh                              # List files
cat docs/FILE.md | less             # Read documentation
python -m lerobot.scripts.<SCRIPT>  # Run LeRobot scripts
jupyter notebook                    # Start Jupyter
exit                                # Exit container

# Git
git status                          # Check changes
git log --oneline -10              # View commits
git diff                            # View modifications

# File Search
find . -name "*.py" -type f         # Find Python files
grep -r "pattern" --include="*.py"  # Search in Python files
```

---

## Getting Help

- **LeRobot Documentation**: [huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot)
- **Docker Documentation**: [docs.docker.com](https://docs.docker.com/)
- **Report Issues**: Check repository's issue tracker
- **Team Questions**: Ask in your team's communication channel

---

**You're all set!** 🚀

Start exploring with `docker/build.sh` followed by `docker/run.sh` to dive into the AlohaMini walkthrough.
