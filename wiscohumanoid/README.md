# Wisconsin Humanoids - AlohaMini Setup

This directory contains the Docker environment and documentation for getting started with the AlohaMini robot.

## Directory Structure

```
wiscohumanoid/           # folder for wiscohumanoids-specific work
├── docker/              # docker container setup
│   ├── Dockerfile       # container image definition
│   ├── build.sh         # script to BUILD the docker image (generally a one-time thing)
│   ├── build.sh         # script to JOIN an already running container (create a new shell/terminal in that container instance)
│   └── run.sh           # script to RUN the container (do this on startup)
│

└── docs/                # Documentation & guides
    ├── STARTUP_GUIDE.md              # Start here! Quick setup guide
    ├── ALOHAMINI_ARCHITECTURE.md     # Complete system architecture
    ├── ALOHAMINI_CAPABILITIES_REPORT.md  # Current state & roadmap
    ├── DEPLOYMENT_GUIDE.md           # Hardware deployment guide
    └── AlohaMini_Walkthrough.ipynb   # Interactive tutorial notebook
```

## Quick Start

### 1. Build Docker Image

From the **project root** (`lerobot_alohamini/`):

```bash
wiscohumanoid/docker/build.sh
```

### 2. Run Container

```bash
wiscohumanoid/docker/run.sh
```

### 3. Explore Documentation

Inside the container:

```bash
# View startup guide
cat docs/STARTUP_GUIDE.md | less

# View architecture
cat docs/ALOHAMINI_ARCHITECTURE.md | less

# Run interactive notebook
jupyter notebook docs/AlohaMini_Walkthrough.ipynb
```

## Documentation Guide

### For New Team Members
1. **Start with**: [`docs/STARTUP_GUIDE.md`](docs/STARTUP_GUIDE.md)
   Quick 5-step guide to build Docker and explore the codebase

2. **Then read**: [`docs/ALOHAMINI_ARCHITECTURE.md`](docs/ALOHAMINI_ARCHITECTURE.md)
   Comprehensive technical deep-dive into the system

3. **Run through**: [`docs/AlohaMini_Walkthrough.ipynb`](docs/AlohaMini_Walkthrough.ipynb)
   Interactive tutorial walking through the codebase

4. **Understand capabilities**: [`docs/ALOHAMINI_CAPABILITIES_REPORT.md`](docs/ALOHAMINI_CAPABILITIES_REPORT.md)
   Current implementation status and future opportunities

### For Hardware Deployment
- **Follow**: [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md)
  Complete guide for deploying to Raspberry Pi/Jetson hardware

## Path References

All documentation assumes you're working from the **Docker container** where:
- Project root is mounted at: `/workspace/`
- Documentation is at: `/workspace/docs/`
- Source code is at: `/workspace/lerobot_alohamini/`

## Troubleshooting

See the troubleshooting sections in:
- [`docs/STARTUP_GUIDE.md`](docs/STARTUP_GUIDE.md#troubleshooting) - Docker issues
- [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md#troubleshooting) - Hardware issues

## Additional Resources

- **LeRobot Documentation**: https://huggingface.co/docs/lerobot
- **Docker Documentation**: https://docs.docker.com/
