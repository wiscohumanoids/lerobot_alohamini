# WiscoHumanoids AlohaMini





## Directory Structure

Since this is a fork of a fork of an open-source repo, it's a fair bit messy. Overall structure for our purposes is as follows:

```
lerobot_alohamini/
├── src/
|   ├── utils/           # WA-AM-specific source for teleop, data, server, etc.
|   └── lerobot/         # modified LeRobot source
├── simulation/          # scripts & assets for simulation
├── ...
└── wiscohumanoid/       # our extra scripts, setup, info, etc.
    ├── docker/          # docker container setup
    ├── scripts/         # scripts for simple operation
    └── docs/            # documentation & guides
```

## Setup & Onboarding

1. First, build the Docker image at `sudo ./docker/build.sh`. This entire environment - whether it's running on your laptop, the Jetson, or a remote server, will use the associated container.
2. Start the container through `sudo ./docker/run.sh`, or join (create a new shell for) an existing instance with `sudo ./docker/join.sh`. 
3. **Explore documentation & codebase:**
   * Familiarize yourself with the overall robot by rading through [`setup_docs/STARTUP_GUIDE.md`](setup_docs/STARTUP_GUIDE.md), [`setup_docs/ALOHAMINI_ARCHITECTURE.md`](setup_docs/ALOHAMINI_ARCHITECTURE.md), and [`setup_docs/ALOHAMINI_CAPABILITIES_REPORT.md`](setup_docs/ALOHAMINI_CAPABILITIES_REPORT.md)
      * *Note: all documentation assumes you're working from the **container**, and will NOT function easily otherwise!*
   * If helpful, try this interactive tutorial at [`setup_docs/AlohaMini_Walkthrough.ipynb`](setup_docs/AlohaMini_Walkthrough.ipynb)
   Interactive tutorial walking through the codebase
   * Read any & all research material available in our [shared Google Drive](https://drive.google.com/drive/folders/1nRqpTXZkhCgcrnd-XB8d54YkAPep3cdM)

## Troubleshooting

See the following docs:
* [`setup_docs/STARTUP_GUIDE.md`](setup_docs/STARTUP_GUIDE.md#troubleshooting) - Docker issues
* [`setup_docs/DEPLOYMENT_GUIDE.md`](setup_docs/DEPLOYMENT_GUIDE.md#troubleshooting) - Hardware issues


## AlohaMini Startup

To operate the physical robot, do the following **IN THE CONTAINER ON THE JETSON**:
1. Start the host server using `./scripts/host.sh`

*Reminder: **YOU** are the **CLIENT** and the **ROBOT** is the **HOST.***

## Teleop Setup

![alohamini teleop setup](./media/teleop_setup.jpg)

Step by step:
1. Gather the **leader arms**, **two clamps *each***, two USB-C -> USB-C cables, and two **5V adapters** (one per arm).
2. Set up according to the image above, noting that the leader arm with the **green** label is on the **left** and that with the **red** label is on the **right**. Connect the SCServo controller board on the back of each leader arm to 5V power, and separately through USB to your local machine.
3. Expose leader arm USB ports to Docker (varies by device, necessary since docker tries to isolate from the system):
   * **Windows:** if using Docker w/ WSL2 (recommended), install some tool such as [usbipd](https://github.com/dorssel/usbipd-win) that can attach COM ports to WSL. In our experience, the devices typically appear as `/dev/ttyACM0` and `/dev/ttyACM1`.
   * **Linux & MacOS:** should appear normally and be detected...? TODO: verify this
4. Enter the Docker container on your **local machine**, and *inside* run `./scripts/teleop.sh` (use `--help` to see all options).
