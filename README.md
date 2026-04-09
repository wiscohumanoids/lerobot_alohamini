# WiscoHumanoids AlohaMini

***IMPORTANT:*** FORMERLY, we were using a Docker image in the name of cross-platform compatibility, but this ridiculously overcomplicated communication with the robot, leader arms (teleop), and Huggingface. Please do a fresh re-install of this repository according to [these instructions](#installation).


## Directory Structure

Overall structure is as follows:

```
lerobot_alohamini/
├── calibration/         # robot & leader calibration files (mounted accordingly in container)
├── datasets/            # storage (mounted accordingly in container for persistence)
├── docs/                # mostly outdated extra docs
├── scripts/             # scripts for working with the alohamini in the container
├── simulation/          # scripts & assets for simulation
└── src/
    ├── utils/           # WA-AM-specific source for teleop, data, server, etc.
    ├── pi0.5/           # WA-AM-specific source for teleop, data, server, etc.
    └── lerobot/         # modified LeRobot source
```

## Installation

First, clone this repo into a folder of your choosing:

```bash
git clone https://github.com/wiscohumanoids/lerobot_alohamini.git
cd lerobot_alohamini
```

*If using WINDOWS*, ensure you have WSL2 Ubuntu 24.04 LTS installed and enter it by running `wsl`. Now, create & source the virtual environment we'll use for dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install all dependencies as described in `pyproject.toml` (plus a few extra, this might take a while...):

```bash
pip install -e .[all]
pip install feetech-servo-sdk
pip install zmq
```

Run the following calibration setup script with:

```bash
./scripts/calib_sync.sh
```

You should now be all set!

## Onboarding

In order to get up to speed, please read any & all research material available in our [shared Google Drive](https://drive.google.com/drive/folders/1nRqpTXZkhCgcrnd-XB8d54YkAPep3cdM). Separately, we've written guides acessible in the `docs/` subfolder, if desired.

* **Teleoperation:** [`docs/TELEOP.md`]
* **Recording data:** [`docs/RECORD.md`]
* **Training policies:** [`docs/TRAIN.md`]
* **π0.5:** [`docs/PI0.5.md`]


## Important Notes

* **Connecting to the Jetson:** working on the Alohamini almost always requires that we start a specific host process (see guides). To do so, we connect to Alohamini's Jetson Orin Nano via SSH, typically wirelessly over Eduroam. You should be able to find the credentials in our Discord, but if not, please ask! We *highly recommend* using extension(s) in VS Code (or your editor of choice) to set up a one-click profile for reuse.
* **Shared Google Workspace:** our shared Google Drive can be accessed [here](https://drive.google.com/drive/folders/1H31gOL2ykyDkuMSXI1YodYHdKjC_M-xH). (Save the link!)


## Troubleshooting

See the following docs:
* [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md#troubleshooting) - Hardware issues
* [`docs/STARTUP_GUIDE.md`](docs/STARTUP_GUIDE.md#troubleshooting) - Docker issues
