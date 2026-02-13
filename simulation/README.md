# AlohaMini Sim Integration

Please ensure you've thoroughly read all the documentation & completed the setup guides available in `wiscohumanoid/docs` before beginning with sim.


## Note

As of **2/16**, sim is a WIP and this is very much incomplete. Remote simulation is a top priority for the team we are actively pursuing. In the meantime, we use a workaround for IsaacSim setup detailed below. We're not entirely locked in to IsaacSim just yet -- if you have experience with or would prefer another platform, please let us know! We were looking into MuJoCo, Gazebo, and so on...


## Workflow

1. On compatible machines (***temporarily***, see below), source the virtual environment and launch simulation as follows from the **host machine**:
    ```bash
    .\.venv-tmp\Scripts\activate
    python sim_env.py
    ```
    *This handles creation of & interaction with the IsaacSim environment, and acts like a server that communicates with later scripts*.

2. In the Docker container, you have a few basic options:
    
    - **Standalone teleop:** launch with `python keyboard_teleop.py --hide_state` for control with keyboard
    - **Record a dataset:** NOT TESTED
    - **Replay a dataset:** NOT TESTED
    - **Train a policy:** NOT TESTED
    - **Evaluate a policy**: NOT TESTED

    Feel free to look these up if you're unfamiliar with what this all is. Helpful references are provided [here](#references).



## Setup
*Again, our immediate goal is to make simulation entirely remote, and thus GPU-agnostic.*

1. Consult [NVIDIA's website](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) to verify that your system is compatbile with IsaacSim.
    
2. Ensure Python 3.11 is installed on your local machine.

3. Navigate to `simulation`, and create a virtual environment as follows (you may need to install the package `venv` first):
    ```bash
    python -m venv .venv-tmp
    ```
    This virtual environment isolates IsaacSim dependencies and should only be run from the **host**, *not* the container!

4. Source the virtual environment with `.\.venv-tmp\Scripts\activate`.
5. Download & install all necessary dependencies using the following command:
    ```bash
    pip install -r placeholder/requirements.txt
    ```
    We will be using the IsaacSim Python API to control simulation.

6. To **begin simulation** at any time, run `python sim_host.py`. This will only work from the host machine (***not*** in Docker!!), and serves as the ZMQ host which communicates with the LeRobot-based teleop/recorder scripts.


## References

WIP