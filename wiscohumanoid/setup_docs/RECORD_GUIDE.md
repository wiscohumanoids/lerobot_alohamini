# Recording Guide

## Jetson Setup

Connect to the Jetson through the USB-C cable, EtherNet, or Wi-Fi first. After that, on your terminal type `ssh badger@192.168.55.1`. `ssh` is how we're communicating securely with the Jetson. `badger` is the username on the Jetson Nano device. `192.168.55.1` is the remote ip of the Jetson (you still need it even connecting to it physically). (The password is also 'badger')

After that navigate into `lerobot_alohamini/wiscohumanoid` and then run `docker/run.sh` to enable docker. After that, you can start the `lekiwi_host.py` on the Jetson with `python3 -m src.lerobot.robots.alohamini.lekiwi_host`. If not use docker, simply go into `lerobot_alohamini` and enable the virtual environment with `source lerobot_venv/bin/activate`.

A couple things to notice: make sure to configure the video ports correctly in the `config_lekiwi.py` to enable only the video ports you want for recording. Also, make sure to configure the usb ports connected to the motor controller boards correctly.

If you wish to edit the code on the Jetson through VS Code, you would need to install the "Remote - SSH" by Microsoft.

## Macbook Setup

On your Macbook, make sure you have cloned the entire repository in [https://github.com/wiscohumanoids/lerobot_alohamini](https://github.com/wiscohumanoids/lerobot_alohamini) to whatever folder of your choosing. After that, create and enable your virtual environment and install all the necessary python libraries as specified in the `pyproject.toml`. After that, you can start running the recording by running `python3 -m src.lerobot.examples.alohamini.record_bi`. Make sure you have setup the correct usb inputs from the leader arms in the `config_lekiwi.py` file.

### Extra info

Common issue: ConnectionError: Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 4 tries. [TxRxResult] There is no status packet!

You might have to exit the docker container and reenter again on the Jetson if it's not able to identify the ports.

The /dev/videoX numbers are assigned by the Linux kernel based on USB enumeration order — whichever camera the kernel discovers first gets the lowest number. This order depends on:

Which USB port the camera is plugged into
The hub's internal port numbering
Boot timing / driver probe order
That's why they shift around on reboot, and why we switched to /dev/v4l/by-path/ symlinks — those are based on the physical USB port topology (e.g., platform-3610000.usb-usb-0:2.3.1:1.0) which is stable.

If it mentions something like the error message below. Simple turn of the battery and turn it on again.

```
RuntimeError: FeetechMotorsBus motor check failed on port '/dev/ttyACM1':

Missing motor IDs:
  - 11 (expected model: 777)

Full expected motor list (id: model_number):
{1: 777, 2: 777, 3: 777, 4: 777, 5: 777, 6: 777, 11: 777}

Full found motor list (id: model_number):
{1: 777, 2: 777, 3: 777, 4: 777, 5: 777, 6: 777}
```

Useful commands:
`ls -la /dev/v4l/by-path/`: check the usb ports for the cameras
`ls -la /dev/v4l/by-path/`: check the usb ports for the motor boards
