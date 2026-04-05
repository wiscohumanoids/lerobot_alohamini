# Recording Guide

## Jetson Setup

Connect to the Jetson through the USB-C cable, EtherNet, or Wi-Fi first. After that, on your terminal type `ssh badger@192.168.55.1`. `ssh` is how we're communicating securely with the Jetson. `badger` is the username on the Jetson Nano device. `192.168.55.1` is the remote ip of the Jetson (you still need it even connecting to it physically). (The password is also 'badger')

After that navigate into `lerobot_alohamini/wiscohumanoid` and then run `docker/run.sh` to enable docker. After that, you can start the `lekiwi_host.py` on the Jetson with `python3 -m src.lerobot.robots.alohamini.lekiwi_host`.

A couple things to notice: make sure to configure the video ports correctly in the `config_lekiwi.py` to enable only the video ports you want for recording. Also, make sure to configure the usb ports connected to the motor controller boards correctly.

If you wish to edit the code on the Jetson through VS Code, you would need to install the "Remote - SSH" by Microsoft.

## Macbook Setup

On your Macbook, make sure you have cloned the entire repository in [https://github.com/wiscohumanoids/lerobot_alohamini](https://github.com/wiscohumanoids/lerobot_alohamini) to whatever folder of your choosing. After that, create and enable your virtual environment and install all the necessary python libraries as specified in the `pyproject.toml`. After that, you can start running the recording by running `python3 -m src.lerobot.examples.alohamini.record_bi`. Make sure you have setup the correct usb inputs from the leader arms in the `config_lekiwi.py` file.

### Extra info

You might have to exit the docker container and reenter again on the Jetson if it's not able to identify the ports.

The /dev/videoX numbers are assigned by the Linux kernel based on USB enumeration order — whichever camera the kernel discovers first gets the lowest number. This order depends on:

Which USB port the camera is plugged into
The hub's internal port numbering
Boot timing / driver probe order
That's why they shift around on reboot, and why we switched to /dev/v4l/by-path/ symlinks — those are based on the physical USB port topology (e.g., platform-3610000.usb-usb-0:2.3.1:1.0) which is stable.
