# Teleoperation

Teleoperation is how we manually guide the robot's actions through a pair of physical leader arms (for arm movement) and keyboard input (for movement, rotation, vertical lift, so on).

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
   * **Linux & MacOS:** shouldn't have any platform-specific steps.
4. On your **local machine**, run `./scripts/teleop.sh` (use `--help` to see all options).
