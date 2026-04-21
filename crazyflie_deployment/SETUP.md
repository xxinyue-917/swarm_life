# Crazyflie Deployment — Environment Setup

Complete setup steps to reproduce this workspace on a fresh Ubuntu 24.04 machine.

## System requirements

- **Ubuntu 24.04 LTS** (Noble Numbat)
- **ROS2 Jazzy Jalisco** (Tier 1 on 24.04)
- Python 3.12 (system default)
- Crazyradio PA (USB)

## 1. Install ROS2 Jazzy

Follow the official guide: <https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>

Quick version:
```bash
sudo apt update && sudo apt install software-properties-common curl
sudo add-apt-repository universe
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-jazzy-desktop python3-colcon-common-extensions python3-rosdep
sudo rosdep init && rosdep update
```

## 2. Create Python venv (for Crazyflie tools)

```bash
mkdir -p ~/Crazyflie && cd ~/Crazyflie
python3 -m venv crazyflie_venv
source crazyflie_venv/bin/activate
pip install --upgrade pip
pip install cflib cfclient empy==3.3.4 "numpy<2"
```

`empy==3.3.4` and `numpy<2` are required for Crazyswarm2's build-time code generators.

## 3. Install Crazyswarm2

```bash
cd ~/Crazyflie
mkdir -p ros2_ws/src && cd ros2_ws/src
git clone https://github.com/IMRCLab/crazyswarm2 --recursive
cd ..
rosdep install --from-paths src --ignore-src -r -y
```

Build in a **clean shell** (no venv/conda) to avoid Python dependency collisions:
```bash
cd ~/Crazyflie/ros2_ws
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c 'source /opt/ros/jazzy/setup.bash && colcon build --symlink-install'
```

## 4. Install Crazyradio PA udev rules

```bash
sudo tee /etc/udev/rules.d/99-bitcraze.rules > /dev/null <<'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0664", GROUP="plugdev"
EOF
sudo usermod -a -G plugdev $USER
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Log out and back in for group membership to take effect.

## 5. Clone this repo + build the particle_life package

```bash
git clone <swarm_life repo> ~/swarm_life
cd ~/swarm_life/crazyflie_deployment
colcon build --symlink-install
```

## 6. Add workspace sourcing to `.bashrc`

Add these lines to `~/.bashrc`, in this order:

```bash
# source ROS2
source /opt/ros/jazzy/setup.bash
# source crazyswarm2 workspace
source ~/Crazyflie/ros2_ws/install/setup.bash
# source particle_life overlay workspace (this project)
if [ -f ~/swarm_life/crazyflie_deployment/install/setup.bash ]; then
  source ~/swarm_life/crazyflie_deployment/install/setup.bash
fi
# localhost-only ROS + unique domain to avoid shared-lab cross-talk
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=42
# venv LAST so its Python wins the PATH race over conda
source ~/Crazyflie/crazyflie_venv/bin/activate
```

**Ordering matters**: ROS2 first, then Crazyswarm2, then your overlay, then the venv.

## 7. Verify

Open a fresh terminal:
```bash
ros2 pkg list | grep -E "crazyflie|particle_life"
```
Expected output:
```
crazyflie
crazyflie_examples
crazyflie_interfaces
crazyflie_py
crazyflie_sim
motion_capture_tracking
motion_capture_tracking_interfaces
particle_life
```

## 8. First test — simulation

```bash
ros2 launch particle_life particle_life.launch.py backend:=sim
```
A pygame window opens, the drone takes off to 1m, hovers 5s, lands.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'em'` during build | Missing `empy` | `pip install empy==3.3.4` in the venv |
| `numpy/ndarrayobject.h: No such file` | venv numpy 2.x incompatible with Jazzy | Build in clean shell: `env -i HOME=$HOME PATH=/usr/bin:/bin bash -c '...'` |
| `No rule to make target '...libfastcdr.so.2.2.5'` | Stale build cache after ROS2 upgrade | `rm -rf build install log` then rebuild |
| `ros2 pkg list` doesn't show `particle_life` | Overlay not sourced | `source ~/swarm_life/crazyflie_deployment/install/setup.bash` |
| `ros2 pkg list` doesn't show `crazyflie` | Crazyswarm2 workspace not sourced | `source ~/Crazyflie/ros2_ws/install/setup.bash` |

## Package versions (2026-04 baseline)

- ROS2 Jazzy
- Crazyswarm2 upstream commit `1b75fa8` (IMRCLab/main)
- cflib 0.1.32
- cfclient 2026.4
- numpy 1.26.4 (system) / venv pin `"numpy<2"`
- empy 3.3.4
