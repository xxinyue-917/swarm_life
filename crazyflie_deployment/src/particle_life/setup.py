from setuptools import find_packages, setup
from glob import glob

package_name = 'particle_life'

# The canonical project configs live at crazyflie_deployment/config/ (two levels up
# from this setup.py). Installing them into the ROS package share makes the whole
# stack portable: anyone who clones swarm_life and `colcon build`s gets working
# crazyflies/motion_capture/arena/species/preset YAMLs in the install tree.
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('../../config/*.yaml')),
        ('share/' + package_name + '/config/presets', glob('../../config/presets/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xxinyue',
    maintainer_email='xxinyue@umich.edu',
    description='Particle life swarm behaviors on Crazyflie drones',
    license='MIT',
    entry_points={
        'console_scripts': [
            'particle_life = particle_life.particle_life:main',
            'viewer = particle_life.viewer:main',
            'flight_test = particle_life.flight_test:main',
            'flight_test_all = particle_life.flight_test_all:main',
            'hover_test = particle_life.hover_test:main',
            'hover_all = particle_life.hover_all:main',
            'hover_one = particle_life.hover_one:main',
            'circle_all = particle_life.circle_all:main',
            'fake_server = particle_life.fake_server:main',
        ],
    },
)
