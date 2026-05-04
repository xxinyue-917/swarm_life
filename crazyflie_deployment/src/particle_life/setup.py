from setuptools import find_packages, setup
from glob import glob

package_name = 'particle_life'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xxinyue',
    maintainer_email='xxinyue@umich.edu',
    description='Particle life swarm behaviors on Crazyflie drones',
    license='MIT',
    entry_points={
        'console_scripts': [
            'particle_life_node = particle_life.particle_life_node:main',
            'viewer = particle_life.viewer:main',
        ],
    },
)
