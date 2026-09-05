from glob import glob

from setuptools import find_packages, setup

package_name = "pennair_vision"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nelson Nishio",
    maintainer_email="datinnodev@gmail.com",
    description="ROS 2 nodes for the PennAir shape detector.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "video_publisher = pennair_vision.video_publisher:main",
            "shape_detector = pennair_vision.shape_detector:main",
        ],
    },
)
