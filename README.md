# NozzleNet ROS Package

The NozzleNet ROS package provides a ROS node for detecting the status of a nozzle using a pre-trained ResNet50 model.

## Features

- Subscribes to the `/suction_camera/image_raw/compressed` topic to receive compressed images.
- Processes the images using a pre-trained ResNet50 model to determine the status of the nozzle.
- Publishes the status of the nozzle to the `/nozzle_status` topic.

## Dependencies

- ROS Kinetic (or newer)
- Python 2.7 (or Python 3.x)
- OpenCV
- NumPy
- ONNX Runtime
- TorchVision
- Pillow

## Installation

1. Clone this repository into your ROS workspace:

   ```bash
   cd /path/to/your/catkin_workspace/src
   git clone <repository_url>

2. Navigate to your ROS workspace and build the package:

   ```bash
   cd /path/to/your/catkin_workspace
   catkin_make
   ```

## Usage

1. Launch the ROS master node:

   ```bash
   roscore
   ```

2. Run the NozzleNet node using the provided launch file:

   ```bash
   roslaunch nozzle_net_pkg nozzle_net.launch
   ```

## Configuration

Modify the nozzle_config.yaml file to adjust parameters such as image dimensions, model paths, or image processing settings as needed.

## Customization

You can customize the behavior of the NozzleNet node by modifying the nozzle_net_node.py script. Ensure that you understand the implications of the changes you make.

## License

This package is released under the MIT License. See LICENSE for details.

## Contact

For any inquiries or issues regarding this package, please contact james.macaleese@buchermunicipal.net.

