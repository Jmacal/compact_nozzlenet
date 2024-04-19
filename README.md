
# Compact NozzleNet ROS Package

The NozzleNet ROS package is designed to enhance robotic systems with the ability to monitor and detect the status of a nozzle in real-time. Using a pre-trained ResNet model, this package accurately assesses whether a nozzle is blocked, clear, or requires checking, integrating seamlessly with ROS-based robotic applications.

## Features

- **Real-time Detection**: Subscribes to `/suction_camera/image_raw/compressed` to receive compressed image feeds, facilitating prompt and efficient image processing.
- **Advanced Image Analysis**: Employs a sophisticated ResNet50 model for precise determination of nozzle status, leveraging deep learning to enhance accuracy and reliability.
- **Comprehensive Status Reporting**: Publishes detailed nozzle status to the `/nozzle_status` topic, including the probability of being blocked, duration of blockage, and a categorical status prediction.

## Published Topic Format

The NozzleNet node publishes detailed information on the `/nozzle_status` topic, providing insights into the nozzle's condition. An example message includes:

- `blocked_probability`: The likelihood of the nozzle being blocked, represented as a float (e.g., 0.3704260289669037).
- `duration_blocked`: The duration (in seconds) the nozzle has been blocked. A value of 0.0 indicates no current blockage.
- `prediction`: A string indicating the nozzle's status, which can be `nozzle_clear` (not blocked), `nozzle_blocked` (blocked), or `check_nozzle` (nozzle lifted, requires inspection).

## Prerequisites

- **ROS Environment**: Tested on ROS Kinetic and newer, including Noetic, with Python 3.5+ compatibility.
- **Python Dependencies**: OpenCV, NumPy, TorchVision, Pillow, etc. Install via `requirements.txt`.

## Installation

1. **Clone the Repository**:
   Clone into your ROS workspace's `src` directory.
   ```bash
   cd /path/to/your/catkin_workspace/src
   git clone https://github.com/BucherMunicipal/compact_nozzle_net_pkg
   ```

2. **Install Python Dependencies**:
   Inside the cloned directory, install dependencies.
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Build the Package**:
   From your ROS workspace root, execute `catkin_make`.
   ```bash
   cd /path/to/your/catkin_workspace
   catkin_make
   ```

## Usage

1. **Start the ROS Master**:
   ```bash
   roscore
   ```

2. **Launch NozzleNet Node**:
   To begin image processing and status publication.
   ```bash
   roslaunch compact_nozzle_net_pkg nozzle_net.launch model_name:=model_name

   ```
   model_name options: resnet50_v3, resnet50_v3_v2, resnet50_v6, resnet50_v6_v2.
   If no model is specified, model_name resnet50_v3 is used which points to resnet50_v3.onnx.1.1.8502.GPU.FP16.engine .

## Models

**resnet50_v3** 
ResNet50, Trained on unique frames in dataset, additional transformations including colour transform.

**resnet50_v3_v2** 
ResNet50, Trained on full dataset, no deleted frames, additional transformations including colour transform. v2 trained on additional data.

**resnet50_v6** 
ResNet50, Trained on full dataset, no deleted frames, no colour transform, improved normalisation values, increased cropping.

**resnet50_v6_v2** 
ResNet50, Trained on full dataset, no deleted frames, no colour transform, improved normalisation values, increased cropping. v2 trained on additional data.


## Contact

James MacAleese, james.macaleese@buchermunicipal.net.
