#!/usr/bin/env python3

import os
import rospy
from sensor_msgs.msg import CompressedImage
from compact_nozzle_net_pkg.msg import NozzleStatus  # Custom message type
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import rospkg 
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import time

class NozzleNet:
    def __init__(self):
        try:
            # Retrieve model name from ROS parameter server, set by the launch file
            model_name = rospy.get_param('~model_name', 'resnet50_v0')

            # Construct the full engine path using the retrieved engine name
            model_path = os.path.join(rospkg.RosPack().get_path('compact_nozzle_net_pkg'), 'model', model_name + '.onnx.1.1.8502.GPU.FP16.engine')

            # Load the TensorRT engine
            self.engine = self.load_engine(model_path)
            self.context = self.engine.create_execution_context()

        except Exception as e:
            rospy.logerr("Failed to load TensorRT engine: {}".format(e))
            raise e

        try:
            if model_name in ['resnet50_v0', 'resnet50_v1', 'resnet50_v2', 'resnet50_v3']:
                # Define image transformations
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.Lambda(lambda x: transforms.functional.crop(x, 0, 110, 224, 224)),  # Adjusted crop transformation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Define image transformations
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.Lambda(lambda x: transforms.functional.crop(x, 70, 105, 224, 224)),  # Adjusted crop transformation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4387, 0.4716, 0.4956], std=[0.1933, 0.1992, 0.2179])
                ])

        except Exception as e:
            rospy.logerr("Failed to define image transformations: {}".format(e))
            raise e

        # Initialize ROS publisher and subscriber
        self.publisher = rospy.Publisher('/nozzle_status', NozzleStatus, queue_size=10)
        rospy.Subscriber('/suction_camera/image_raw/compressed', CompressedImage, self.image_callback)

        self.latest_image = None
        self.first_blocked_time = None
        self.is_last_blocked = False
        self.labels = ['check_nozzle', 'nozzle_blocked', 'nozzle_clear']

    def load_engine(self, model_path):
        # Load and deserialize the TensorRT engine
        with open(model_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def image_callback(self, data):
        # Process image data from ROS topic
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except ValueError as e:
            rospy.logwarn("Failed to decode image: {}".format(e))

    def process_latest_image(self):
        # Process the latest received image
        if self.latest_image is None:
            return

        try:
            # Convert image only once
            image_pil = Image.fromarray(cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB))
            transformed_img = self.transform(image_pil)
            transformed_img = transformed_img.unsqueeze(0)

            # Allocate device memory for inputs and outputs
            input_name = 'input_0'
            output_name = 'output_0'
            input_shape = (1, 3, 224, 224)
            input_h = cuda.mem_alloc(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 4)
            output_shape = (1, len(self.labels))
            output_h = cuda.mem_alloc(output_shape[0] * output_shape[1] * 4)

            # Transfer input data to device
            cuda.memcpy_htod(input_h, transformed_img.numpy().tobytes())

            # Run inference
            self.context.execute_v2(bindings=[int(input_h), int(output_h)])

            # Transfer predictions back from device
            output_data = np.zeros(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output_data, output_h)

            # Efficient handling of blocked status
            current_time = rospy.Time.now()
            is_blocked = np.argmax(output_data) == 1
            if is_blocked and self.first_blocked_time is None:
                self.first_blocked_time = current_time
            elif not is_blocked:
                self.first_blocked_time = None

            # Only calculate duration when needed
            duration_blocked = (current_time - self.first_blocked_time).to_sec() if self.first_blocked_time else 0.0

            # Construct message only when necessary
            nozzle_status = NozzleStatus()
            nozzle_status.blocked_probability = output_data[0][1]
            nozzle_status.duration_blocked = duration_blocked
            nozzle_status.prediction = self.labels[np.argmax(output_data)]
            self.publisher.publish(nozzle_status)

        except Exception as e:
            rospy.logerr("Failed to process image: {}".format(e))

if __name__ == '__main__':
    # Initialize ROS node and start processing
    rospy.init_node('nozzle_net')
    processor = NozzleNet()
    rate = rospy.Rate(10)  # Set the rate for the while loop
    while not rospy.is_shutdown():
        try:
            processor.process_latest_image()
        except Exception as e:
            rospy.logerr("Error during image processing: {}".format(e))
        rate.sleep()
