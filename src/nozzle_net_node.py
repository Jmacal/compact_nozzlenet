#!/usr/bin/env python3

import os
import rospy
from sensor_msgs.msg import CompressedImage
from nozzle_net.msg import NozzleStatus  # Custom message type
import cv2
import numpy as np
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import yaml  # Import YAML module for configuration loading
import rospkg  # Import rospkg module

class NozzleNet:
    def __init__(self):
        try:
            self.load_config()  # Load configuration settings
            # Initialize ONNX model with the adjusted model path
            model_path = os.path.join(rospkg.RosPack().get_path('nozzle_net_pkg'), self.config['model_path'])
            self.model = onnxruntime.InferenceSession(model_path)
        except (yaml.YAMLError, FileNotFoundError) as e:
            rospy.logerr("Failed to load configuration: {}".format(e))
            raise e
        except onnxruntime.OnnxRuntimeException as e:
            rospy.logerr("Failed to load ONNX model: {}".format(e))
            raise e

        try:
            # Define image transformations
            self.transform = transforms.Compose([
                transforms.Resize((self.config['image_height'], self.config['image_width'])),
                transforms.Lambda(lambda x: transforms.functional.crop(x, self.config['crop_top'], self.config['crop_bottom'], self.config['image_width'], self.config['image_height'])),  # Adjusted crop transformation
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config['mean'], std=self.config['std'])
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

    def load_config(self):
        package_path = rospkg.RosPack().get_path('nozzle_net_pkg')
        config_file = os.path.join(package_path, 'config', 'nozzle_config.yaml')
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def image_callback(self, data):
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except ValueError as e:
            rospy.logwarn("Failed to decode image: {}".format(e))

    def process_latest_image(self):
        if self.latest_image is None:
            return

        try:
            image_pil = Image.fromarray(cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB))
            transformed_img = self.transform(image_pil)
            transformed_img = transformed_img.unsqueeze(0)

            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            pred = self.model.run([output_name], {input_name: transformed_img.numpy()})[0]

            current_time = rospy.Time.now()
            if pred.argmax() == 1 and self.first_blocked_time is None:
                self.first_blocked_time = current_time
            elif pred.argmax() != 1:
                self.first_blocked_time = None

            duration_blocked = (current_time - self.first_blocked_time).to_sec() if self.first_blocked_time else 0.0

            nozzle_status = NozzleStatus()
            nozzle_status.blocked_probability = pred[0][1]
            nozzle_status.duration_blocked = duration_blocked
            nozzle_status.prediction = self.labels[pred.argmax()]
            self.publisher.publish(nozzle_status)
        except Exception as e:
            rospy.logerr("Failed to process image: {}".format(e))

if __name__ == '__main__':
    rospy.init_node('nozzle_net')
    processor = NozzleNet()
    rate = rospy.Rate(10)  # Set the rate for the while loop
    while not rospy.is_shutdown():
        try:
            processor.process_latest_image()
        except Exception as e:
            rospy.logerr("Error during image processing: {}".format(e))
        rate.sleep()

   
