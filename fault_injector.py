#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import random
from random import random
from random import uniform
from random import randint
from random import choice
import numpy as np

add_fault = True
last_faulty_image = None

# # # FAULTY FUNCTION # # #
def apply_gaussian_noise(image, mean, std_dev):
    
    global add_fault
    global last_faulty_image

    if add_fault:
        h, w, c = image.shape
        noise = np.random.normal(mean, std_dev, (h, w, c)).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        last_faulty_image = noisy_image
        add_fault = False

    elif last_faulty_image is not None:
        noisy_image = last_faulty_image

    else:
        noisy_image = image

    return noisy_image

def apply_image_blur(image, kernel_size):

    global add_fault
    global last_faulty_image

    if add_fault:
        blurred_image = cv2.blur(image, (kernel_size, kernel_size))
        last_faulty_image = blurred_image
        add_fault = False

    elif last_faulty_image is not None:
        blurred_image = last_faulty_image
    
    else: 
        blurred_image = image

    return blurred_image

# # # CALLBACK # # #
def image_callback_gaussian_noise(msg, std_dev_random):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    faulty_image = apply_gaussian_noise(cv_image, mean=0, std_dev = std_dev_random)
    
    # Convert the modified image back to sensor_msgs/Image
    modified_msg = bridge.cv2_to_imgmsg(faulty_image, encoding='bgr8')
    modified_msg.header = msg.header

    pub = rospy.Publisher('/camera/color/image_faulty', Image, queue_size=10)
    
    # Publish the modified image
    pub.publish(modified_msg)

def callback_image_blur(msg, kernel):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    kernel_size = kernel
       
    faulty_image = apply_image_blur(cv_image, kernel_size)
    
    # Convert the modified image back to sensor_msgs/Image
    modified_msg = bridge.cv2_to_imgmsg(faulty_image, encoding='bgr8')
    modified_msg.header = msg.header

    pub = rospy.Publisher('/camera/color/image_faulty', Image, queue_size=10)
    
    # Publish the modified image
    pub.publish(modified_msg)

def image_fault_injector_node():
    rospy.init_node('image_fault_injector')
    with open(".../image_faults.yaml", 'r') as stream:
        fault_config = yaml.safe_load(stream)
    fault_type = fault_config["Fault_types"]
    fault_index = randint(0, len(fault_type) - 1)
    
    # global selected_fault_type
    selected_fault_type = fault_type[fault_index]
      
    rospy.logwarn_once("injected fault is : %s", selected_fault_type)
        
# # # CHOOSE A FAULT # # # 
    if selected_fault_type == 'Gaussian_noise':
        std_dev_random = uniform(1.5, 1.5)
        std_dev_random = round(std_dev_random,1)
        rospy.logwarn_once('std_dev is %f', std_dev_random)
        rospy.Subscriber('/camera/color/image_raw', Image, lambda data: image_callback_gaussian_noise(data, std_dev_random))
        rospy.spin()

    elif selected_fault_type == 'Image_blur':
        kernel_size = randint(50, 50)
        rospy.logwarn('kernel_size is %d', kernel_size)
        rospy.Subscriber('/camera/color/image_raw', Image, lambda data: callback_image_blur(data, kernel_size))
        rospy.spin()

if __name__ == '__main__':
    image_fault_injector_node()
    

