"""Perform test request"""
import pprint

import requests

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
TEST_IMAGE = r"strawberry4300\JPEGImages\angular_leafspot1_jpg.rf.a0cd662ad123e928cbd2cb144b1dbf8a.jpg"

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)
