"""Perform test request"""
import pprint
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt

DETECTION_URL = "http://localhost:5000/api"
TEST_IMAGE = r"3.jpg"

img = cv2.imread(TEST_IMAGE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.imencode(".jpg", img)[1].tobytes()

response = requests.post(DETECTION_URL, data=img)

# 返回结果绘图
result = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)

# pprint.pprint(response)

plt.imshow(result)
plt.show()
