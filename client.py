import requests
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt

DETECTION_URL = "http://localhost:5000/api"
TEST_IMAGE = "3.jpg"

# 读取图像文件
img = cv2.imread(TEST_IMAGE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.imencode(".jpg", img)[1].tobytes()

# 发送POST请求到服务器
response = requests.post(DETECTION_URL, data=img)

# 解析服务器返回的数据
response_data = response.json()

# 获取返回的图像字节并解码成图像
returned_img_base64 = response_data.get("image_bytes")
returned_img_bytes = base64.b64decode(returned_img_base64)
returned_img = cv2.imdecode(np.frombuffer(returned_img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

# 获取返回的JSON数据
results_json = response_data.get("results")

# 打印返回的JSON数据
print("Results:", results_json)

# 显示返回的图像
plt.imshow(returned_img)
plt.show()

