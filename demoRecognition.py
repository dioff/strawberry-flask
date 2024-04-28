import cv2
import numpy as np
import requests
from io import BytesIO

def draw_boxes(image_path, result):
    image = cv2.imread(image_path)
    obj = result["result"]
    for detection in obj:
        xmin = int(detection["xmin"])
        ymin = int(detection["ymin"])
        xmax = int(detection["xmax"])
        ymax = int(detection["ymax"])
        name = detection["name"]
        confidence = detection["confidence"]
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(image, f"{name}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    cv2.imshow("Image with boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def upload_image(file_path):
    url = "http://www.iothinking.com:8086/api/predict"
    files = {'image': open(file_path, 'rb')}

    try:
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print("Error:", response.status_code)
            return None
    except Exception as e:
        print("Exception:", e)
        return None

# 上传照片并获取结果
file_path = "1.jpg"  # 替换为你的图像文件路径
result = upload_image(file_path)
if result is not None:
    print("返回结果:", result)
    draw_boxes(file_path, result)
