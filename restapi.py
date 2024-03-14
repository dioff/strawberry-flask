import argparse
import io

import torch
from PIL import Image
from flask import Flask, request

# 图像处理
import numpy as np
import cv2

app = Flask(__name__)

DETECTION_URL = "/api"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    # 对图像进行处理
    if request.data:
        img = cv2.imdecode(np.frombuffer(request.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        results = model(img)  # reduce size=320 for faster inference
        results = results.render()[0]
        return cv2.imencode(".jpg", img)[1].tobytes()
    
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=640)  # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load(".", "custom", path=r"runs\train\exp\weights\best.pt", source="local")  # force_reload to recache
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
