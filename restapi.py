import argparse
import io
import json
import base64
import torch
from PIL import Image
from flask import Flask, request, jsonify

# 图像处理
import numpy as np
import cv2

app = Flask(__name__)

DETECTION_URL = "/api"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return jsonify({"error": "Method not allowed"}), 405

    response_data = {}

    # 对图像进行处理
    if request.data:
        img = cv2.imdecode(np.frombuffer(request.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        results = model(img)  # reduce size=320 for faster inference
        results_json = results.pandas().xyxy[0].to_json(orient="records")

        response_data["results"] = json.loads(results_json)
        
        results = results.render()[0]

        # 将图像编码为字节流
        # img_bytes = cv2.imencode(".jpg", img)[1].tobytes()
        _, img_bytes = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(img_bytes).decode()
        response_data["image_bytes"] = img_base64

    return jsonify(response_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load(".", "custom", path=r"runs\train\exp\weights\best.pt", source="local")  # force_reload to recache
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
