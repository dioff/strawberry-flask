import requests

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
file_path = r"C:\Users\Lazzy\Desktop\识别程序\3.jpg"  # 替换为你的图像文件路径
result = upload_image(file_path)
if result is not None:
    print("返回结果:", result)
