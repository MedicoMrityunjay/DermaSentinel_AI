import requests
import numpy as np
import cv2
import io

def test_api():
    url = "http://localhost:8000/diagnose"
    
    # Create dummy image (Reddish lesion)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 100, (0, 0, 200), -1) # Red circle
    
    # Encode
    _, buf = cv2.imencode(".jpg", img)
    img_bytes = io.BytesIO(buf.tobytes())
    
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {
        "age": 45,
        "sex": "male",
        "site": "torso"
    }
    
    try:
        resp = requests.post(url, files=files, data=data)
        print(f"Status Code: {resp.status_code}")
        print("Response:")
        print(resp.json())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
