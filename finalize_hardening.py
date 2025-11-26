import urllib.request
import urllib.error
import sys
import time

url = "http://127.0.0.1:8000/diagnose"
boundary = "boundary"
headers = {
    "Content-Type": f"multipart/form-data; boundary={boundary}"
}

# Construct multipart body
body = []
body.append(f"--{boundary}".encode())
body.append(b'Content-Disposition: form-data; name="age"')
body.append(b'')
body.append(b'150')

body.append(f"--{boundary}".encode())
body.append(b'Content-Disposition: form-data; name="sex"')
body.append(b'')
body.append(b'male')

body.append(f"--{boundary}".encode())
body.append(b'Content-Disposition: form-data; name="site"')
body.append(b'')
body.append(b'torso')

body.append(f"--{boundary}".encode())
body.append(b'Content-Disposition: form-data; name="file"; filename="dummy.jpg"')
body.append(b'Content-Type: image/jpeg')
body.append(b'')
body.append(b'\x00' * 100)
body.append(f"--{boundary}--".encode())
body.append(b'')

data = b'\r\n'.join(body)

req = urllib.request.Request(url, data=data, headers=headers, method="POST")

print("🔥 Running Smoke Test: Invalid Age (150)...")
# Wait a bit for server reload
time.sleep(2)

try:
    with urllib.request.urlopen(req) as response:
        print(f"❌ FAILURE: Expected 400, got {response.status}")
        sys.exit(1)
except urllib.error.HTTPError as e:
    if e.code == 400:
        print("✅ SUCCESS: Server correctly rejected invalid age (400 Bad Request).")
        print(f"Response: {e.read().decode()}")
    else:
        print(f"❌ FAILURE: Expected 400, got {e.code}")
        print(f"Response: {e.read().decode()}")
        sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: Connection failed. Is the server running? {e}")
    sys.exit(1)
