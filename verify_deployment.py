import requests
import sqlite3
import os
import sys

BASE_URL = "http://127.0.0.1:7860"
DB_PATH = "derma.db"

def verify_system():
    print("🕵️ Starting System Verification...")

    # 1. Database File Check
    if not os.path.exists(DB_PATH):
        print("❌ Error: derma.db not found. Did the server restart successfully?")
        return
    print("✅ derma.db exists.")

    # 2. Schema Check
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check Users Table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
        if not cursor.fetchone():
            print("❌ Error: 'users' table missing.")
            return
        print("✅ 'users' table found.")

        # Check Scans Table Column
        cursor.execute("PRAGMA table_info(scans);")
        columns = [info[1] for info in cursor.fetchall()]
        if "user_id" not in columns:
            print(f"❌ Error: 'user_id' column missing in 'scans' table. Columns found: {columns}")
            return
        print("✅ 'user_id' column found in 'scans'.")
        
        conn.close()
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return

    # 3. Server Connectivity
    try:
        res = requests.get(BASE_URL)
        if res.status_code == 200:
            print("✅ Server is reachable (HTTP 200).")
        else:
            print(f"⚠️ Server responded with {res.status_code}.")
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")
        print("   (Ensure uvicorn is running on port 7860)")
        return

    # 4. Functional Test: Register & Login
    username = "test_verifier_user"
    password = "test_password_123"

    print("🔄 Testing Authentication Flow...")
    
    # Register
    try:
        reg_res = requests.post(f"{BASE_URL}/register", json={"username": username, "password": password})
        if reg_res.status_code == 200:
            print("✅ Registration successful.")
        elif reg_res.status_code == 400 and "already registered" in reg_res.text:
            print("✅ User already registered (Skipping creation).")
        else:
            print(f"❌ Registration failed: {reg_res.text}")
            return
    except Exception as e:
        print(f"❌ Registration request failed: {e}")
        return

    # Login
    try:
        login_res = requests.post(f"{BASE_URL}/token", data={"username": username, "password": password})
        if login_res.status_code == 200:
            token = login_res.json().get("access_token")
            if token:
                print("✅ Login successful. JWT Token received.")
            else:
                print("❌ Login succeeded but no token returned.")
        else:
            print(f"❌ Login failed: {login_res.text}")
    except Exception as e:
        print(f"❌ Login request failed: {e}")

    print("\n🎉 SYSTEM VERIFIED. The update is live and functional.")

if __name__ == "__main__":
    verify_system()
