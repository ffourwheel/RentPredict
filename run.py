import subprocess
import sys
import os

def install_requirements():
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    print("กำลังติดตั้ง dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file, "-q"])
    print("ติดตั้งสำเร็จ!\n")

def main():
    install_requirements()

    import webbrowser
    import threading
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:5000")
    threading.Thread(target=open_browser, daemon=True).start()

    from app import app
    print("เปิดเบราว์เซอร์ไปที่: http://127.0.0.1:5000")
    print("กด Ctrl+C เพื่อหยุด server")
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    main()
