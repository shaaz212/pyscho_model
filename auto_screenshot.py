import pyautogui
import time
from datetime import datetime
from pathlib import Path

folder = Path("screenshots")
folder.mkdir(exist_ok=True)

print("Taking screenshots every 5 seconds. Press Ctrl+C to stop.")

try:
    while True:
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        pyautogui.screenshot().save(folder / filename)
        print(f"Saved: {filename}")
        time.sleep(5)
except KeyboardInterrupt:
    print("Screenshot capture stopped.")