import cv2
import numpy as np 
import time
import threading

class ScreenUI:
    def __init__(self, robot, fps, camera_name):
        self.robot = robot
        self.fps = fps
        self.camera_name = camera_name
        self.stop_event = threading.Event()
        self.thread = None
        self.placeholder_img = cv2.imread("media/readme/lerobot-logo-thumbnail.png")

        if self.placeholder_img is None: #Placeholder if image not found
            self.placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder if image not found
            cv2.putText(self.placeholder_img, "Alohamini Camera Feed", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return

    def start(self):
        self.thread = threading.Thread(target=self.run_screen, daemon=True)
        self.thread.start()
        print("🖥️ Screen UI started.")

    def stop(self):
        self.stop_event.set()

        if self.thread:
            self.thread.join(timeout=2.0)

        cv2.destroyAllWindows() # Cleanup windows on exit
        print("🖥️ Screen UI stopped.")

    def run_screen(self):
        #Camera loop
        while not self.stop_event.is_set():
            t0 = time.perf_counter()

            if self.robot.is_connected:
                obs = self.robot.get_observation()

                if obs and self.camera_name in obs:
                    frame = obs[self.camera_name]
                else:
                    frame = self.placeholder_img
            else:
                frame = self.placeholder_img

            cv2.imshow("AlohaMini Camera", frame)

            if cv2.waitKey(1) == 27:  # ESC closes window
                self.stop()
                break

            dt = time.perf_counter() - t0
            sleep = max(1.0 / self.fps - dt, 0)
            time.sleep(sleep)
