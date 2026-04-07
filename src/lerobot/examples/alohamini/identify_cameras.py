"""
Displays all detected camera feeds in Rerun with their USB port path and mean
brightness overlaid. Wave your hand in front of each camera to identify which
physical port it corresponds to.

Run ON THE JETSON. To view on your Mac:
  1. On Mac:    rerun
  2. On Jetson: python3 src/lerobot/examples/alohamini/identify_cameras.py --remote-ip <mac-ip>

To run locally (Rerun viewer spawns on the Jetson itself):
    python3 src/lerobot/examples/alohamini/identify_cameras.py
"""

import argparse
import sys
import threading
import time

import cv2
import numpy as np
import rerun as rr

from lerobot.robots.alohamini.config_lekiwi import lekiwi_cameras_config


def open_captures() -> list[tuple[str, cv2.VideoCapture]]:
    """Open captures using the same camera config as LeKiwi teleop."""
    captures = []
    for label, cfg in lekiwi_cameras_config().items():
        path = str(cfg.index_or_path)
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[WARN] Could not open {path} ({label}), skipping.")
            continue
        if cfg.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        if cfg.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        if cfg.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        if cfg.fps:
            cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize internal queue so reads are always the latest frame
        print(f"[OK]   Opened camera {label} -> {path}")
        captures.append((label, cap))
    return captures


def draw_label(frame: np.ndarray, label: str, brightness: float) -> np.ndarray:
    """Burn the port label and brightness value into the frame."""
    out = frame.copy()
    text_port = f"port: {label}"
    text_bright = f"brightness: {brightness:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, text_port,   (10, 30),  font, 0.8, (0, 0, 0),   3, cv2.LINE_AA)
    cv2.putText(out, text_port,   (10, 30),  font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text_bright, (10, 65),  font, 0.8, (0, 0, 0),   3, cv2.LINE_AA)
    cv2.putText(out, text_bright, (10, 65),  font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote-ip", type=str, default=None, help="IP of the Mac running the Rerun viewer (e.g. 192.168.1.42)")
    parser.add_argument("--remote-port", type=int, default=9876, help="Port the Rerun viewer is listening on (default: 9876)")
    parser.add_argument(
        "--send-fps",
        type=float,
        default=10.0,
        help="Max per-camera FPS sent to Rerun. Lower values reduce network usage (default: 10).",
    )
    parser.add_argument(
        "--compress-images",
        action="store_true",
        help="Send compressed images to Rerun to reduce bandwidth.",
    )
    return parser.parse_args()


def init_rerun(remote_ip: str | None, remote_port: int) -> None:
    rr.init("camera_identify")
    if remote_ip:
        url = f"rerun+http://{remote_ip}:{remote_port}/proxy"
        print(f"Connecting to Rerun viewer at {url}")
        rr.connect_grpc(url=url)
    else:
        rr.spawn()


def main() -> None:
    args = parse_args()

    cameras = lekiwi_cameras_config()
    if not cameras:
        print("No cameras found in LeKiwi config.")
        sys.exit(1)

    print(f"\nFound {len(cameras)} camera(s):")
    for label, cfg in cameras.items():
        print(f"  {label} -> {cfg.index_or_path}")

    captures = open_captures()
    if not captures:
        print("Could not open any cameras.")
        sys.exit(1)

    init_rerun(args.remote_ip, args.remote_port)

    print("\nStreaming to Rerun — wave your hand in front of each camera to identify it.")
    print("Press Ctrl+C to stop.\n")

    stop_event = threading.Event()
    send_period = 1.0 / max(args.send_fps, 0.1)

    def stream_camera(label: str, cap: cv2.VideoCapture) -> None:
        next_send_t = 0.0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            now = time.monotonic()
            if now < next_send_t:
                continue
            next_send_t = now + send_period
            brightness = float(np.mean(frame))
            annotated = draw_label(frame, label, brightness)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            image_entity = rr.Image(rgb).compress() if args.compress_images else rr.Image(rgb)
            rr.log(f"cameras/{label}", image_entity)

    threads = [threading.Thread(target=stream_camera, args=(label, cap), daemon=True) for label, cap in captures]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2)
        for _, cap in captures:
            cap.release()


if __name__ == "__main__":
    main()
