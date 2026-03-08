import os
import zmq
import sys
import cv2
import json
import time
import random
import base64
import argparse
import numpy as np
from isaacsim import SimulationApp

def log(msg: str):
    print(f"\033[1;36m{msg}\033[0m")

def error(msg: str):
    print(f"\033[1;31m[ERROR] {msg}\033[0m")