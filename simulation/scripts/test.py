import time
from pynput import keyboard

def a():
    print("test!")

l = keyboard.Listener(on_press=a)
l.start()

while True:
    time.sleep(1)