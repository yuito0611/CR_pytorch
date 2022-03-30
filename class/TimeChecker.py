import time
import math

class TimeChecker():
    def __init__(self):
        self.start_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        now = time.time()
        s = now - self.start_time
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)