import os
import psutil

class MemoryFollower():

    def __init__(self):
        pid = os.getpid()
        self.ps = psutil.Process(pid)
        self.flag = 0
        self.memory_info = []

    def print(self):
        memoryUse = self.ps.memory_info()
        print("\n###")
        print(f"## Flag {self.flag} -> memory used : {memoryUse.rss//10**6} MB")
        print("###\n")
        self.memory_info.append(memoryUse.rss//10**6)
        self.flag += 1

    def reset(self):
        self.flag = 0
        self.memory_info = []