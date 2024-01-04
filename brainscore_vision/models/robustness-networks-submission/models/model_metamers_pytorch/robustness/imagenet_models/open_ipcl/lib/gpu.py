'''
    nvidia-smi -q -a
'''
import subprocess
import re
import numpy as np
import time

def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')

def print_gpu_info():
    command = 'nvidia-smi -q'.split()
    temps = []
    for line in run_command(command):
        txt = str(line)
        print(txt)
            
def get_gpu_temps():
    command = 'nvidia-smi -q'.split()
    temps = []
    for line in run_command(command):
        txt = str(line)
        if 'GPU Current Temp' in txt:   
            temp = int(re.findall(r'\d+', txt)[0])
            temps.append(temp)

    return np.array(temps)

def is_temp_safe(threshold=88, sleep=5):
    temps = get_gpu_temps()
    safe = (temps < threshold).all()
    
    if safe:
        #print("GPU temperatures are safe:", temps)
        pass
    else:
        print(f"GPUs too hot! sleeping {sleep}s...", temps)
        time.sleep(sleep)
        
    return safe