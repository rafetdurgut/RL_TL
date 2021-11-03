from itertools import product
import os
import threading
import time

def thread_function(conf):
    time.sleep(1)
    print(conf)
    os.system(f"start cmd /k Python ./Run_For_SUKP.py {' '.join(map(str,conf.values()))}")
parameters = {"Method": ["average", "extreme"], "W": [5,25], "Pmin": [0.1,0.2], "Alpha": [0.1, 0.5, 0.9]}
configurations = [dict(zip(parameters, v)) for v in product(*parameters.values())]
for c in configurations:
    x = threading.Thread(target=thread_function, args=(c,))
    x.start()