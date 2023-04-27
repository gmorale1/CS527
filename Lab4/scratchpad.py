import numpy as np
from concurrent.futures import ThreadPoolExecutor
import subprocess
import pandas as pd
import seaborn as sns

args = [
    "python", "./pso.py",
    "--num_particles", str(40),
    "--inertia", str(0.5),
    "--cognition", str(1),
    "--social", str(1),
    "--func", "Rosenbrock",
]
result = subprocess.run(args, capture_output=True, text=True)

lines = result.stdout.strip().split("\n")

# Parse the output and extract the variable values
output_list = []
for line in lines:
    print(line)
#     name, value = line.split(":")
#     output_list.append({name: value})
# output_list.append({"test_num":test})
# test_df = pd.DataFrame(output_list)

# results = pd.concat([results,test_df], ignore_index=True)