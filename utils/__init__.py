import os
import sys

# Add project library to Python path
python_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Adding '{python_src_dir}' to sys.path")
sys.path.append(python_src_dir)  # /code/python/src/
sys.path.append(python_src_dir + "/utils/")  # /code/python/src/
sys.path.append(os.path.dirname(python_src_dir))  # /code/python/
 
 