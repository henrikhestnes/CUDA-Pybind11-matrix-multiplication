# CUDA-Pybind11-matrix-multiplication
Code for GPU-accelerating matrix multiplication in Python by exposing C++ and CUDA code to Python using Pybind11.

## Prerequisites
- Cuda installed in /usr/local/cuda
- CMake 3.3 or later
- Python 3.8.10 or later
- PythonInterp 3.6 or later
- PythonLibs 3.6 or later

## Usage
Should compile out of the box by doing the following:
##### Bind C++ module to Python
```sudo chmod +x bind_code.sh```<br>
```./bind_code.sh```
##### Test code in Python
```python3 matmul.py```
