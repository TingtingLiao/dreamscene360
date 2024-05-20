## Introduction
An unofficial pytorch implementation of [**DreamScene360**](https://dreamscene360.github.io/)
: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting. 

<!-- ![a cozy living room_panorama_image](https://github.com/TingtingLiao/dreamscene360/assets/45743512/1be2908e-94e4-4d57-bb8a-92c7d437e62c)  -->

![panorama_image](https://github.com/TingtingLiao/dreamscene360/assets/45743512/c0869131-ab1b-41dd-834d-7450d9c84554)
![depth_blending_vis](https://github.com/TingtingLiao/dreamscene360/assets/45743512/0e725bc3-9021-4224-8266-7571deac5aac)

  
## Install
```bash
git clone  --recursive https://github.com/TingtingLiao/dreamscene360.git 
cd dreamscene360
conda create -n dreamscene360 python=3.10 
conda activate dreamscene360 

# torch2.3.0+cu12.1 
pip install torch torchvision torchaudio
 
# requirements
pip install -r requirements.txt

# xformers  
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
 
# diffusion360 
pip install git+https://github.com/archerfmy/sd-t2i-360panoimage
huggingface-cli download archerfmy0831/sd-t2i-360panoimage
 
pip install -e submodules/diff-surfel-rasterization 
pip install -e submodules/simple-knn
 
sudo apt-get install libgtest-dev libeigen3-dev libboost-all-dev libopencv-dev libatlas-base-dev
sudo apt-get install liblapack-dev libsuitesparse-dev libcxsparse3 libgflags-dev libgoogle-glog-dev libgtest-dev
conda install -c conda-forge libstdcxx-ng=12  

# pybind11
cd submodules/360monodepth/code/cpp/3rd_party
git clone https://github.com/pybind/pybind11.git 
cd pybind11 && mkdir build && cd build
cmake .. && make -j8 
sudo make install
cd ../../ 

# ceres-solver
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver && mkdir build && cd build
cmake .. && make -j8 
sudo make install 
cd ../../../  

# instaOmniDepth
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release  ..
make -j8 
cd ../python
python setup.py build
python setup.py bdist_wheel 
pip install dist/instaOmniDepth-0.1.0-cp310-cp310-linux_x86_64.whl # if failed, please check your file version in dist/ 
```

## Usage 
```bash 
python main.py
```
## Acknowledgement 
Special thanks to the projects and their contributors:
* [DreamScene360](https://dreamscene360.github.io/)
* [Diffusion360](https://github.com/ArcherFMY/SD-T2I-360PanoImage)
* [360monodepth](https://github.com/manurare/360monodepth)
* [2DGS](https://github.com/hbb1/2d-gaussian-splatting)