## Introduction
An unofficial pytorch implementation of [**DreamScene360**](https://dreamscene360.github.io/)
Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting. 

![a cozy living room_panorama_image](https://github.com/TingtingLiao/dreamscene360/assets/45743512/1be2908e-94e4-4d57-bb8a-92c7d437e62c) 

  
## Install
```bash
git clone https://github.com/TingtingLiao/dreamscene360.git 
cd dreamscene360
conda create -n test python=3.10 

# torch2.3.0+cu12.1 
pip install torch torchvision torchaudio

# xformers  
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# requirements
pip install -r requirements.txt

# diffusion360 
pip install git+https://github.com/archerfmy/sd-t2i-360panoimage
huggingface-cli download archerfmy0831/sd-t2i-360panoimage

pip install -e submodules/diff-surfel-rasterization
pip install -e submodules/simple-knn

# install the 
sudo apt-get install libgtest-dev libeigen3-dev libboost-all-dev libopencv-dev libatlas-base-dev
sudo apt-get install liblapack-dev libsuitesparse-dev libcxsparse3 libgflags-dev libgoogle-glog-dev libgtest-dev

# pybind11
cd ./3dparty
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
cd ../../  

# instaOmniDepth
cmake -DCMAKE_BUILD_TYPE=Release  ..
make -j8
cd ../python  
python setup.py build
python setup.py bdist_wheel 
pip install dist/instaOmniDepth-0.1.0-cp310-cp310-linux_x86_64.whl # check your file version by "ls dist/"
conda install -c conda-forge libstdcxx-ng=12 
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

## Citation 
If you find this project helpful, please consider citing it: 

```bibtex 
@misc{dreamscene,
  author = {Tingting Liao},
  title = {DreamScene360:Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TingtingLiao/dreamscene360}}
}

```