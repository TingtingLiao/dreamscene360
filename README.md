## Introduction
An unofficial implementation of [**DreamScene360**](https://dreamscene360.github.io/). Our project supports both text and image to scene generation.  

https://github.com/TingtingLiao/dreamscene360/assets/45743512/c972d419-eb5d-4899-8a83-9f04c6c2d7bc

![panorama_image](https://github.com/TingtingLiao/dreamscene360/assets/45743512/2352781b-2ef5-4d84-a65b-30046733d6d3)

![depth_blending_vis](https://github.com/TingtingLiao/dreamscene360/assets/45743512/bbc8da33-48a3-4c4e-a1b8-97b3304e6ef0)

[The Forbidden City.webm](https://github.com/TingtingLiao/dreamscene360/assets/45743512/bf515549-ff38-4091-aee1-9e6148225e33)

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
# text-to-scene generation 
python main.py prompt="a cozy living room"  upscale=True 

# (for <16GB gpu, set upscale=False) 
python main.py prompt="a cozy living room" upscale=False 

# with gui 
python main.py prompt="a cozy living room" upscale=True gui=True 

# image-to-scene 
python main.py input=data/i2p-image.jpg prompt="an office room" upscale=True 
```
## Gallery  

https://github.com/TingtingLiao/dreamscene360/assets/45743512/a910ed06-1316-4782-884c-85ae93582153

https://github.com/TingtingLiao/dreamscene360/assets/45743512/689c5abe-2628-4a36-8a9c-6d3249af6038

https://github.com/TingtingLiao/dreamscene360/assets/45743512/4a4d6fd1-91c3-40d8-b24d-642e5288852c

https://github.com/TingtingLiao/dreamscene360/assets/45743512/0edce697-62e3-459e-a414-3ccc1227cc41
 
https://github.com/TingtingLiao/dreamscene360/assets/45743512/98d81fed-9436-4b54-a72b-0abf51235a26

![panorama_image](https://github.com/TingtingLiao/dreamscene360/assets/45743512/77cf447e-85ee-4320-831d-76865a1ee92e)

![depth_blending_vis](https://github.com/TingtingLiao/dreamscene360/assets/45743512/9972dfe6-344a-4a05-8afb-445a670bf9ac) 

[training_process.webm](https://github.com/TingtingLiao/dreamscene360/assets/45743512/ce82c9b6-0b8c-4f7d-80e1-f2531ff9796b)

## Acknowledgement 
Special thanks to the projects and their contributors:
* [DreamScene360](https://dreamscene360.github.io/)
* [Diffusion360](https://github.com/ArcherFMY/SD-T2I-360PanoImage)
* [360monodepth](https://github.com/manurare/360monodepth)
* [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
* [Equirec2Perspec](https://github.com/fuenwang/Equirec2Perspec)