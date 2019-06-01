
# **TomoLab** : TomographyLab software for tomographic vision

This project is a Python 3 porting of our previous project `Occiput.io`, which is now no longer supported
___________

**TomoLab** is (or will be) a tomographic reconstruction software for PET, PET-MRI and SPECT in 2D, 3D (volumetric) and 4D (spatio-temporal) for Python 3.x.

The software provides high-speed reconstruction using Graphics Processing Units (GPU).
***Note***: *an NVidia CUDA-compatible GPU is required.*  

`TomoLab` can be utilized with arbitrary scanner geometries. It can be utilized for abstract tomographic reconstruction experiments to develop new algorithms and explore new system geometries, or to connect to real-world scanners,  providing production quality image reconstruction with standard (MLEM, OSEM, Ordinary Poisson OSEM) and advanced algorithms.

`TomoLab` implements algorithms for motion correction (direct motion estimation), kinetic imaging, multi-modal reconstruction, respiratory and cardiac gated imaging.
The source code contains Jupyter notebooks with examples.

## Installation
Given the early stage of this project, no installation instruction are currently available, as a lot of the code originally written for Python 2.7 still needs to be ported to Python 3.x

If you want to try out `TomoLab` while it is being developed, we provide a (devlopment) Docker Image, build to natively support current version of `TomoLab`. For more information about this, please refer to [*TomographyLab/DockerImage*](https://github.com/TomographyLab/DockerImage).

If, instead, you would like to install `TomoLab` directly in your system, you can (for the time being) have a look at this [DockerFile](https://github.com/TomographyLab/DockerImage/blob/master/config/occiput.Dockerfile) to see what the main dependencies are.

If you have troubles with any of these steps, please just open an Issue here and we will try to sort it out.

## Getting started

Examples and demos of the features of `TomoLab` are in the `/tomolab/Examples` folder.
A better documentation, and instruction about the best order in which you can study those notebooks will come (hopefully) very soon.

## Website

For more information check out our [website](http://tomographylab.scienceontheweb.net/): it is still based on `Occiput.io`, previous version of this project, but it should still be a valid starting point to understand the ideas behind this project, and to access some of the publications produced thanks to it.

----

#### Current status of the porting of Occiput.io (python 2.7) to TomoLab (python 3.x)


- **Reorganization of the code**
  - [x] integrating major (`Occiput.io`'s) dependencies within the main `TomoLab` project
  - [x] switching to relative imports throughout the code
  - [ ] consistently following PEP8 style rules
  - [ ] choosing a naming convention for modules, classes and variable (in therm of Captialization, usage of underscores, and so on) and keeping it consistent


- **Simulation**
  - [x] Several synthetic phantoms ready to be generated. A set of routines allow to create complex geometries (which may also be combined together by addition or subtraction), specifying the desired size and shape.
  - [x] Prepared a documentation notebook to showcase synthetic phantom generation capabilities
  - [x] Python 3.x interfaces to projection and backprojection operation successfully built on top of NityRec low level (C++, CUDA) libraries.


- **PET reconstruction**
  - [x] Static reconstruction using OSEM and MLEM
  - [ ] Implementing basic smoothing prior for OSL-MAP-OSEM
  - [ ] Dynamic Reconstruction
  - [ ] Class for efficiently handling 2D+t reconstruction (for research purpose)
  - [ ] Cyclic Scan Reconstruction, informed of motion information coming from MR vNAV data


- **MR reconstruction**
  - [ ] Everything still needs to be checked after moving from python 2 to python 3


- **SPECT reconstruction**
  - [ ] Everything still needs to be checked after moving from python 2 to python 3


- **CT reconstruction**
  - [ ] CT reconstruction is not yet available. Anyway, it should be straightforward to leverage PE ray-tracing system for CT reconstruction, in the next future.


- **Image registration**
  - [ ]


- **(PET and DCE-MRI) Kinetic modeling**
  - [ ]
