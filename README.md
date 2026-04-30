# Astrocook

Python scripts to generate 2D and 3D textures for Unity out of various astrophysics data formats  

**klodufy.py** creates 3D textures for Unity out of uniform cube density maps (voxel clouds)  
**particles_textufy.py** creates text files with rows of particles data for Unity to transform into 2D textures  

## Usage

- Install Python 3.13 (max verison supported by *sarracen* package as of 2025-10)  
- Install *numpy*, *scipy*, *sarracen* and *h5py* packages running `pip install numpy`, `pip install scipy`, `pip install sarracen` and `pip install h5py`  
- Run `py klodufy.py` or `python klodufy.py` depending on your main Python CLI call.  
- The *data* directory contains sources, while *output* contains your exported text files.  
- The *data* directory is left empty, for you to fill it with relevant data files.  
- Adapt the script by commenting or uncommenting revelant code sections.  
- Edit file according to your needs.   

## Note for SHAMROCK users

- /!\ Warning: install sarracen DEVELOPMENT build  
- As of early 2026, sarracen.read_shamrock doesn't exist in stable build  
- install sarracen dev build with "pip install git+https://github.com/ttricco/sarracen.git"  

## Supported file types

- PHANTOM dumps, SHAMROCK dumps, RAMSES dumps (HDF5 or Numpy arrays), TXT files, SAV files, just write to me if you have something else. As long as there is a Python reader this should be fairly quick to read.  

## General note

- This quick-and-dirty homemade code needs improvement for better usability! Any suggestion is welcome.  
- Have a nice data cooking!  
