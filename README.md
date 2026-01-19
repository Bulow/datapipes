# Datapipes


## Installation

### Automatic (Windows)

Choose one:

#### Python only
1. Make sure you have an Nvidia GPU
2. Run `install.bat`

#### Python, including Matlab Integration
1. Make sure you have an Nvidia GPU
2. Run `install_datapipes_for_matlab.bat`

### "Manual" (Linux, Windows)

1. Install uv
2. Then in an empty folder, run `uv tool install --python 3.12 git+https://github.com/Bulow/datapipes`
3. Install matlab wrapper (Optional): run `datapipes init-matlab`


### Getting started
##### Python
Create new python project using datapipes: `datapipes init` 
Open quickstart.ipynb (preferably in visual studio code)

##### Matlab
If you installed the Matlab integration, a new library called MatDatapipes will have been added to your Matlab path in the default MATLAB library folder.
Any Matlab project now has access to the MatDatapipes library. Open quickstart.m to get started.