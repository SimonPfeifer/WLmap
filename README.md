# WLmap
Making weak lensing maps from cosmological simulations.

## Getting Started

### Prerequisites

You need to have a Python2.7 installation with the following libraries:

```
numpy 1.15.4
h5py 2.8.0
```

`WLmap` was written to work with the BAHAMAS simulation and thus with HDF5 snapshot output from GADGET-3. It uses a python package called `readEagle` to read in the simulation data. It is available on the [EAGLE wiki](http://eagle.strw.leidenuniv.nl/wiki/doku.php?id=start) which you most likely have access to if you already have access to the simulations.

If you don't have access to `readEagle` or want to modify `WLmap` to work with different simulation data, you will only need to modify a few lines in the functions `__init__()` and `read_snapshot()`.

### Usage

There is a working example in the script `example.py` which should explain the basic functionality. In brief, the main workflow is as follows:

- Initialise; set path to simulation data, lightcone parameters, cosmology, etc.
- Generate lightcone/lense planes
- Generate convergence map
- Generate shear maps
- Save lightcone and maps to seperate HDF5 files
