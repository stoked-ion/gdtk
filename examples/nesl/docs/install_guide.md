# NESL and NCB: Installation Guide

_Taine J. Rossini_

Version 1.0.0.

---

The user must have Python3 headers available in order to build `nesl` and `ncb`. On Ubuntu, this can be done by running the following command,

```
sudo apt update && sudo apt install python3-dev
```

---

The `makefile` assumes that the user has already installed the `equilibrium-c` library in their home directory. If not, it can be installed as follows,

```
cd ~
git clone https://github.com/uqngibbo/equilibrium-c/
```

---

Additionally, the following paths are used in the `makefile`, if yours are different, please edit the `makefile` accordingly. It is assumed that the user has already appended the `.bashrc` file according to the instructions in the `gdtk` installation guide.

```
INSTALL_DIR ?= $(DGD)
MAIN_DIR := $(DGD_REPO)
EQC_MAIN_DIR = $(HOME)/equilibrium-c/
EQC_INSTALL_DIR = $(HOME)/eqcinst/
```

---

It is recommended to clean the `gdtk/src/gas` library before building.

```
cd $DGD_REPO/src/gas
make clean
```

---

`nesl` can then be built and installed by running the following commands,

```
cd $DGD_REPO/examples/nesl
make install
```
