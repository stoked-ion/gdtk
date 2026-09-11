# STRATOS: Installation Guide

_Taine J. Rossini_

Version 1.0.0.

---

The following paths are used in the `makefile`, if yours are different, please edit the `makefile` accordingly. It is assumed that the user has already appended the `.bashrc` file according to the instructions in the `gdtk` installation guide.

```
INSTALL_DIR ?= $(DGD)
MAIN_DIR := $(DGD_REPO)
```

---

It is recommended to clean the `gdtk/src/gas` library before building.

```
cd $DGD_REPO/src/gas
make clean
```

---

`stratos` can then be built and installed by running the following commands,

```
cd $DGD_REPO/examples/stratos
make install
```

---

The local `Python` environment also requires installation of some packages. On `Linux` this can be performed using the following commands,

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy cffi matplotlib
```
