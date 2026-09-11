# Makefile for nesl.
# Taine Rossini (t.rossini@uq.edu.au) - 20-March-2026

INSTALL_DIR ?= $(DGD)
MAIN_DIR := $(DGD_REPO)
LMR_DIR := $(MAIN_DIR)/src/lmr
EQC_MAIN_DIR = $(HOME)/equilibrium-c/
EQC_INSTALL_DIR = $(HOME)/eqcinst/

install : $(INSTALL_DIR)
	@echo "Installing gas and kinetics library..."
	$(MAKE) -C $(LMR_DIR) install

	@echo "Installing equilibrium-c library..."
	$(MAKE) -C $(EQC_MAIN_DIR)/source install
	cd $(EQC_MAIN_DIR)/examples && cp -f normalshock.py $(EQC_INSTALL_DIR)

	@echo "Copying nesl to $(INSTALL_DIR)/bin..."
	cp -f nesl.py $(INSTALL_DIR)/bin/nesl
	cp -f nesl_condition_builder.py $(INSTALL_DIR)/bin/nesl_condition_builder

	@echo "Copying nesl_utils folder to $(INSTALL_DIR)/lib..."
	cp -r nesl_utils $(INSTALL_DIR)/lib

	@echo "Creating virtual environment..."
	python3 -m venv $(INSTALL_DIR)/lib/.nesl
        
	@echo "Installing Python dependencies..."
	$(INSTALL_DIR)/lib/.nesl/bin/pip install --upgrade pip
	$(INSTALL_DIR)/lib/.nesl/bin/pip install -r requirements.txt
	
	@echo "Compiling Cython dependencies..."
	cd $(INSTALL_DIR)/lib/nesl_utils/ && $(INSTALL_DIR)/lib/.nesl/bin/python cythonbuild.py build_ext --inplace

	@echo "Making nesl program executable..."
	chmod +x $(INSTALL_DIR)/bin/nesl
	chmod +x $(INSTALL_DIR)/bin/nesl_condition_builder
		
	@echo "nesl is now installed."

clean : 
	- $(MAKE) -C $(LMR_DIR) clean
	- $(MAKE) -C $(EQC_MAIN_DIR)/source clean
	rm -rf $(INSTALL_DIR)/lib/nesl_utils
	rm -f $(INSTALL_DIR)/bin/nesl   
	rm -rf $(INSTALL_DIR)/lib/.nesl

$(INSTALL_DIR):
	- mkdir $(INSTALL_DIR)