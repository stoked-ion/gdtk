# Makefile for crbl.
# Taine Rossini (t.rossini@uq.edu.au) - 7-May-2026

INSTALL_DIR ?= $(DGD)
MAIN_DIR := $(DGD_REPO)
LMR_DIR := $(MAIN_DIR)/src/lmr
EQC_MAIN_DIR = $(HOME)/equilibrium-c/

install : $(INSTALL_DIR)
	@echo "Installing gas and kinetics library..."
	$(MAKE) -C $(LMR_DIR) install

	@echo "Installing equilibrium-c library..."
	$(MAKE) -C $(EQC_MAIN_DIR)/source install

	@echo "Copying crbl to $(INSTALL_DIR)/bin..."
	cp -f crbl_exec.py $(INSTALL_DIR)/bin/crbl
	cp -f crbl.py $(INSTALL_DIR)/lib/crbl.py	

	@echo "Creating virtual environment..."
	python3 -m venv $(INSTALL_DIR)/lib/.crbl
        
	@echo "Installing Python dependencies..."
	$(INSTALL_DIR)/lib/.crbl/bin/pip install --upgrade pip
	$(INSTALL_DIR)/lib/.crbl/bin/pip install -r requirements.txt
	
	@echo "Making crbl program executable..."
	chmod +x $(INSTALL_DIR)/bin/crbl
		
	@echo "crbl is now installed."

clean : 
	- $(MAKE) -C $(LMR_DIR) clean
	- $(MAKE) -C $(EQC_MAIN_DIR)/source clean
	rm -f $(INSTALL_DIR)/bin/crbl
	rm -f $(INSTALL_DIR)/lib/crbl.py   
	rm -rf $(INSTALL_DIR)/lib/.crbl

$(INSTALL_DIR):
	- mkdir $(INSTALL_DIR)
