# Makefile for stratos.
# Taine Rossini (t.rossini@uq.edu.au) - 18-May-2026

INSTALL_DIR ?= $(DGD)
MAIN_DIR := $(DGD_REPO)
LMR_DIR := $(MAIN_DIR)/src/lmr

install : $(INSTALL_DIR)
	@echo "Installing gas and kinetics library..."
	$(MAKE) -C $(LMR_DIR) install

	@echo "Copying stratos to $(INSTALL_DIR)/lib..."
	cp -r stratos.py $(INSTALL_DIR)/lib	

clean : 
	- $(MAKE) -C $(LMR_DIR) clean
	rm -f $(INSTALL_DIR)/lib/stratos.py   

$(INSTALL_DIR):
	- mkdir $(INSTALL_DIR)
