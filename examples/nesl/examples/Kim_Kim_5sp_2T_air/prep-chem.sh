cp ${DGD_REPO}/examples/kinetics/air-chemistry-2T/air-5sp-gas-model.lua .
prep-gas air-5sp-gas-model.lua air-5sp.gas
prep-chem air-5sp.gas rr-kim-air5-2T.inp air-5sp.chem
prep-kinetics air-5sp.gas air-5sp.chem ee-kim-air5-2T.inp air-5sp.kin
