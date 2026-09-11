cp ${DGD_REPO}/examples/kinetics/air-chemistry-2T/air-5sp-gas-model.lua .
cp ${DGD_REPO}/examples/kinetics/air-chemistry-2T/Park-air-reactions-2T.lua .
cp ${DGD_REPO}/examples/kinetics/air-chemistry-2T/Park-air-energy-exchange-2T.lua .
prep-gas air-5sp-gas-model.lua air-5sp.gas
sed -i '/-- Associative ionization reactions/,$d' Park-air-reactions-2T.lua
prep-chem air-5sp.gas Park-air-reactions-2T.lua air-5sp.chem
sed -i -e '/-- ET Rates from Gnoffo/,$d' -e 's/,[A-Z0-9]*+//g' Park-air-energy-exchange-2T.lua
prep-kinetics air-5sp.gas air-5sp.chem Park-air-energy-exchange-2T.lua air-5sp.kin
