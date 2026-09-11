cp ${DGD_REPO}/examples/kinetics/air-chemistry-1T/air-7sp-1T.inp .
cp ${DGD_REPO}/examples/kinetics/air-chemistry-1T/GuptaEtAl-air-reactions.lua .
sed -i -e "s/, 'NO+'//g" -e "s/, 'e-'//g" air-7sp-1T.inp
prep-gas air-7sp-1T.inp air-5sp.gas
prep-chem air-5sp.gas GuptaEtAl-air-reactions.lua air-5sp.chem
