lmr prep-chem -g gas-giant-2T-6sp-mt.gas -i gas-giant-2T-rr.lua -o gas-giant-2T-6sp.chem
lmr prep-energy-exchange -g gas-giant-2T-6sp-mt.gas -r gas-giant-2T-6sp.chem -i gas-giant-2T-ee.lua -o gas-giant-2T-6sp.kin

cat extra-mechanisms.kin >> gas-giant-2T-6sp.kin