prep:
	bash prep-chem.sh
	nesl --grid=job.toml

run:
	nesl --job=job.toml

clean:
	rm -f gm-air11-2T.inp ee-kim-air11-2T.inp rr-kim-air11-2T.inp 
	rm -f air-11sp-gas-model.lua Park-air-reactions-2T.lua Park-air-energy-exchange-2T.lua air-11sp.gas air-11sp.chem air-11sp.kin 
	rm -f grid.vtk *.png *.pdf
