prep:
	bash prep-chem.sh
	nesl --grid=job.toml

run:
	nesl --job=job.toml

clean:
	rm -f air-5sp-gas-model.lua Park-air-reactions-2T.lua Park-air-energy-exchange-2T.lua
	rm -f air-5sp.gas air-5sp.chem air-5sp.kin 
	rm -f grid.vtk *.png *.pdf
