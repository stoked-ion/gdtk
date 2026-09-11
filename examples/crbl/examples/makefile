prep:
	bash prep-chem.sh

run_bl_import:
	python3 test_bl_import.py

run_vi_import:
	python3 test_vi_import.py

run_bl:
	crbl --bl=test_bl.py

run_vi:
	crbl --vi=test_vi.py

post:
	paraview output.vts

clean:
	rm -f air-7sp-1T.inp GuptaEtAl-air-reactions.lua air-7sp.gas air-7sp.chem
	rm -f output.vts
	rm -f loads.dat
	rm -f *.png