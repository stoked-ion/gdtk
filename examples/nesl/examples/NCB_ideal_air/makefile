prep:
	cp ${DGD_REPO}/examples/lmr/2D/laminar-flat-plate/ideal-air.lua .
	prep-gas ideal-air.lua air-ideal.gas

run:
	nesl_condition_builder --job=nesl_config.toml --bounds=ncb_config.toml --max_cpus=12

clean:
	rm -f ideal-air.lua air-ideal.gas
	rm -rf runs