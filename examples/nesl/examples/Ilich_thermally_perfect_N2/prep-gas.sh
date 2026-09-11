cp ${DGD_REPO}/examples/kinetics/fixed-volume-reactor-n2/nitrogen-2sp.inp .
sed -i "s/,[[:space:]]*'N'//" nitrogen-2sp.inp
prep-gas nitrogen-2sp.inp n2-therm-perf.gas