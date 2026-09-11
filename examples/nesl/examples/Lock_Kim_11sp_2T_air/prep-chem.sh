cp ${DGD_REPO}/examples/eilmer/2D/fireII/kim2021/ee-kim-air11-2T.inp .
cp ${DGD_REPO}/examples/eilmer/2D/fireII/kim2021/rr-kim-air11-2T.inp .
cp ${DGD_REPO}/examples/eilmer/2D/fireII/kim2021/gm-air11-2T.inp .
prep-gas gm-air11-2T.inp air-11sp.gas
prep-chem air-11sp.gas rr-kim-air11-2T.inp air-11sp.chem
prep-kinetics air-11sp.gas air-11sp.chem ee-kim-air11-2T.inp air-11sp.kin
