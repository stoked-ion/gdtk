"""
Python copy of the ignition-delay.lua script

Run using:
 $ prep-gas Jachimowski-1992-species.inp Jachimowski-1992-gas-model.lua
 $ prep-chem Jachimowski-1992-gas-model.lua Jachimowski-1992.lua Jachimowski-1992-reac-file.lua
 $ python3 ignition-delay.py

You may have install the optional python bindings as follows:

 $ cd ~/gdtk/src/gas
 $ make libgas.so install

And add the following to your .bashrc:

export PYTHONPATH=${PYTHONPATH}:${DGD}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DGD}/lib

@author: Nick Gibbons (August 2026)
"""

from numpy import arange
from gdtk.gas import GasModel, GasState, ThermochemicalReactor

class NoIgnitionException(Exception):
    pass

def ignition_delay(T, gm, chemUpdate, igCriteria):
    Q = GasState(gm)
    Q.p = pInit
    Q.T = T
    total = 2.0 + 1.0 + 3.76
    molef = {"H2":2.0/total, "O2":1.0/total, "N2":3.76/total}
    Q.massf = gm.molef2massf(molef)
    Q.update_thermo_from_pT()

    t = 0.0
    dt = 1.0e-6
    dtSuggest = 1.0e-11
    while t <= tFinal:
       dtSuggest = chemUpdate.update_state(Q, dt, dtSuggest)
       t = t + dt
       dt = dtSuggest
       gm:updateThermoFromRHOU(Q)
       Q.update_thermo_from_rhou()
       conc = Q.conc_as_dict
       if conc['OH'] > igCriteria:
           return t
    raise NoIgnitionException("No ignition at T={:f} in {:f} ms".format(T,t/1e-3))

def sweep_temperatures(gm, chemUpdate, Ts, igCriteria):

    igTemps = []
    igDelays = []

    for T in Ts:
        try:
            tIg = ignition_delay(T, gm, chemUpdate, igCriteria)
            igTemps.append(T)
            igDelays.append(tIg)
        except(NoIgnitionException) as e:
            print(e)
    return igTemps, igDelays

if __name__=='__main__':
    spFile = "Jachimowski-1992-gas-model.lua"
    reacFile = "Jachimowski-1992-reac-file.lua"
    outFile = "J92-ignition-delay.dat"

    P_atm = 101.325e3
    tFinal = 1500.0e-6 # s
    pInit = P_atm

    Tlow = 900.0       # K
    Thigh = 1300.0     # K
    dT = 10.0
    Ts = arange(Tlow, Thigh+1.0, dT)

    igCriteria = 5.0e-3 # mol/m^3 : OH

    gm = GasModel(spFile)
    chemUpdate = ThermochemicalReactor(gm, reacFile)
    igTemps, igDelays = sweep_temperatures(gm, chemUpdate, Ts, igCriteria)

    with open(outFile, 'w') as f:
        f.write('# 1:T(K)  2:t(s)\n')
        for T, tIg in zip(igTemps, igDelays):
            f.write("{:20.12e} {:20.12e}\n".format(T, tIg))
    print("Done.")

