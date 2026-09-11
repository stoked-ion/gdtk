"""
Compute reaction rates of a gas state using python.

To use this script, you may have install the optional python bindings as follows:

 $ cd ~/gdtk/src/gas
 $ make libgas.so install

And add the following to your .bashrc:

export PYTHONPATH=${PYTHONPATH}:${DGD}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DGD}/lib

@author: Nick Gibbons (August 2026)
"""

from numpy import arange, absolute
from gdtk.gas import GasModel, GasState, ThermochemicalReactor
from copy import copy
import matplotlib.pyplot as plt

def constant_volume_reactor(TInit, pInit, gm, chemUpdate):
    Q = GasState(gm)
    Q.p = pInit
    Q.T = TInit
    total = 2.0 + 1.0 + 3.76
    molef = {"H2":2.0/total, "O2":1.0/total, "N2":3.76/total}
    Q.massf = gm.molef2massf(molef)
    Q.update_thermo_from_pT()

    ts = []
    Ts = []
    ps = []
    Ys = []
    t = 0.0
    dt = 1.0e-6
    dtSuggest = 1.0e-11

    ts.append(t)
    Ts.append(Q.T)
    ps.append(Q.p)
    Ys.append(copy(Q.massf))

    while t <= tFinal:
        dtSuggest = chemUpdate.update_state(Q, dt, dtSuggest)
        t = t + dt
        dt = dtSuggest
        gm:updateThermoFromRHOU(Q)
        Q.update_thermo_from_rhou()

        ts.append(t)
        Ts.append(Q.T)
        ps.append(Q.p)
        Ys.append(copy(Q.massf))
    return ts, Ts, ps, Ys

spFile = "Jachimowski-1992-gas-model.lua"
reacFile = "Jachimowski-1992-reac-file.lua"
outFile = "J92-ignition-delay.dat"

gm = GasModel(spFile)
chemUpdate = ThermochemicalReactor(gm, reacFile)

P_atm = 101.325e3
tFinal = 1200.0e-6 # s
pInit = P_atm*0.5
TInit = 1200.0 # K

# You may already have a sequence of flowstates from CFD or somewhere.
# For this example, we begin by generating some using a constant volume reactor.

ts, Ts, ps, Ys = constant_volume_reactor(TInit, pInit, gm, chemUpdate) 

# Now let's march through the collected flowstates, and use the .source_terms
# method attached to chemUpdate to compute the reactions rates. These are
# reported in kg/m3/s, like they are used in the steady-state solver.

reaction_rates = {sp:list() for sp in gm.species_names}
for T,p,Y in zip(Ts, ps, Ys):
    gs = GasState(gm)
    gs.T = T
    gs.p = p
    gs.massf = copy(Y)
    gs.update_thermo_from_pT()

    source_terms = chemUpdate.source_terms(gs)
    for spname, rate in zip(gm.species_names, source_terms):
        reaction_rates[spname].append(rate)

# Finally let's make a plot of them over time.
fig = plt.figure(figsize=(14,8))
axes0,axes1 = fig.subplots(1,2)
axes0

axes0.plot(ts, Ts, color='red', linewidth=2.0, label='T')
axes0.grid()
axes0.legend(framealpha=1.0)
axes0.set_ylabel('Temperature (K)')
axes0.set_xlabel('Time (sec)')

#rates_to_plot = ['O2', 'H2O', 'NO', 'HNO', 'NO2']
colors = ['red', 'blue', 'green', 'magenta', 'black', 'firebrick', 'darkblue',
           'gold', 'olive', 'purple', 'slategrey', 'teal', 'darkgoldenrod']
for idx,sp in enumerate(reaction_rates.keys()):
    #if sp not in rates_to_plot: continue

    # These can be negative, but we want to look at log space, so take abs here
    abs_sp_rate = [abs(Yi) for Yi in reaction_rates[sp]]
    line = axes1.semilogy(ts, abs_sp_rate, color=colors[idx], linewidth=2.0, label=sp)
    y = abs_sp_rate[-1]
    offset = 6
    axes1.annotate('<'+sp, xy=(1,y), xytext=(offset,0), color=line[0].get_color(), 
                xycoords = axes1.get_yaxis_transform(), textcoords="offset points",
                size=14, va="center")

#ymin, ymax = axes1.get_ylim()
axes1.set_ylim(1e-7, 5e2)
axes1.grid()
axes1.set_ylabel('Reaction Rate (kg/m3/s)')
axes1.set_xlabel('Time (sec)')
plt.show()


