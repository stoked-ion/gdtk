-- Dissociation rates from Leibowitz paper (1973)
-- All else from Leibowitz & Kuo paper (1976)
-- Author: Daisy-May Joslyn

Config{
   tempLimits = {lower=300.0, upper=50000.0}
}
energyModes = {["vibroelectronic"]=0}
-- Arrhenius reaction rates are evaluated using the translation temperature only:
-- return A*pow(T, n)*exp(-C/T);


-- Leibowitz, 1973

Reaction{'H2 + H+ <=> H + H + H+',
   fr={'Park', A=8.34e+19, n=-1.0, C=51958, s=0.5}
}

Reaction{'H2 + He <=> H + H + He',
   fr={'Park', A=4.17e+18, n=-1.0, C=51958, s=0.5}
}

Reaction{'H2 + H2 <=> H + H + H2',
   fr={'Park', A=1.04e+19, n=-1.0, C=51958, s=0.5}
}

Reaction{'H2 + H <=> H + H + H',
   fr={'Park', A=8.34e+19, n=-1.0, C=51958, s=0.5}
}

Reaction{'H2 + e- <=> H + H + e-',
   fr={'Arrhenius', A=8.34e+19, n=-1.0, C=51958, s=0.5, rateControllingTemperature="vibroelectronic"},
   br={'fromEqConst', rateControllingTemperature="vibroelectronic"}
}

Reaction{'H2 + He+ <=> H + H + He+',
   fr={'Park', A=4.17e+18, n=-1.0, C=51958, s=0.5}
}

-- Leibowitz & Kuo, 1976

Reaction{'H + e- <=> H+ + e- + e-',
   fr={'Arrhenius', A=2.28e+13, n=0.5, C=157800, rateControllingTemperature="vibroelectronic"},
   br={'fromEqConst', rateControllingTemperature="vibroelectronic"}
}

Reaction{'He + e- <=> He+ + e- + e-',
   fr={'Arrhenius', A=1.33e+13, n=0.5, C=285200, rateControllingTemperature="vibroelectronic"},
   br={'fromEqConst', rateControllingTemperature="vibroelectronic"}
}

Reaction{'H + H <=> H+ + e- + H',
   fr={'Arrhenius', A=6.17e+10, n=0.5, C=116100,},
   br={'fromEqConst', rateControllingTemperature="vibroelectronic"}
}

-- disabling this as it was causing issues earlier apparently and there shouldn't be much ionisation anyway
-- Reaction{'H + He <=> H+ + e- + He',
--    fr={'Arrhenius', A=4.88e+10, n=0.5, C=116100},
--    br={'fromEqConst', rateControllingTemperature="vibroelectronic"}
-- }

-- below rates are from the two-stage excitation reaction 
Reaction{'H + e- <=> H+ + e- + e-',
   fr={'Arrhenius', A=4.12e+13, n=0.5, C=116000, rateControllingTemperature="vibroelectronic"},
   br={'fromEqConst', rateControllingTemperature="vibroelectronic"}
}

Reaction{'He + e- <=> He+ + e- + e-',
   fr={'Arrhenius', A=2.25e+13, n=0.5, C=232100, rateControllingTemperature="vibroelectronic"},
   br={'fromEqConst', rateControllingTemperature="vibroelectronic"}
}
