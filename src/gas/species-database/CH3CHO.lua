db.CH3CHO = {}
db.CH3CHO.atomicConstituents = {C=2,H=4,O=1,}
db.CH3CHO.charge = 0
db.CH3CHO.M = {
   value = 44.052560e-3,
   units = 'kg/mol',
   description = 'molecular mass',
   reference = 'Periodic table'
}
db.CH3CHO.gamma = {
   value = 1.1762e00,
   units = 'non-dimensional',
   description = 'ratio of specific heats at 300.0K',
   reference = 'evaluated using Cp/R from Chemkin-II coefficients'
}
db.CH3CHO.sigma = {
   value = 3.970,
   units = 'Angstrom',
   description = 'Lennard-Jones potential distance',
   reference = 'GRI-Mech 3.0 transport file.'
}
db.CH3CHO.epsilon = {
   value = 436.000,
   units = 'K',
   description = 'Lennard-Jones potential well depth.',
   reference = 'GRI-Mech 3.0 transport file.'
}
db.CH3CHO.grimechThermoCoeffs = {
   notes = 'data from GRIMECH 3.0',
   nsegments = 2,
   T_break_points = {200.0, 1000.0, 6000.0},
   T_blend_ranges = {400.0},
   segment0 = {
      0,
      0,
      0.47294595E+01,
     -0.31932858E-02,
      0.47534921E-04,
     -0.57458611E-07,
      0.21931112E-10,
     -0.21572878E+05,
      0.41030159E+01,
   },
   segment1 = {
      0,
      0,
      0.54041108E+01,
      0.11723059E-01,
     -0.42263137E-05,
      0.68372451E-09,
     -0.40984863E-13,
     -0.22593122E+05,
     -0.34807917E+01,
   }
}
db.CH3CHO.ceaThermoCoeffs = {
   notes = 'NASA/TP—2002-211556',
   nsegments = 2,
   T_break_points = {200.0, 1000.0, 6000.0},
   T_blend_ranges = {400.0},
   segment0 = {
     -1.373904e+05,
      2.559938e+03,
     -1.340470e+01,
      5.922129e-02,
     -6.240006e-05,
      3.703324e-08,
     -9.342697e-12,
     -3.318731e+04,
      1.007418e+02,
   },
   segment1 = {
      3.321177e+06,
     -1.449720e+04,
      2.708421e+01,
     -2.879320e-03,
      5.556310e-07,
     -5.732675e-11,
      2.443965e-15,
      6.507756e+04,
     -1.536236e+02
    }
}
db.CH3CHO.chemkinViscosity = {
   notes = 'Generated by species-generator.py',
   nsegments = 1, 
   segment0 ={
      T_lower = 200.000,
      T_upper = 6000.000,
      A = -2.464943462372e+01,
      B = 3.963220206531e+00,
      C = -3.676625277378e-01,
      D = 1.349561598291e-02,
   }
}
db.CH3CHO.chemkinThermCond = {
   notes = 'Generated by species-generator.py',
   nsegments = 1, 
   segment0 ={
      T_lower = 200.000,
      T_upper = 6000.000,
      A = -2.472743688385e+01,
      B = 5.846139483528e+00,
      C = -4.559596622026e-01,
      D = 1.169834480741e-02,
   }
}

