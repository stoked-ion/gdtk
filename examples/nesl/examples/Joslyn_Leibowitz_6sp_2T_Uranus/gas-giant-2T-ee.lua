-- Daisy J 
-- Jan 2025

Mechanism{
   '(H2) ~~ (H, H+)',
   type = 'V-T',
   rate = 'Landau-Teller',
   relaxation_time = {"Millikan-White", a=9.673, b=0.072500}
}

Mechanism{
   '(H2) ~~ (He, He+)',
   type = 'V-T',
   rate = "Landau-Teller",
   relaxation_time = {"Millikan-White", a=69.971, b=0.004682}
}

Mechanism{
   '(H2) ~~ (H2)',
   type = 'V-T',
   rate = "Landau-Teller",
   relaxation_time = {"Millikan-White", a=65.11, b=0.006821}
}

-- ET Rates from Park 2013, doi:10.2514/1.T3689, fit by Daisy Joslyn
Mechanism{
    "(e-) ~~ (H2)",
    type = "E-T",
    exchange_cross_section = {type="GnoffoNeutral", a = 9.96911e-20, b = 5.29296e-24, c = -1.41167e-28}
}

Mechanism{
    "(e-) ~~ (He)",
    type = "E-T",
    exchange_cross_section = {type="GnoffoNeutral", a = 5.81937e-20, b = 1.55222e-25, c = -1.25311e-29}
}

Mechanism{
    "(e-) ~~ (H)",
    type = "E-T",
    exchange_cross_section = {type="GnoffoNeutral", a = 1.09693e-20, b = -8.77377e-26, c = -4.27648e-31}
}
