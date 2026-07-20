from model import WilsonCowanParameters, wilson_cowan

params = WilsonCowanParameters(
    tau_E = 1, # time constants
    tau_I= 1,
    c1 = 10, # weight
    c2 = 10,
    c3 = 10,
    c4 = 10,
    P = 0, # external inputs
    Q = 0,
    a_E = 1, # sigmoid parameters
    a_I = 1,
    theta_E = 0,
    theta_I = 0
    )

