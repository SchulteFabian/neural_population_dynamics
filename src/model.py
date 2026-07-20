#-----------------------
# We implement the Wilson Cowan equations. 
# Given the current states and all parameters, we want to compute the instantaneous change of the states.
#-----------------------

import numpy as np
from dataclasses import dataclass


@dataclass
class WilsonCowanParameters:
    tau_E: float
    tau_I: float
    c1: float
    c2: float
    c3: float
    c4: float
    P: float
    Q: float
    a_E: float
    a_I: float
    theta_E: float
    theta_I: float


def sigmoid(x, a, theta):
    return 1 / (1 + np.exp(-a*(x-theta)))



def wilson_cowan (state, params):
    """
    Compute the Wilson-Cowan dynamics.

    Parameters
    ----------
    state[0] = E : float
        Excitatory population activity.
    state[1] = I : float
        Inhibitory population activity.
    parameters : dict
        Model parameters. tau = time constans, c = weights, P,Q = external inputs, a,theta = sigmoid parameters

    Returns
    -------
    dE : float
        Time derivative of excitatory activity.
    dI : float
        Time derivative of inhibitory activity.
    """
    
    E, I = state

    input_E = params.c1*E - params.c2*I + params.P
    input_I = params.c3*E - params.c4*I + params.Q

    dE = (-E + (1 - E) * sigmoid(input_E, params.a_E, params.theta_E)) / params.tau_E
    dI = (-I + (1 - I) * sigmoid(input_I, params.a_I, params.theta_I)) / params.tau_I

    return np.array([dE, dI])