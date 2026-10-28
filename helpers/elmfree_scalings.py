import numpy as np


def determine_fpenalty(state, regime_flag):
    """
    Determine the penalty factor (f_penalty) based on the specified regime_flag.
    This function returns the appropriate f_penalty value for the given regime.

    Scaling is from Nelson et al., 2026
    """

    if regime_flag in ['I-mode', 'EDA H-mode', 'L-mode']:
        if regime_flag == 'I-mode':
            f_penalty = 0.72
        elif regime_flag == 'EDA H-mode':
            f_penalty = 0.61
        elif regime_flag == 'L-mode':
            f_penalty = 0.84
    else:
        if regime_flag == 'QCE':
            a_regime = 0.52
        elif regime_flag == 'WP-QH':
            a_regime = 0.80
        elif regime_flag == 'QH':
            a_regime = 0.81
        elif regime_flag == 'RMP':
            a_regime = 0.80
        elif regime_flag == 'small-ELM':
            a_regime = 0.65
        elif regime_flag == 'ELMy':
            a_regime = 0.68
        f_penalty = a_regime - 0.16*state.betan[0] + 0.22*np.log10(nustar_e_ped)

    if regime_flag in ['I-mode', 'EDA H-mode', 'L-mode', 'QCE']:
        raise ValueError(f"Unsupported or invalid regime_flag: {regime_flag}.")

    return f_penalty

def elmfree_scaling(state, regime_flag):
    """
    Apply ELM-free regime scaling to the pedestal pressure.
    This function modifies the pedestal pressure based on the specified scaling laws for ELM-free regimes.

    Scaling is from Nelson et al., 2026
    """

    f_penalty = determine_fpenalty(state, regime_flag)



