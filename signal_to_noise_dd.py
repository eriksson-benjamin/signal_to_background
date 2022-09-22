#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:10:32 2022

@author: beriksso
"""

"""
Calculate signal-to-noise ratio for DD peak using NES fit components.
"""


import sys
sys.path.insert(0, 'C:/python/useful_definitions/')
import useful_defs as udfs
import matplotlib.pyplot as plt


def import_data(shot_number):
    """Import TOFu and NES data for shot number."""
    nes_1 = f'nes_fits/{shot_number}/{shot_number}_KinCut.pickle'
    nes_2 = f'nes_fits/{shot_number}/{shot_number}_NoKinCut.pickle'
    tofu_1 = f'data/{shot_number}/{shot_number}_KinCut.pickle'
    tofu_2 = f'data/{shot_number}/{shot_number}_NoKinCut.pickle'
    
    # Read input data    
    n1 = udfs.unpickle(nes_1)
    plt.close('data')
    n2 = udfs.unpickle(nes_2)
    plt.close('data')
    t1 = udfs.unpickle(tofu_1)
    t2 = udfs.unpickle(tofu_2)
    
    return n1, n2, t1, t2

shot_number = 99552
n1, n2, t1, t2 = import_data(shot_number)

