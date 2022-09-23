#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:21:43 2022

@author: beriksso
"""


'''
Fit NBI/TH/back scatter to DD and DT peak
'''

import basics
import KM11data
import nes
from nes import tofor
from nes.utils import jetplasma
from nes.utils import stix
from nes.utils import spec
from nes.utils import reactions
import numpy as np
from nes.tofor.commands import load_response_function
import useful_defs as udfs
import sys
import matplotlib.pyplot as plt
import useful_defs as udfs
import json
import matplotlib
from nes.utils import jetnbi


def load(file_name, name=None, shot=None, t0=None, t1=None, drf=''):
    # Load data from file
    d = np.load(file_name)

    # Choose bins
    bin_choice = (d['bins'] > -0.1) & (d['bins'] < 199.7)
    data = d['counts'][bin_choice]

    # Load response function
    r = load_response_function(drf)

    if name is None:
        name = file_name

    nes_data = nes.Data(data, name=name, response=r,
                        back=d['bgr_level'],
                        x_label='$t_{TOF}$ (ns)', y_label='counts/bin')

    # JET specific attributes
    nes_data.shot = shot
    nes_data.t0 = t0
    nes_data.t1 = t1

    return nes_data


# D plasma with trace amounts of T and D NBI
shot = 99552

# Decide which DRF/data set to use (kinematic cuts or not)
setting = 0

# Decide which DRF to use (generated using light yield or not)
light_yield = True
suffix = '_ly' if light_yield else ''

'''
0: No kinematic cuts
1: Kinematic cuts with a=1, b=1, c=1
2: Kinematic cuts with a=0.7, b=1.4, c=1.2
3: Kinematic cuts with a=0.7, b=1.4, c=1.2 for data, a=1, but b=1, c=1 for DRF
'''
if setting == 0:
    drf = f'/home/beriksso/NES/drf/tofu_drf{suffix}.json'
    file_name = f'data/{shot}/{shot}_NoKinCut.pickle'
elif setting == 1:
    drf = f'/home/beriksso/NES/drf/tofu_drf_kin{suffix}.json'
    file_name = f'data/{shot}/{shot}_kin.pickle'
elif setting == 2:
    drf = f'/home/beriksso/NES/drf/tofu_drf_scaled_kin{suffix}.json'
    file_name = f'data/{shot}/{shot}_KinCut.pickle'
elif setting == 3:
    drf = f'/home/beriksso/NES/drf/tofu_drf_kin{suffix}.json'
    file_name = f'data/{shot}/{shot}_scaled_kin.pickle'

else:
    raise ValueError(f'Invalid setting: {setting}')

# Get time range
P = udfs.unpickle(file_name)
t0, t1 = P['time_range']

# Load data
data = load(file_name, shot=shot, t0=t0, t1=t1, drf=drf)

data.name = f'JPN {shot}, {t0}-{t1} s.'
tofor.fit.data = data

# Set fitting limits
tofor.fit.data_xlim = (20, 80)

# Disallow rigid shift
tofor.fit.rigid_shift.lock = True

# Calculate D NBI slowed down distribution using Fokker-Planck equation
E, D = jetnbi.get_dist(shot, t0, t1, ion='D')

# Get plasma parameters
ne, Te, B, R0 = jetplasma.get_params(shot, t0, t1)

plasma = stix.plasma.DPlasma(ne, Te, B, R=R0)
Emax = 3000.0
nbi_particle = stix.plasma.deuteron

fp = stix.fokker_planck.FokkerPlanckEquation(Emax, plasma, nbi_particle)
fp.set_NBI(1e19, 100.0)

fp.solve()

# Calculate deuterium NBI on D plasma
dd_reaction = reactions.DDNHe3Reaction()
dt_reaction = reactions.DTNHe4Reaction()
dd_scalc = spec.SpectrumCalculator(dd_reaction)
dt_scalc = spec.SpectrumCalculator(dt_reaction)
dd_scalc.u1 = [0, 0, 1]
dt_scalc.u1 = [0, 0, 1]

# Set reactant velocity distributions
dd_scalc.reactant_a.sample_E_dist(fp.E, fp.dNdE, pitch_range=[0.5, 0.7])
dd_scalc.reactant_b.sample_maxwellian_dist(Te)
dt_scalc.reactant_a.sample_E_dist(E, D, pitch_range=[0.5, 0.7])
dt_scalc.reactant_b.sample_maxwellian_dist(Te)

bt_dd, En_dd = dd_scalc(bin_width=50.0)
bt_dt, En_dt = dt_scalc(bin_width=50.0)

# BT reactivity
sigmav_bt_dd = bt_dd.sum()
sigmav_bt_dt = bt_dt.sum()

# Fit components to TOFu data
# ---------------------------

# Set BT DD component
bt_dd_comp = tofor.fix2
bt_dd_comp.En = En_dd
bt_dd_comp.shape = bt_dd
bt_dd_comp.name = 'NBI (DD)'
bt_dd_comp.use = True

# Set BT DT component
bt_dt_comp = tofor.fix1
bt_dt_comp.En = En_dt
bt_dt_comp.shape = bt_dt
bt_dt_comp.name = 'NBI (DT)'
bt_dt_comp.use = True

# Set thermal component
tofor.thermal.use = True
tofor.thermal.T = Te
tofor.thermal.T.lock = True

tofor.fit.set_startvals()
tofor.fit()
tofor.fit()

tofor.commands.fit_scatter()
tofor.fit.rigid_shift.lock = False
tofor.fit()
tofor.fit()

I_bt_dt0 = bt_dt_comp.N.value
I_bt_dd0 = bt_dd_comp.N.value
nt_over_nd0 = (sigmav_bt_dd / sigmav_bt_dt) * (I_bt_dt0 / I_bt_dd0)


to_save = {}
to_save['I_bt_dt'] = I_bt_dt0
to_save['I_bt_dd'] = I_bt_dd0
to_save['figure'] = tofor.fit.datafig
to_save['time_range'] = [t0, t1]
to_save['n_spectrum'] = {'bt_dt': [bt_dt_comp.En, bt_dt_comp.spectrum],
                         'bt_dd': [bt_dd_comp.En, bt_dd_comp.spectrum],
                         'th_dd': [tofor.thermal.En, tofor.thermal.spectrum],
                         'scatter': [tofor.scatter.En, tofor.scatter.spectrum]}
to_save['tof_spectrum'] = {'bt_dt': [tofor.fit.comp_data['NBI (DT)']],
                           'bt_dd': [tofor.fit.comp_data['NBI (DD)']],
                           'th_dd': [tofor.fit.comp_data['TH (DD)']], 
                           'scatter': [tofor.fit.comp_data['scatter']],
                           'tof_axis': [data.response.to_axis]}

udfs.pickler(f'{shot}_{setting}.pickle', to_save, check=True)
