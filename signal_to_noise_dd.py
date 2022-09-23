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
from nes.tofor.commands import load_response_function
import numpy as np


def import_data(fit_path, data_path):
    """Import TOFu and NES data for shot number."""
    
    # Read fit data
    F = udfs.unpickle(fit_path)
    plt.close('data')
    
    # Read TOFu data
    D = udfs.unpickle(data_path)
    
    # Save bins/counts/background level
    bgr = np.append(np.flip(D['bgr_level'][1:]), D['bgr_level'])
    data = (D['bins'], D['counts'], bgr)
    
    # Save fit components
    ft = (np.array(F['tof_spectrum']['bt_dd']) + 
          np.array(F['tof_spectrum']['bt_dt']) + 
          np.array(F['tof_spectrum']['th_dd']) + 
          np.array(F['tof_spectrum']['scatter'])).squeeze()
    fit = (np.array(F['tof_spectrum']['tof_axis']).squeeze(), ft)
    
    return data, fit
    

def plot_data(title, data, fit, ylim):
    """Plot TOFu data with NES fits."""
    plt.figure(title)
    
    # Plot data
    plt.plot(data[0], data[1], 'k.', markersize=1)
    plt.errorbar(data[0], data[1], yerr=np.sqrt(data[1]), linestyle='None', 
                 color='k')
    
    # Plot total fit
    plt.plot(fit[0], fit[1]+data[2][499:], 'r-', label='total')
    
    # Plot background
    plt.plot(fit[0], data[2][499:], 'C0--', label='background')
    
    # Plot signal
    plt.plot(fit[0], fit[1], 'C1-.', label='signal')
    
    # Configure plot
    plt.xlim(45, 80)
    plt.ylim(ylim)
    
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('counts')
    plt.legend()
    
    
def calculate_sb(data, fit, xlim):
    """Calculate signal-to-background ratio."""
    # Get signal/background components
    y_sig = fit[1]
    x_sig = fit[0]
    
    y_bgr = data[2]
    x_bgr = data[0]
    
    # Calculate integral for given x limits
    bool_bgr = ((x_bgr >= xlim[0]) & (x_bgr <= xlim[1]))
    bool_sig = ((x_sig >= xlim[0]) & (x_sig <= xlim[1]))
    
    i_sig = np.trapz(y_sig[bool_sig], x=x_sig[bool_sig])
    i_bgr = np.trapz(y_bgr[bool_bgr], x=x_bgr[bool_bgr])
    
    return i_sig/i_bgr
    

def _plot_for_paper(shot_number):
    """Plot TOFu data with NES fits."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    
    # Import data with kinematic cuts
    suffix = 'NoKinCut'    
    data_path = f'data/{shot_number}/{shot_number}_{suffix}.pickle'
    fit_path = f'nes_fits/{shot_number}/{shot_number}_{suffix}.pickle'
    dat_1, fit_1 = import_data(fit_path, data_path)
    
    # Import data without kinematic cuts

    suffix = 'KinCut'
    data_path = f'data/{shot_number}/{shot_number}_{suffix}.pickle'
    fit_path = f'nes_fits/{shot_number}/{shot_number}_{suffix}.pickle'
    dat_2, fit_2 = import_data(fit_path, data_path)
    
    dats = (dat_1, dat_2)
    fits = (fit_1, fit_2)
    for ax, data, fit in zip(axes.flatten(), dats, fits):
        # Plot data
        ax.plot(data[0], data[1], 'k.', markersize=1)
        ax.errorbar(data[0], data[1], yerr=np.sqrt(data[1]), linestyle='None', 
                     color='k')
        
        # Plot total fit
        ax.plot(fit[0], fit[1]+data[2][499:], 'r-', label='total')
        
        # Plot background
        ax.plot(fit[0], data[2][499:], 'C0--', label='background')
        
        # Plot signal
        ax.plot(fit[0], fit[1], 'C1-.', label='signal')
    
    # Configure plot
    fig.set_size_inches(4, 7)
    plt.subplots_adjust(hspace=0.05)
    
    # Limits
    plt.xlim(45, 80)
    axes[0].set_ylim(0, 500)
    axes[1].set_ylim(0, 250)
    
    # Labels
    axes[1].set_xlabel('$t_{TOF}$ (ns)')
    axes[0].set_ylabel('counts')
    axes[1].set_ylabel('counts')
    axes[0].legend(loc='upper right')
    
    # Letters
    axes[0].text(0.03, 0.9, '(a)', transform=axes[0].transAxes)
    axes[1].text(0.03, 0.9, '(b)', transform=axes[1].transAxes)
    

def plot_for_paper(shot_number):
    """Plot TOFu data with NES fits."""
    fig, axes = plt.subplots(3, 1)
    
    # Import data with kinematic cuts
    suffix = 'NoKinCut'    
    data_path = f'data/{shot_number}/{shot_number}_{suffix}.pickle'
    fit_path = f'nes_fits/{shot_number}/{shot_number}_{suffix}.pickle'
    dat_1, fit_1 = import_data(fit_path, data_path)
    
    # Import data without kinematic cuts
    suffix = 'KinCut'
    data_path = f'data/{shot_number}/{shot_number}_{suffix}.pickle'
    fit_path = f'nes_fits/{shot_number}/{shot_number}_{suffix}.pickle'
    dat_2, fit_2 = import_data(fit_path, data_path)
    
    # Plot full TOF spectrum
    axes[0].plot(dat_1[0], dat_1[1], 'k.', markersize=1)
    axes[0].errorbar(dat_1[0], dat_1[1], yerr=np.sqrt(dat_1[1]), 
        linestyle='None', color='k')
    
    dats = (dat_1, dat_2)
    fits = (fit_1, fit_2)
    for ax, data, fit in zip(axes[1:], dats, fits):
        # Plot data
        ax.plot(data[0], data[1], 'k.', markersize=1)
        ax.errorbar(data[0], data[1], yerr=np.sqrt(data[1]), linestyle='None', 
                     color='k')
        
        # Plot total fit
        ax.plot(fit[0], fit[1]+data[2][499:], 'r-', label='total')
        
        # Plot background
        ax.plot(fit[0], data[2][499:], 'C0--', label='background')
        
        # Plot signal
        ax.plot(fit[0], fit[1], 'C1-.', label='signal')
    
    # Configure plot
    fig.set_size_inches(4, 10)
    plt.subplots_adjust(hspace=0.15)
    
    # Limits
    axes[0].set_yscale('log')
    axes[0].set_xlim(10, 80)
    axes[1].set_xlim(53, 75)
    axes[2].set_xlim(53, 75)
    axes[0].set_ylim(100, 4000)
    axes[1].set_ylim(0, 600)
    axes[2].set_ylim(0, 250)
    
    # Vertical lines
    axes[0].axvline(53, color='k', linestyle='dotted')
    axes[0].axvline(75, color='k', linestyle='dotted')
    
    # Labels
    axes[2].set_xlabel('$t_{TOF}$ (ns)')
    axes[0].set_ylabel('counts')
    axes[1].set_ylabel('counts')
    axes[2].set_ylabel('counts')
    axes[1].legend(loc='upper right')
    
    # Letters
    axes[0].text(0.03, 0.9, '(a)', transform=axes[0].transAxes)
    axes[1].text(0.03, 0.9, '(b)', transform=axes[1].transAxes)
    axes[2].text(0.03, 0.9, '(c)', transform=axes[2].transAxes)

def main(shot_number):
    """Calculate signal-to-background ratio and plot."""
    # Import data with kinematic cuts
    suffix = 'KinCut'
    data_path = f'data/{shot_number}/{shot_number}_{suffix}.pickle'
    fit_path = f'nes_fits/{shot_number}/{shot_number}_{suffix}.pickle'
    dat_1, fit_1 = import_data(fit_path, data_path)
    
    # Import data without kinematic cuts
    suffix = 'NoKinCut'
    data_path = f'data/{shot_number}/{shot_number}_{suffix}.pickle'
    fit_path = f'nes_fits/{shot_number}/{shot_number}_{suffix}.pickle'
    dat_2, fit_2 = import_data(fit_path, data_path)
    
    # Plot
    plot_data(f'{shot_number} - 1', dat_1, fit_1, (0, 250))
    plot_data(f'{shot_number} - 2', dat_2, fit_2, (0, 500))
    
    # Calculate signal-to-background ratio
    sb_1 = calculate_sb(dat_1, fit_1, (57, 72))
    sb_2 = calculate_sb(dat_2, fit_2, (57, 72))
    print(f'S/B 1: {sb_1}')
    print(f'S/B 2: {sb_2}')
    print(f'Improvement: {sb_1/sb_2}')


if __name__ == '__main__':
    shot_number = 99552

    main(shot_number)
    plot_for_paper(shot_number)
    
    
    
    
    




