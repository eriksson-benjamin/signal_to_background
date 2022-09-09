#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:24:11 2022

@author: beriksso
"""

import numpy as np
import useful_defs as udfs
import matplotlib.pyplot as plt
udfs.set_nes_plot_style()
import scipy as sp

def import_data(file):
    """Return data."""
    p = udfs.unpickle(file)
    bgr_level = np.append(np.flip(p['bgr_level'][1:]), p['bgr_level'])
    return p['bins'], p['counts'], bgr_level


def plot_data(bins, counts, bgr_level, shot_number):
    """Plot data."""
    plt.figure(shot_number)
    plt.plot(bins, counts, 'k.')
    plt.errorbar(bins, counts, yerr=np.sqrt(counts), color='k', 
                 linestyle='None')
    plt.plot(bins, bgr_level, 'C0--', label='random coincidences')
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('counts')


def gaussian(amplitude, mu, sigma, x, bgr_level):
    """Gaussian function with background."""
    return amplitude * np.exp(-(x - mu)**2 / sigma**2) + bgr_level
    

def fit_function(parameters, bins, counts, bgr_level, fit_range):
    """Fit function for fitting a Gaussian."""
    # Make gaussian
    g = gaussian(*parameters, bins, bgr_level)
    
    # Calculate chi2 over fit range
    mask = ((bins > fit_range[0]) & (bins < fit_range[1]))
    chi2 = np.sum((counts[mask] - g[mask])**2 / g[mask])
    
    return chi2

    
def plot_gaussian(parameters, bins, counts, bgr_level, shot_number):
    """Plot Gaussian and total fit on data."""
    plt.figure(shot_number)
    gauss = gaussian(*parameters, bins, 0)
    total = gaussian(*parameters, bins, bgr_level)
    plt.plot(bins, gauss, 'k-.', label='gaussian fit')
    plt.plot(bins, total, 'r-', label='total')
    plt.legend()
    
    
def signal_to_noise(parameters, bins, counts, bgr_level):
    """Calculate signal to noise ratio for 3 sigma interval."""
    # Make Gaussian without background component
    gauss = gaussian(*parameters, bins, 0)
    
    # Find 3 sigma interval
    mu = parameters[1]
    sigma = parameters[2]
    mask = ((bins > mu - 3*sigma) & (bins < mu + 3*sigma))

    # Integrate trapezoidally
    signal = np.trapz(gauss[mask], dx=np.diff(bins)[0])
    noise = np.trapz(bgr_level[mask], dx=np.diff(bins)[0])
    
    return signal/noise


def plot_for_paper(shot_number):
    """Create plot for technical TOFu paper."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    suffixes = ['NoKinCut', 'KinCut']
    for i, ax in enumerate(axes.flatten()):
        # Import data
        file = f'data/{shot_number}/{shot_number}_{suffixes[i]}.pickle'
        bins, counts, bgr_level = import_data(file)
        
        # Fit Gaussian to DT peak
        initial_guess = (100, 26, 1)
        fit_range = (15, 35)
        popt = sp.optimize.minimize(fit_function, initial_guess, 
                                    args=(bins, counts, bgr_level, fit_range))
        
        # Calculate signal to noise ratio
        print(signal_to_noise(popt.x, bins, counts, bgr_level))
    
        # Plot data
        ax.plot(bins, counts, 'k.', markersize=3)
        ax.errorbar(bins, counts, yerr=np.sqrt(counts), linestyle='None',
                    color = 'k')
        
        
        # Plot total fit
        gauss = gaussian(*popt.x, bins, 0)
        ax.plot(bins, gauss + bgr_level, 'r-', label='total')
        
        # Plot background
        ax.plot(bins, bgr_level, 'C0--', label='background')
        
        # Plot Gaussian fit
        ax.plot(bins, gauss, 'C1-.', label='Gaussian fit')
        
        ax.set_ylabel('counts')
        
        
    # Configure plots
    axes[1].set_xlabel('$t_{TOF}$ (ns)')
    axes[0].set_ylim(-15, 350)
    axes[1].set_xlim(10, 45)
    axes[1].set_ylim(-15, 200)
    axes[0].legend(loc='upper right')
    axes[0].text(0.05, 0.9, '(a)', transform=axes[0].transAxes)
    axes[1].text(0.05, 0.9, '(b)', transform=axes[1].transAxes)
    
    fig.set_size_inches(4, 7)
    plt.subplots_adjust(hspace=0.05)
    
def main(shot_number, suffix):
    """Run analysis for one input."""
    # Import data
    file = f'data/{shot_number}/{shot_number}_{suffix}.pickle'
    bins, counts, bgr_level = import_data(file)
    
    # Plot data
    plot_data(bins, counts, bgr_level, shot_number)
    
    # Fit Gaussian to DT peak
    initial_guess = (100, 26, 1)
    fit_range = (15, 35)
    popt = sp.optimize.minimize(fit_function, initial_guess, 
                                args=(bins, counts, bgr_level, fit_range))
    
    # Plot
    plot_gaussian(popt.x, bins, counts, bgr_level, shot_number)
    
    # Calculate signal to noise ratio
    print(signal_to_noise(popt.x, bins, counts, bgr_level))
    
    
if __name__ == '__main__':
    shot_number = 100850
    main(shot_number, 'KinCut') 
    plot_for_paper(shot_number)
    
    
    
    
