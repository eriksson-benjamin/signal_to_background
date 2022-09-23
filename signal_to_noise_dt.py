#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:24:11 2022

@author: beriksso
"""

import numpy as np
import sys
sys.path.insert(0, 'C:/python/useful_definitions/')
import useful_defs as udfs
import matplotlib.pyplot as plt
udfs.set_nes_plot_style()
import scipy as sp


def import_data(file):
    """Return data."""
    p = udfs.unpickle(file)
    bgr_level = np.append(np.flip(p['bgr_level'][1:]), p['bgr_level'])
    return p['bins'], p['counts'], bgr_level


def plot_data(bins, counts, bgr_level, title):
    """Plot data."""
    plt.figure(title)
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


def plot_gaussian(parameters, bins, counts, bgr_level, title):
    """Plot Gaussian and total fit on data."""
    plt.figure(title)
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
    mask = ((bins > mu - 3 * sigma) & (bins < mu + 3 * sigma))

    # Integrate trapezoidally
    signal = np.trapz(gauss[mask], dx=np.diff(bins)[0])
    noise = np.trapz(bgr_level[mask], dx=np.diff(bins)[0])

    return signal / noise


def plot_for_paper_dt(shot_number):
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

        # Plot data
        ax.plot(bins, counts, 'k.', markersize=3)
        ax.errorbar(bins, counts, yerr=np.sqrt(counts), linestyle='None',
                    color='k')

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


def plot_for_paper_dd(shot_number):
    """Create plot for technical TOFu paper."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    suffixes = ['NoKinCut', 'KinCut']
    for i, ax in enumerate(axes.flatten()):
        # Import data
        file = f'data/{shot_number}/{shot_number}_{suffixes[i]}.pickle'
        bins, counts, bgr_level = import_data(file)

        # Fit Gaussian to DD peak
        initial_guess = (100, 63.4, 1)
        fit_range = (57, 71)

        popt = sp.optimize.minimize(fit_function, initial_guess,
                                    args=(bins, counts, bgr_level, fit_range))

        # Plot data
        ax.plot(bins, counts, 'k.', markersize=3)
        ax.errorbar(bins, counts, yerr=np.sqrt(counts), linestyle='None',
                    color='k')

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
    axes[0].set_ylim(-50, 2000)
    axes[1].set_xlim(35, 80)
    axes[1].set_ylim(-50, 1700)
    axes[0].legend(loc='center left')
    axes[0].text(0.05, 0.9, '(a)', transform=axes[0].transAxes)
    axes[1].text(0.05, 0.9, '(b)', transform=axes[1].transAxes)

    fig.set_size_inches(4, 7)
    plt.subplots_adjust(hspace=0.05)


def plot_for_paper(shot_number):
    """Create plot for technical TOFu paper."""
    fig, axes = plt.subplots(3, 1)
    
    suffixes = ['NoKinCut', 'KinCut']
    
    
    # Plot full TOF spectrum
    file = f'data/{shot_number}/{shot_number}_{suffixes[0]}.pickle'
    bins, counts, bgr_level = import_data(file)
    
    axes[0].plot(bins, counts, 'k.', markersize=1)
    axes[0].errorbar(bins, counts, yerr=np.sqrt(counts), linestyle='None', 
                     color='k')
    
    
    for i, ax in enumerate(axes[1:]):
        # Import data
        file = f'data/{shot_number}/{shot_number}_{suffixes[i]}.pickle'
        bins, counts, bgr_level = import_data(file)

        # Fit Gaussian to DT peak
        initial_guess = (100, 26, 1)
        fit_range = (15, 35)

        popt = sp.optimize.minimize(fit_function, initial_guess,
                                    args=(bins, counts, bgr_level, fit_range))

        # Plot data
        ax.plot(bins, counts, 'k.', markersize=3)
        ax.errorbar(bins, counts, yerr=np.sqrt(counts), linestyle='None',
                    color='k')

        # Plot total fit
        gauss = gaussian(*popt.x, bins, 0)
        ax.plot(bins, gauss + bgr_level, 'r-', label='total')

        # Plot background
        ax.plot(bins, bgr_level, 'C0--', label='background')

        # Plot Gaussian fit
        ax.plot(bins, gauss, 'C1-.', label='Gaussian fit')

        ax.set_ylabel('counts')

    # Labels
    axes[2].set_xlabel('$t_{TOF}$ (ns)')
    axes[1].legend(loc='upper right')
    
    # Limits
    axes[0].set_yscale('log')
    axes[0].set_ylim(100, 4000)
    axes[1].set_ylim(-15, 350)
    axes[2].set_ylim(-15, 200)
    axes[0].set_xlim(10, 90)
    axes[1].set_xlim(10, 45)
    axes[2].set_xlim(10, 45)
    
    # Letters    
    axes[0].text(0.03, 0.9, '(a)', transform=axes[0].transAxes)
    axes[1].text(0.03, 0.9, '(b)', transform=axes[1].transAxes)
    axes[2].text(0.03, 0.9, '(c)', transform=axes[2].transAxes)
    
    # Lines
    axes[0].axvline(20, color='k', linestyle='dotted')
    axes[0].axvline(33, color='k', linestyle='dotted')
    
    fig.set_size_inches(4, 10)
    plt.subplots_adjust(hspace=0.15)

def main(shot_number, suffix):
    """Run analysis for one input."""
    # Import data
    file = f'data/{shot_number}/{shot_number}_{suffix}.pickle'
    bins, counts, bgr_level = import_data(file)

    # Plot data
    plot_data(bins, counts, bgr_level, suffix)

    # Fit Gaussian to DT peak
    initial_guess = (100, 26.4, 1)
    fit_range = (15, 35)
    
#    initial_guess = (100, 63.4, 1)
#    fit_range = (57, 71)

    popt = sp.optimize.minimize(fit_function, initial_guess,
                                args=(bins, counts, bgr_level, fit_range))

    # Plot
    plot_gaussian(popt.x, bins, counts, bgr_level, suffix)

    # Calculate signal to noise ratio
    sb = signal_to_noise(popt.x, bins, counts, bgr_level)

    return sb


if __name__ == '__main__':
    shot_number = 100850
#    shot_number = 99552
    sb_a = main(shot_number, 'KinCut')
    sb_b = main(shot_number, 'NoKinCut')

    print(f'S/B (a): {sb_a:.3f}')
    print(f'S/B (b): {sb_b:.3f}')
    print(f'Improvement: {sb_a/sb_b:.3f}')
    plot_for_paper_dt(shot_number)
    plot_for_paper_dd(shot_number)
    plot_for_paper(shot_number)