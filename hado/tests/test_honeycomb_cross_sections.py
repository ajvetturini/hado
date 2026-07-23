from hado.core.automation.routing.honeycomb_xsect import select_honeycomb_ring_by_diameter
from hado.core.automation.routing.honeycomb_xsect import set_hollow_honeycomb
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_by_n():
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()
    n_per_bundles = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    for idx, i in enumerate(n_per_bundles):
        ax = axes[idx]
        m = n = i // 2
        evens, odds, _ = set_hollow_honeycomb(m, n, 3.25)
        evens, odds = np.array(evens), np.array(odds)
        ax.scatter(evens[:, 0], evens[:, 1], color='blue', s=10, label='Even')
        ax.scatter(odds[:, 0], odds[:, 1], color='red', s=10, label='Odd')

        ax.set_title(f'{i} nm Ring', fontsize=10)
        ax.axis('equal')

        if idx >= 10:
            ax.set_xlabel('X (nm)')
        if idx % 5 == 0:
            ax.set_ylabel('Y (nm)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95))

    plt.tight_layout()
    plt.show()

def plot_thicknesses():
    already_seen = set()
    points = []

    for i in np.arange(5.0, 37.5, 2.5):
        ring = select_honeycomb_ring_by_diameter(i, 3.25, 2.25)
        m, n = ring['M'], ring['N']
        if (m, n) in already_seen: continue
        already_seen.add((m, n))
        points.append((ring['evens'], ring['odds'], ring['target_diameter'], ring['actual_diameter']))

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    axes = axes.flatten()
    for idx, (evens, odds, target, actual) in enumerate(points):
        ax = axes[idx]
        evens, odds = np.array(evens), np.array(odds)
        ax.scatter(evens[:, 0], evens[:, 1], color='blue', s=10, label='Even')
        ax.scatter(odds[:, 0], odds[:, 1], color='red', s=10, label='Odd')

        ax.set_title(f'Diameter (nm) | Target: {target:.2f} | Actual: {actual:.2f}', fontsize=10)
        ax.axis('equal')

        if idx >= 8:
            ax.set_xlabel('X (nm)')
        if idx % 4 == 0:
            ax.set_ylabel('Y (nm)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot honeycomb cross-sections.')
    parser.add_argument('--plot', choices=['by_n', 'thicknesses'],
                        required=False, help='Which plot to generate', default='thicknesses')
    args = parser.parse_args()

    if args.plot == 'by_n':
        plot_by_n()
    elif args.plot == 'thicknesses':
        plot_thicknesses()
