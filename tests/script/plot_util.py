from matplotlib import pyplot as plt
import numpy as np

from tests.script import utils_ext


def plot_power_spectrum(power_spectrum, max_l=6):
    bar_width = 0.9
    gap = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    l_values = list(range(max_l + 1))
    x_positions = np.array(l_values) * (1 + gap)
    powers = [power_spectrum.get(l, 0) for l in l_values]

    ax.bar(x_positions, powers, bar_width, label='Power Spectrum')

    ax.set_xlabel('l', fontsize=20)
    ax.set_ylabel('Power Spectrum \( P_{nl} \)', fontsize=20)
    ax.set_title('Power Spectrum of Spherical Harmonics Coefficients', fontsize=20)
    ax.set_xticks(x_positions, fontsize=20)
    ax.set_xticklabels(l_values, fontsize=20)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_coefficients(coefficients, max_l=6, total_l=20):
    # Prepare data
    l_values = list(range(total_l + 1))  # Calculate all l from 0 to 20
    m_values_by_l = {l: [] for l in l_values}
    coeff_values_by_l = {l: [] for l in l_values}

    # Extract and aggregate coefficients by summing over all n
    coeff_sum = {}  # Store the sum of coefficients for each (l, m)
    for (n, l, m), coeff in coefficients.items():
        if l <= total_l:
            key = (l, m)
            if key not in coeff_sum:
                coeff_sum[key] = 0
            coeff_sum[key] += coeff  # Accumulate coefficients for the same (l, m) across all n

    # Organize data, use only the first max_l + 1 l for plotting
    for (l, m), coeff in coeff_sum.items():
        if l <= max_l:  # Process only l within plotting range
            m_values_by_l[l].append(m)
            coeff_values_by_l[l].append(coeff)

    # Set bar plot parameters
    bar_width = 0.5  # Increased bar width
    gap = 0.1  # Reduced gap between l values
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for the first max_l + 1 l values
    for i, l in enumerate(l_values[:max_l + 1]):  # Plot only the first 6 l
        m_vals = np.array(m_values_by_l[l])
        coeffs = np.array(coeff_values_by_l[l])
        if len(m_vals) == 0:
            continue
        # Calculate x positions with gaps
        x_positions = m_vals + (i * (2 * max_l + 1 + gap))
        ax.bar(x_positions, coeffs, bar_width, label=f'l={l}')

    # Set plot properties
    ax.set_xlabel('m')
    ax.set_ylabel('Summed Coefficient Magnitude')
    ax.set_title('Aggregated Spherical Harmonics Coefficients by l and m (First 6 l)')
    ax.legend()

    # Adjust x-axis ticks
    all_m_positions = []
    all_m_labels = []
    for i, l in enumerate(l_values[:max_l + 1]):
        m_range = np.arange(-l, l + 1)
        x_pos = m_range + (i * (2 * max_l + 1 + gap))
        all_m_positions.extend(x_pos)
        all_m_labels.extend([str(m) for m in m_range])

    ax.set_xticks(all_m_positions)
    ax.set_xticklabels(all_m_labels, rotation=45)

    plt.tight_layout()
    plt.show()

from matplotlib import cm
def plot_bispectrum(bispectrum, max_l=5):
    """
    Plot the aggregated bispectrum values using subplots for each l.

    Args:
        bispectrum: Dictionary with (l, l1, l2) as key and bispectrum value
        max_l: Maximum l value to consider for plotting (default=5)
    """
    # Prepare data for plotting
    l_values = range(max_l + 1)
    data = np.zeros((max_l + 1, max_l + 1, max_l + 1))
    for (l, l1, l2), value in bispectrum.items():
        if l <= max_l and l1 <= max_l and l2 <= max_l:
            data[l, l1, l2] = value

    # Create subplots for each l
    fig, axes = plt.subplots(max_l + 1, 1, figsize=(10, 5 * (max_l + 1)), sharex=True, sharey=True)
    if max_l == 0:
        axes = [axes]  # Ensure axes is a list for single subplot case

    for l in l_values:
        ax = axes[l]
        im = ax.imshow(data[l], cmap=cm.viridis, aspect='auto', extent=[-0.5, max_l + 0.5, -0.5, max_l + 0.5])
        ax.set_title(f'L = {l}')
        ax.set_xlabel('L2')
        ax.set_ylabel('L1')
        plt.colorbar(im, ax=ax, label='Bispectrum Value')

    plt.tight_layout()
    plt.show()


def plot_two_bispectra(bispectrum1, bispectrum2, max_l=5):
    """
    Plot two bispectra side by side for visual comparison.

    Args:
        bispectrum1: First bispectrum dictionary.
        bispectrum2: Second bispectrum dictionary.
        max_l: Maximum l value to consider.
    """
    l_values = range(max_l + 1)
    data1 = np.zeros((max_l + 1, max_l + 1, max_l + 1))
    data2 = np.zeros((max_l + 1, max_l + 1, max_l + 1))

    for (l, l1, l2), value in bispectrum1.items():
        if l <= max_l and l1 <= max_l and l2 <= max_l:
            data1[l, l1, l2] = value

    for (l, l1, l2), value in bispectrum2.items():
        if l <= max_l and l1 <= max_l and l2 <= max_l:
            data2[l, l1, l2] = value

    fig, axes = plt.subplots(max_l + 1, 2, figsize=(12, 4 * (max_l + 1)), sharex=True, sharey=True)

    for l in l_values:
        im1 = axes[l, 0].imshow(data1[l], cmap=cm.viridis, aspect='auto',
                                extent=[-0.5, max_l + 0.5, -0.5, max_l + 0.5])
        axes[l, 0].set_title(f'Bispectrum before rotation — L = {l}')
        axes[l, 0].set_xlabel('L2')
        axes[l, 0].set_ylabel('L1')
        plt.colorbar(im1, ax=axes[l, 0])

        im2 = axes[l, 1].imshow(data2[l], cmap=cm.viridis, aspect='auto',
                                extent=[-0.5, max_l + 0.5, -0.5, max_l + 0.5])
        axes[l, 1].set_title(f'Bispectrum after rotation — L = {l}')
        axes[l, 1].set_xlabel('L2')
        axes[l, 1].set_ylabel('L1')
        plt.colorbar(im2, ax=axes[l, 1])

    fig.suptitle("Bispectrum comparison", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.985])  # leave space for title
    plt.show()


def plot_bispectra_path(keys_to_plot, bispectra_by_key, n_frames, title):
    # frames = np.arange(1, n_frames + 1)  # base 1
    # fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
    #
    # # for key in keys_to_plot:
    # #     values = bispectra_by_key[key]
    # #     values_real = [v.real for v in values]
    # #     label = rf"$B_{{{key[0]}{key[1]}{key[2]}}}$"
    # #     axes[0].plot(frames, values_real, marker='o', label=label)
    # #
    # # axes[0].hlines(0, 1, n_frames, colors='k', linestyles='--', linewidth=1)
    # # axes[0].set_ylabel("Real Part")
    # # axes[0].set_title("Real Part of Bispectrum")
    # # axes[0].grid(True)
    # # axes[0].legend()
    #
    #
    #
    # for key in keys_to_plot:
    #     values = bispectra_by_key[key]  # 直接使用，无需取 .real
    #     values_imag = [v.imag for v in values]
    #     label = rf"$B_{{{key[0]}{key[1]}{key[2]}}}$"
    #     axes[0].plot(frames, values_imag, marker='o', label=label)
    #
    # axes[0].hlines(0, 1, n_frames, colors='k', linestyles='--', linewidth=1)
    # axes[0].set_ylabel("Imaginary Part")
    # axes[0].set_xlabel("Frame Index")
    # axes[0].set_title("Bispectrum")
    # axes[0].grid(True)
    # axes[0].legend()
    #
    # fig.suptitle(title, fontsize=16)
    #
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    #
    # plt.show()


    plt.figure(figsize=(10, 5))
    frames = np.arange(1, n_frames + 1)  # base 1
    for key in keys_to_plot:
        values = bispectra_by_key[key]
        values_imag = [v.imag for v in values]
        label = rf"$B_{{{key[0]}{key[1]}{key[2]}}}$"
        plt.plot(frames, values_imag, marker='o', label=label)

    plt.hlines(0, 1, n_frames, colors='k', linestyles='--', linewidth=3)

    plt.xlabel("Frame Index", fontsize=24)
    plt.ylabel("Bispectrum Value", fontsize=24)
    plt.title(title, fontsize=24)
    # plt.tick_params("both", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.grid(True)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_spherical_harmonics_comparison(coeffs_before, coeffs_after, max_l=5, total_l=20):
    # Prepare storage
    l_values = list(range(total_l + 1))
    coeff_sum_before = {}
    coeff_sum_after = {}

    # Aggregate (l, m) coefficients
    for (n, l, m), coeff in coeffs_before.items():
        if l <= total_l:
            coeff_sum_before[(l, m)] = coeff_sum_before.get((l, m), 0) + coeff

    for (n, l, m), coeff in coeffs_after.items():
        if l <= total_l:
            coeff_sum_after[(l, m)] = coeff_sum_after.get((l, m), 0) + coeff

    # Set up plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # Plot settings
    group_gap = 3.0  # gap between l blocks
    bar_width = 1.2
    xticks = []
    xticklabels = []

    position_cursor = 0

    for l in range(max_l + 1):
        m_range = np.arange(-l, l + 1)
        num_m = len(m_range)

        # Compute positions for each m in this l block
        x_positions = position_cursor + np.arange(num_m)
        position_cursor = x_positions[-1] + group_gap

        # Collect labels
        xticks.extend(x_positions)
        xticklabels.extend([str(m) for m in m_range])
        # Get coefficients
        coeffs_before_vals = [coeff_sum_before.get((l, m), 0) for m in m_range]
        coeffs_after_vals = [coeff_sum_after.get((l, m), 0) for m in m_range]

        # Plot bars
        ax1.bar(x_positions, coeffs_before_vals, width=bar_width, label=fr'$\ell$={l}')
        ax2.bar(x_positions, coeffs_after_vals, width=bar_width, label=fr'$\ell$={l}')

    # Styling for ax1
    ax1.set_title("Before Rotation", fontsize=22)
    ax1.set_xlabel("m", fontsize=30)
    ax1.set_ylabel("Summed Coefficient Magnitude", fontsize=20)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, rotation=45, ha='center', fontsize=18)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Styling for ax2
    ax2.set_title("After Rotation", fontsize=22)
    ax2.set_xlabel("m", fontsize=30)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, rotation=45, ha='center', fontsize=18)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


def calculate_aggregated_power_spectrum(coefficients, max_l=20):
    # Calculate aggregated power spectrum summing over m and n for each l
    power_by_l = {}
    for (n, l, m), coeff in coefficients.items():
        if l <= max_l:
            if l not in power_by_l:
                power_by_l[l] = 0
            power_by_l[l] += np.real(np.conjugate(coeff) * coeff)  # Sum over m and n
    return power_by_l


def plot_power_spectrum_comparison(coeffs_before, coeffs_after, max_l=5, total_l=20):
    # Calculate aggregated power spectrum for both sets
    power_before = calculate_aggregated_power_spectrum(coeffs_before, total_l)
    power_after = calculate_aggregated_power_spectrum(coeffs_after, total_l)

    # Prepare data for plotting, use only the first max_l + 1 l
    l_values = list(range(max_l + 1))
    powers_before = [power_before.get(l, 0) for l in l_values]
    powers_after = [power_after.get(l, 0) for l in l_values]

    # Set up subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot before rotation
    x_positions = np.array(l_values) * 1.1
    ax1.bar(x_positions, powers_before, 0.9, label='Power Spectrum')
    ax1.set_xlabel(r'$\ell$', fontsize=20)
    ax1.set_ylabel(r'Aggregated Power Spectrum $P_\ell$', fontsize=20)
    ax1.set_title('Before Rotation', fontsize=20)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(l_values, fontsize=20)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.legend(fontsize=20)

    # Plot after rotation
    ax2.bar(x_positions, powers_after, 0.9, label='Power Spectrum')
    ax2.set_xlabel(r'$\ell$', fontsize=20)
    ax2.set_title('After Rotation', fontsize=20)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(l_values, fontsize=20)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.show()




