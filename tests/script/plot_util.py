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


def plot_power_spectrum_comparison(coeffs_before, coeffs_after, max_l=5, total_l=20):
    # Calculate aggregated power spectrum for both sets
    power_before = utils_ext.calculate_aggregated_power_spectrum(coeffs_before, total_l)
    power_after = utils_ext.calculate_aggregated_power_spectrum(coeffs_after, total_l)

    # Prepare data for plotting, use only the first max_l + 1 l
    l_values = list(range(max_l + 1))
    powers_before = [power_before.get(l, 0) for l in l_values]
    powers_after = [power_after.get(l, 0) for l in l_values]

    # Set up subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot before rotation
    x_positions = np.array(l_values) * 1.1
    ax1.bar(x_positions, powers_before, 0.5, label='Power Spectrum')
    ax1.set_xlabel('l')
    ax1.set_ylabel('Aggregated Power Spectrum \( P_l \)')
    ax1.set_title('Before Rotation')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(l_values)
    ax1.legend()

    # Plot after rotation
    ax2.bar(x_positions, powers_after, 0.5, label='Power Spectrum')
    ax2.set_xlabel('l')
    ax2.set_title('After Rotation')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(l_values)
    ax2.legend()

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
        axes[l, 0].set_title(f'Bispectrum 1 — L = {l}')
        axes[l, 0].set_xlabel('L2')
        axes[l, 0].set_ylabel('L1')
        plt.colorbar(im1, ax=axes[l, 0])

        im2 = axes[l, 1].imshow(data2[l], cmap=cm.viridis, aspect='auto',
                                extent=[-0.5, max_l + 0.5, -0.5, max_l + 0.5])
        axes[l, 1].set_title(f'Bispectrum 2 — L = {l}')
        axes[l, 1].set_xlabel('L2')
        axes[l, 1].set_ylabel('L1')
        plt.colorbar(im2, ax=axes[l, 1])

    plt.tight_layout()
    plt.show()


def plot_bispectra_path(keys_to_plot, bispectra_by_key, n_frames, title):
    plt.figure(figsize=(10, 5))
    frames = np.arange(1, n_frames + 1)  # base 1
    for key in keys_to_plot:
        values = bispectra_by_key[key]  # 直接使用，无需取 .real
        label = rf"$B_{{{key[0]}{key[1]}{key[2]}}}$"
        plt.plot(frames, values, marker='o', label=label)

    # ✅ 添加水平 0 线
    plt.hlines(0, 0, n_frames - 1, colors='k', linestyles='--', linewidth=1)

    # ✅ 图形标签与样式
    plt.xlabel("Frame Index")
    plt.ylabel("Bispectrum Value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()