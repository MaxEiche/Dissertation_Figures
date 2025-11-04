import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
from pathlib import Path  # add at the top if not already imported

def add_kde_contours(ax, x, y, probs=(0.2, 0.4, 0.6, 0.8), gridsize=300, color='k',
                     alpha=0.9, linewidths=(2.0, 1.5, 1.0), bw_method=0.7, pad=0.10):
    """
    KDE density contours enclosing a given probability mass for (x,y).
    - probs: lower values -> tighter/smaller contours (e.g., 0.3, 0.5, 0.7)
    - bw_method: KDE bandwidth (None | float | 'scott' | 'silverman'). 1.0 ~ default; >1 smoother.
    - pad: fraction of data range to pad the grid by on all sides (prevents open contours at edges)
    """
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return

    xm, xM = np.nanmin(x[m]), np.nanmax(x[m])
    ym, yM = np.nanmin(y[m]), np.nanmax(y[m])
    # pad the grid a bit so contours can close inside the plot
    xr = xM - xm if xM > xm else 1.0
    yr = yM - ym if yM > ym else 1.0
    xm -= pad * xr; xM += pad * xr
    ym -= pad * yr; yM += pad * yr

    xx = np.linspace(xm, xM, gridsize)
    yy = np.linspace(ym, yM, gridsize)
    X, Y = np.meshgrid(xx, yy)

    kde = gaussian_kde(np.vstack([x[m], y[m]]), bw_method=bw_method)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    z = Z.ravel()
    if not np.isfinite(z).any() or np.nanmin(Z) == np.nanmax(Z):
        return  # flat or invalid surface

    # Build levels for requested cumulative probs (higher prob -> lower density threshold)
    z_sorted = np.sort(z)[::-1]
    cdf = np.cumsum(z_sorted); cdf /= cdf[-1]
    levels = []
    for p in sorted(probs):
        idx = np.searchsorted(cdf, p)
        thr = z_sorted[idx] if idx < len(z_sorted) else z_sorted[-1]
        levels.append(float(thr))

    # Strictly increasing, unique levels for matplotlib.contour
    levels = np.sort(np.unique(levels))
    if levels.size == 0 or not np.isfinite(levels).all():
        return

    ax.contour(X, Y, Z, levels=levels, colors=color, linewidths=linewidths, alpha=alpha)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the Excel file
file_path = r'Y:\Members\MaxE\Thesis\Data for Figures\Figure_6\ME_A2-4 Data.xlsx'
xls = pd.ExcelFile(file_path)

# Load the data from 'Plate 1'
plate_df = pd.read_excel(xls)

# Load the treatment data from the corresponding treatment sheet
treatment_sheet = 'Treatment'
treatment_df = pd.read_excel(xls, treatment_sheet, index_col=0)

# Initialize a dictionary to hold the extracted data
extracted_data = {'Replicate': [], 'Day': [], 'g': [], "r'": [], 't': []}

# Extract the replicate names and the relevant columns for 'g', 'r'' and 't'
replicate_columns = range(3, plate_df.shape[1], 5)
g_columns = range(3, plate_df.shape[1], 5)
r_prime_columns = range(5, plate_df.shape[1], 5)
t_columns = range(6, plate_df.shape[1], 5)

# Extract days from the first column (skip header rows)
days = plate_df.iloc[3:, 0].dropna().astype(int).unique()

# Loop through the columns to extract the data
for rep_col, g_col, r_prime_col, t_col in zip(replicate_columns, g_columns, r_prime_columns, t_columns):
    replicate = plate_df.iloc[1, rep_col]
    for day in days:
        day_data = plate_df.loc[plate_df.iloc[:, 0] == day]
        if not day_data.empty:
            extracted_data['Replicate'].append(replicate)
            extracted_data['Day'].append(day)
            extracted_data['g'].append(day_data.iloc[0, g_col])
            extracted_data["r'"].append(day_data.iloc[0, r_prime_col])
            extracted_data['t'].append(day_data.iloc[0, t_col])
        else:
            print(f"No data found for day: {day}")

# Convert the dictionary to a DataFrame
extracted_df = pd.DataFrame(extracted_data)


##### Get rid of bleedthrough
# Define the fraction to subtract from the green value
fraction = 0.06
# Iterate through the extracted_df and correct the green values
extracted_df['g'] = extracted_df.apply(lambda row: row['g'] - fraction * row["r'"], axis=1)
##### Get rid of bleedthrough

# Normalize the extracted values based on the 't' value of each replicate at Day 1
normalized_data = {'Replicate': [], 'Day': [], 'g_normalized': [], "r'_normalized": [], 't_normalized': []}

# Extract the baseline 't' values for normalization
baseline_t_values = extracted_df[extracted_df['Day'] == 2].set_index('Replicate')['t']

# Loop through the extracted data and normalize
for _, row in extracted_df.iterrows():
    replicate = row['Replicate']
    day = row['Day']
    baseline_t = baseline_t_values.get(replicate, None)
    if baseline_t is not None:
        g_normalized = row['g'] / baseline_t
        r_prime_normalized = row["r'"] / baseline_t
        t_normalized = row['t'] / baseline_t

        normalized_data['Replicate'].append(replicate)
        normalized_data['Day'].append(day)
        normalized_data['g_normalized'].append(g_normalized)
        normalized_data["r'_normalized"].append(r_prime_normalized)
        normalized_data['t_normalized'].append(t_normalized)

# Convert the normalized data to a DataFrame
normalized_df = pd.DataFrame(normalized_data)

# Exclude day 32 from the DataFrame
normalized_df = normalized_df[normalized_df['Day'] != 32]

# Define the new treatments with varying replicate counts
treatments = {
    'No treatment': ['A7', 'B7', 'C7', 'D7', 'E7', 'F7'],
    'MTD': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
    'Red Calibration MTD': ['A8', 'B8', 'C8'],
    'Red Calibration No treatment': ['D8', 'E8', 'F8'],
    'AT100': ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2', 'E1', 'E2'],
    'AT80': ['A3', 'A4', 'B3', 'B4', 'C3', 'C4', 'D3', 'D4', 'E3', 'E4'],
    'AT50': ['A5', 'A6', 'B5', 'B6', 'C5', 'C6', 'D5', 'D6', 'E5', 'E6']
}

# Calculate the maximum number of replicates in any treatment to determine the number of rows
max_replicates = max(len(replicates) for replicates in treatments.values())

# Calculate number of columns for a more quadratic layout
num_cols = len(treatments)
num_rows = max_replicates

### Further analysis

# Combine the data with the treatment information
treatment_df2 = pd.read_excel(xls, treatment_sheet)

def combine_normalized_with_treatment(normalized_df, treatment_df):
    # Extract the relevant columns from the treatment dataframe
    treatment_days = treatment_df.iloc[:, 0].dropna().unique()
    replicate_columns = treatment_df.columns[2:]

    # Create a dictionary to map (Day, Replicate) to treatment value
    treatment_dict = {}
    for index, row in treatment_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in treatment_days:
            continue
        day = int(day_str[1:])  # Assuming the day values are formatted like 'd-2', 'd-1', etc.
        for replicate in replicate_columns:
            treatment_dict[(day, replicate)] = row[replicate]

    # Add the new 'treatment' column to the normalized_df dataframe
    normalized_df['treatment'] = normalized_df.apply(
        lambda row: treatment_dict.get((row['Day'], row['Replicate']), 0), axis=1
    )

    return normalized_df

combined_df = combine_normalized_with_treatment(normalized_df, treatment_df2)

###################

def correlate_red_growth(df1):

    df = df1.copy()

    # Add a prefix to the 'Replicate' column in each dataframe
    df['Replicate'] = df['Replicate'].apply(lambda x: '1' + x)

    # Calculate the cumulative sum of 'treatment' for each 'Replicate'
    df['cumulative_treatment'] = df.groupby((df['treatment'] != df['treatment'].shift(1)).cumsum()).apply(
        lambda x: ((x['treatment'] == 1) | (x['treatment'] == 4)).cumsum() + (x['treatment'] == 3).cumsum() * 2
    ).reset_index(drop=True)

    # Shift the 'cumulative_treatment' column by one day
    df['cumulative_treatment'] = df.groupby('Replicate')['cumulative_treatment'].shift(fill_value=0)

    # Calculate the relative change in 'r_normalized' for each 'Replicate'
    df['relative_change'] = df.groupby('Replicate')[r"r'_normalized"].pct_change(fill_method=None) + 1

    # Absolute change of red normalized intensity per day
    df['absolute_change'] = df.groupby('Replicate')[r"r'_normalized"].diff(periods=1)

    # Keep only the necessary columns
    df = df[['Replicate', 'Day', r"r'_normalized", 't_normalized', 'g_normalized', 'treatment', 'cumulative_treatment', 'relative_change', 'absolute_change']]

    # Remove rows where 'treatment' is 0
    df = df[~df['treatment'].isin([0.0])]

    # Remove rows with NaN values in the 'relative_change' and 'absolute_change' column
    df = df.dropna(subset=['relative_change', 'absolute_change'])

    # Exclude rows with 'relative_change' greater than 2 or less than -2 # Remove outliers??
    #df = df[df['relative_change'].between(0.75, 2)]

    # === Diagnostic block: check for zero absolute changes ===
    zero_mask = df['absolute_change'] == 0
    num_zero = zero_mask.sum()
    num_total = df['absolute_change'].notna().sum()
    print(f"[Debug] Absolute-change zeros: {num_zero} / {num_total} = {100 * num_zero / num_total:.1f}%")

    # List of replicates to include
    valid_replicates = [
        '1A1', '1A2', '1A3', '1A4', '1A5', '1A6', '1A7',
        '1B1', '1B2', '1B3', '1B4', '1B5', '1B6', '1B7',
        '1C1', '1C2', '1C3', '1C4', '1C5', '1C6', '1C7',
        '1D1', '1D2', '1D3', '1D4', '1D5', '1D6', '1D7',
        '1E1', '1E2', '1E3', '1E4', '1E5', '1E6', '1E7',
        '1F1', '1F2', '1F3', '1F4', '1F5', '1F6', '1F7'
    ]

    # Filter the DataFrame to include only the valid replicates
    df = df[df['Replicate'].isin(valid_replicates)]

    # Create a scatter plot with 'g_normalized' on the x-axis and 'relative_change' on the y-axis
    plt.figure(figsize=(12, 6))

    # Parameter to control jittering
    jitter = False

    # Add jittering to the data
    if jitter:
        jittered_t_normalized = df['g_normalized'] + np.random.normal(0, 0.1, size=len(df))
        jittered_relative_change = df['relative_change'] + np.random.normal(0, 0.1, size=len(df))
    else:
        jittered_t_normalized = df['g_normalized']
        jittered_relative_change = df['relative_change']

    scatter = plt.scatter(jittered_t_normalized, jittered_relative_change, c=df['Day'], cmap='viridis', s=5)
    plt.colorbar(scatter, label='Day')
    plt.xlabel('g_normalized')
    plt.ylabel('Relative Growth rate of red intensity in a day')
    plt.title('Relative Change vs g_normalized')
    plt.ylim([0.8, 1.8])  # Set y-axis limits

    # Fit a line to the jittered data
    m, b = np.polyfit(jittered_t_normalized, jittered_relative_change, 1)
    plt.plot(jittered_t_normalized, m * jittered_t_normalized + b, color='red',
             alpha=0.5)  # Make the line semi-transparent

    # Calculate the Pearson correlation coefficient
    corr, _ = pearsonr(jittered_t_normalized, jittered_relative_change)
    print('Pearsons correlation: %.3f' % corr)

    plt.show()

    # Second plot with different colors by treatment type
    # Define treatment types
    treatment_types = {
        'AT100': ['1A1', '1B1', '1C1', '1D1', '1E1', '1A2', '1B2', '1C2', '1D2', '1E2'],
        'AT80': ['1A3', '1B3', '1C3', '1D3', '1E3', '1A4', '1B4', '1C4', '1D4', '1E4'],
        'AT50': ['1A5', '1B5', '1C5', '1D5', '1E5', '1A6', '1B6', '1C6', '1D6', '1E6'],
        'NT': ['1A7', '1B7', '1C7', '1D7', '1E7', '1F7'],
        'MTD': ['1F1', '1F2', '1F3', '1F4', '1F5', '1F6'],
        'Red Calibration MTD': ['1A8', '1B8'],
        'Red Calibration NT': ['1D8']
    }

    # Assign treatment type to each replicate
    df['Treatment'] = 'Other'
    for treatment, replicates in treatment_types.items():
        df.loc[df['Replicate'].isin(replicates), 'Treatment'] = treatment

    # Ensure no NaN values in the 'Treatment' column
    df = df.dropna(subset=['Treatment'])

    # Create a scatter plot with 'g_normalized' on the x-axis and 'relative_change' on the y-axis
    plt.figure(figsize=(12, 6))

    # Add jittering to the data
    if jitter:
        jittered_t_normalized = df['g_normalized'] + np.random.normal(0, 0.1, size=len(df))
        jittered_relative_change = df['relative_change'] + np.random.normal(0, 0.1, size=len(df))
    else:
        jittered_t_normalized = df['g_normalized']
        jittered_relative_change = df['relative_change']

    # Map treatment types to colors
    treatment_colors = {
        'AT100': '#1f77b4',  # Blue
        'AT80': '#ff7f0e',  # Orange
        'AT50': '#d62728',  # Red
        'NT': '#2ca02c',  # Green
    }

    colors = df['Treatment'].map(treatment_colors)
    valid_indices = colors.notna()
    colors = colors[valid_indices]
    jittered_t_normalized = jittered_t_normalized[valid_indices].reset_index(drop=True)
    jittered_relative_change = jittered_relative_change[valid_indices].reset_index(drop=True)
    colors = colors.reset_index(drop=True)

    scatter = plt.scatter(jittered_t_normalized, jittered_relative_change, c=colors, s=5)
    plt.xlabel('g_normalized')
    plt.ylabel('Relative Growth rate of red intensity in a day')
    plt.title('Relative Change vs g_normalized by Treatment Type')
    plt.ylim([0.8, 1.8])  # Set y-axis limits

    # Fit a line to the jittered data
    m, b = np.polyfit(jittered_t_normalized, jittered_relative_change, 1)
    plt.plot(jittered_t_normalized, m * jittered_t_normalized + b, color='red',
             alpha=0.5)  # Make the line semi-transparent

    # Calculate the Pearson correlation coefficient
    corr, _ = pearsonr(jittered_t_normalized, jittered_relative_change)
    print('Pearsons correlation: %.3f' % corr)

    # Add legend for treatment colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=treatment)
               for treatment, color in treatment_colors.items()]
    plt.legend(handles=handles, title='Treatment Types')

    plt.show()

    # === Third plot: Absolute change vs g_normalized by Treatment Type ===
    plt.figure(figsize=(12, 6))

    # Reuse the same jitter flag as above
    if jitter:
        jittered_g = df['g_normalized'] + np.random.normal(0, 0.1, size=len(df))
        jittered_abs_change = df['absolute_change'] + np.random.normal(0, 0.02, size=len(df))
    else:
        jittered_g = df['g_normalized']
        jittered_abs_change = df['absolute_change']

    # Map treatment types to colors (reuse existing dict)
    colors_abs = df['Treatment'].map(treatment_colors)
    valid_idx_abs = colors_abs.notna()
    colors_abs = colors_abs[valid_idx_abs]
    jittered_g = jittered_g[valid_idx_abs].reset_index(drop=True)
    jittered_abs_change = jittered_abs_change[valid_idx_abs].reset_index(drop=True)
    colors_abs = colors_abs.reset_index(drop=True)

    scatter = plt.scatter(jittered_g, jittered_abs_change, c=colors_abs, s=5)
    plt.xlabel('g_normalized')
    plt.ylabel("Absolute day-to-day change of r'_normalized")
    plt.title("Absolute Change (∆r'_normalized) vs g_normalized by Treatment Type")

    # Reasonable symmetric y-limits around zero; adjust if needed
    lo = np.nanpercentile(jittered_abs_change, 1)
    hi = np.nanpercentile(jittered_abs_change, 99)
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    plt.ylim([min(-0.6, lo - pad), max(0.6, hi + pad)])

    # Fit a line on these points
    #m_abs, b_abs = np.polyfit(jittered_g, jittered_abs_change, 1)
    #plt.plot(jittered_g, m_abs * jittered_g + b_abs, color='red', alpha=0.5)

    # Pearson correlation
    #corr_abs, _ = pearsonr(jittered_g, jittered_abs_change)
    #print("Pearson (absolute change): %.3f" % corr_abs)

    # Legend (reuse handles from the previous block)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=treatment)
               for treatment, color in treatment_colors.items()]
    plt.legend(handles=handles, title='Treatment Types')

    plt.yscale('log')
    plt.ylim(0.0001, 2.2)  # optional: matches your clip, gives a bit of headroom

    plt.show()




#########################################################################
    # === NEW PLOT: Treatment topography for relative change ===
    plt.figure(figsize=(8, 6))

    # Compute centroids per treatment
    # rebuild the same mask used in the previous plot
    colors_rel = df['Treatment'].map(treatment_colors)
    valid_idx = colors_rel.notna()

    # build local X/Y for this plot (independent of earlier variables)
    if jitter:
        x_rel = (df.loc[valid_idx, 'g_normalized'] +
                 np.random.normal(0, 0.1, size=valid_idx.sum())).reset_index(drop=True)
        y_rel = ((df.loc[valid_idx, 'relative_change'] - 1.0) * 100.0
                 + np.random.normal(0, 10.0, size=valid_idx.sum())).reset_index(drop=True)
    else:
        x_rel = df.loc[valid_idx, 'g_normalized'].reset_index(drop=True)
        y_rel = ((df.loc[valid_idx, 'relative_change'] - 1.0) * 100.0).reset_index(drop=True)

    labels_rel = df.loc[valid_idx, 'Treatment'].reset_index(drop=True).values
    unique_treatments = sorted(pd.unique(labels_rel))

    for t in unique_treatments:
        sel = (labels_rel == t)
        if sel.sum() < 3:
            continue

        xg = x_rel[sel]
        yg = y_rel[sel]
        color = treatment_colors.get(t, 'k')

        plt.scatter(xg, yg, s=8, c=color, alpha=0.3, label=f"{t} points")

        # KDE mode (peak) as centroid to match the contours visually
        kde_loc = gaussian_kde(np.vstack([xg, yg]), bw_method=0.7)
        gx = np.linspace(np.nanmin(xg), np.nanmax(xg), 200)
        gy = np.linspace(np.nanmin(yg), np.nanmax(yg), 200)
        GX, GY = np.meshgrid(gx, gy)
        Zk = kde_loc(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        iy, ix = np.unravel_index(np.argmax(Zk), Zk.shape)
        x0, y0 = GX[iy, ix], GY[iy, ix]

        plt.scatter(x0, y0, c=color, s=70, marker='X', edgecolor='k', zorder=3, label=f"{t} mode")


        # density contours around the cloud for this treatment
        add_kde_contours(plt.gca(), xg, yg,
                         probs=(0.1, 0.2, 0.3, 0.4),
                         gridsize=300,
                         color=color,
                         alpha=0.9,
                         linewidths=(2.0, 1.5, 1.0),
                         bw_method=0.7,  # tighter
                         pad=0.10)
    plt.xlabel("Normalized susceptible population")
    plt.ylabel("Relative Resistant Growth [%]")
    plt.title("Global Population Competition under AT")
    plt.xlim(0, 6)
    plt.ylim(-10, 60)
    plt.legend(frameon=False, fontsize=8, loc='best')
    plt.tight_layout()

    # --- Save topography plot (relative change) as PDF ---
    output_pdf_path = Path(file_path).parent / "Topography_RelativeChange.pdf"
    plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"[Saved] {output_pdf_path}")

    plt.show()


    # === NEW PLOT: Treatment topography for absolute change ===
    plt.figure(figsize=(8, 6))

    colors_abs = df['Treatment'].map(treatment_colors)
    valid_idx_abs = colors_abs.notna()

    if jitter:
        x_abs = (df.loc[valid_idx_abs, 'g_normalized'] +
                 np.random.normal(0, 0.1, size=valid_idx_abs.sum())).reset_index(drop=True)
        y_abs = (df.loc[valid_idx_abs, 'absolute_change'] +
                 np.random.normal(0, 0.02, size=valid_idx_abs.sum())).reset_index(drop=True)
    else:
        x_abs = df.loc[valid_idx_abs, 'g_normalized'].reset_index(drop=True)
        y_abs = df.loc[valid_idx_abs, 'absolute_change'].reset_index(drop=True)

    labels_abs = df.loc[valid_idx_abs, 'Treatment'].reset_index(drop=True).values
    unique_treatments_abs = sorted(pd.unique(labels_abs))

    for t in unique_treatments_abs:
        sel = (labels_abs == t)
        if sel.sum() < 3:
            continue

        xg = x_abs[sel]
        yg = y_abs[sel]
        color = treatment_colors.get(t, 'k')

        plt.scatter(xg, yg, s=8, c=color, alpha=0.3, label=f"{t} points")

        # KDE mode (peak) as centroid to match the contours visually
        kde_loc = gaussian_kde(np.vstack([xg, yg]), bw_method=0.7)
        gx = np.linspace(np.nanmin(xg), np.nanmax(xg), 200)
        gy = np.linspace(np.nanmin(yg), np.nanmax(yg), 200)
        GX, GY = np.meshgrid(gx, gy)
        Zk = kde_loc(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        iy, ix = np.unravel_index(np.argmax(Zk), Zk.shape)
        x0, y0 = GX[iy, ix], GY[iy, ix]

        plt.scatter(x0, y0, c=color, s=70, marker='X', edgecolor='k', zorder=3, label=f"{t} mode")
        plt.scatter(x0, y0, c=color, s=70, marker='X', edgecolor='k', zorder=3, label=f"{t} centroid")

        add_kde_contours(plt.gca(), xg, yg,
                         probs=(0.1, 0.2, 0.3, 0.4),
                         gridsize=300,
                         color=color,
                         alpha=0.9,
                         linewidths=(2.0, 1.5, 1.0),
                         bw_method=0.7,  # tighter
                         pad=0.10)

    plt.xlabel("g_normalized")
    plt.ylabel("absolute_change (Δ r'_normalized)")
    plt.title("Treatment Group Topography (Absolute Change)")
    plt.legend(frameon=False, fontsize=8, loc='best')
    plt.ylim(-0.25,1)
    plt.tight_layout()
    plt.show()

    return df

correlate_red_growth(combined_df)