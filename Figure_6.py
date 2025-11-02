import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.neighbors import KernelDensity

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the Excel file
file_path = 'Y:\Members\MaxE\Thesis\Data for Figures\Figure_6\ME_A2-4 Data.xlsx'
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

# Plot settings
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 5, num_rows * 5), sharex=True, sharey=True)
fig.suptitle('Normalized Values for Each Replicate and Treatment', fontsize=16)

# Iterate through the treatments and replicates to create subplots
for col, (treatment, replicates) in enumerate(treatments.items()):
    for row, replicate in enumerate(replicates):
        ax = axes[row, col] if num_rows > 1 else axes[col]
        replicate_data = normalized_df[normalized_df['Replicate'] == replicate]

        # Filter out days with NaN values
        replicate_data = replicate_data.dropna(subset=['g_normalized', "r'_normalized", 't_normalized'])

        # Determine the days with actual data
        valid_days = replicate_data['Day']

        # Highlight the treatment periods with different colors based on treatment value
        if replicate in treatment_df.columns:
            treatment_df.reset_index(drop=True, inplace=True)

            # Light yellow background for treatment value 1 or 4
            treatment_days_1_4 = treatment_df.index[treatment_df[replicate].isin([1, 4])].tolist()
            for day in treatment_days_1_4:
                if day in valid_days.values:
                    ax.axvspan(day, day + 1, color='gold', alpha=0.3)

            # Darker yellow background for treatment value 3
            treatment_days_3 = treatment_df.index[treatment_df[replicate] == 3].tolist()
            for day in treatment_days_3:
                if day in valid_days.values:
                    ax.axvspan(day, day + 1, color='red', alpha=0.3)

            # Green background for treatment value 2 or 5
            treatment_days_2_5 = treatment_df.index[treatment_df[replicate].isin([2, 5])].tolist()
            for day in treatment_days_2_5:
                if day in valid_days.values:
                    ax.axvspan(day, day + 1, color='lightgreen', alpha=0.3)

            # Grey background for days with treatment value 0
            other_treatment_days = treatment_df.index[(treatment_df[replicate] == 0)].tolist()
            for day in other_treatment_days:
                if day in valid_days.values:
                    ax.axvspan(day, day + 1, color='grey', alpha=0.3)

        ax.plot(replicate_data['Day'], replicate_data['g_normalized'], color='green', label='g' if row == 0 else "")
        ax.plot(replicate_data['Day'], replicate_data["r'_normalized"], color='red', label="r'" if row == 0 else "")
        ax.plot(replicate_data['Day'], replicate_data['t_normalized'], color='black', label='t' if row == 0 else "")

        ax.axhline(y=1, color='gray', linestyle='--')  # Add dotted line at y = 1

        ax.set_ylim(0.005, 20)  # Set y-axis limits
        ax.set_yscale('log')

        ax.set_title(f'Replicate {replicate}', fontsize=10)

        if row == num_rows - 1:#

            ax.set_xlabel('Days', fontsize=12, labelpad=20)
        if col == 0:
            ax.set_ylabel(f'Replicate {replicate}', fontsize=12, labelpad=20)
        ax.set_xticks(valid_days)

    # Add treatment name as the title above the column of subplots
    axes[0, col].set_title(treatment, fontsize=12, pad=20)

# Add legend to the first subplot
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.9, 0.95])

# Save the plot to a file
plt.savefig(r'Y:\Members\MaxE\Thesis\Data for Figures\Figure_6\normalized_values_plot.png')

# Show plot
plt.show()


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

    # Keep only the necessary columns
    df = df[['Replicate', 'Day', r"r'_normalized", 't_normalized', 'g_normalized', 'treatment', 'cumulative_treatment', 'relative_change']]

    # Remove rows where 'treatment' is 0
    df = df[~df['treatment'].isin([0.0])]

    # Remove rows with NaN values in the 'relative_change' column
    df = df.dropna(subset=['relative_change'])

    # Exclude rows with 'relative_change' greater than 2 or less than -2 # Remove outliers??
    df = df[df['relative_change'].between(0.75, 2)]


    # List of replicates to include
    valid_replicates = [
        '1A1', '1A2', '1A3', '1A4', '1A5', '1A6', '1A7',
        '1B1', '1B2', '1B3', '1B4', '1B5', '1B6', '1B7',
        '1C1', '1C2', '1C3', '1C4', '1C5', '1C6', '1C7',
        '1D1', '1D2', '1D3', '1D4', '1D5', '1D6', '1D7',
        '1E1', '1E2', '1E3', '1E4', '1E5', '1E6', '1E7'
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
        'NT': '#2ca02c'  # Green
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


    ##### Do a plot with relative change of red intensity vs relative change of green intensity

    # Calculate the relative change in 'g_normalized' for each 'Replicate'
    df['relative_change_g'] = df.groupby('Replicate')['g_normalized'].pct_change(fill_method=None) + 1

    # Remove rows with NaN values in the 'relative_change_g' column
    df = df.dropna(subset=['relative_change_g'])

    # Add jittering to the data
    if jitter:
        jittered_relative_change_g = df['relative_change_g'] + np.random.normal(0, 0.1, size=len(df))
        jittered_relative_change_r = df['relative_change'] + np.random.normal(0, 0.1, size=len(df))
    else:
        jittered_relative_change_g = df['relative_change_g']
        jittered_relative_change_r = df['relative_change']

    colors = df['Treatment'].map(treatment_colors)
    valid_indices = colors.notna()
    colors = colors[valid_indices]
    jittered_relative_change_g = jittered_relative_change_g[valid_indices].reset_index(drop=True)
    jittered_relative_change_r = jittered_relative_change_r[valid_indices].reset_index(drop=True)
    colors = colors.reset_index(drop=True)

    # Create a scatter plot with 'relative_change_g' on the x-axis and 'relative_change' on the y-axis
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(jittered_relative_change_g, jittered_relative_change_r, c=colors, s=10)
    plt.xlabel('Relative Change of g_normalized')
    plt.ylabel('Relative Change of r\'_normalized')
    plt.title('Relative Change of r\'_normalized vs Relative Change of g_normalized by Treatment Type')
    plt.ylim([0.6, 2])  # Set y-axis limits
    plt.xlim([0.5, 1.5])  # Set y-axis limits


    # Add legend for treatment colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=treatment)
               for treatment, color in treatment_colors.items()]
    plt.legend(handles=handles, title='Treatment Types')

    plt.show()


    return df

correlate_red_growth(combined_df)




