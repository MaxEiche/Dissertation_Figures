import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import sys

def extract_and_normalize(data_df):
    # Initialize a dictionary to hold the extracted data
    extracted_data = {'Replicate': [], 'Day': [], 'g': [], 'r': [], 't': []}

    # Extract the replicate names and the relevant columns for 'g', 'r', and 't'
    replicate_columns = range(3, data_df.shape[1], 5)  # Updated to every fifth column
    g_columns = range(3, data_df.shape[1], 5)
    r_columns = range(5, data_df.shape[1], 5)
    t_columns = range(6, data_df.shape[1], 5)

    # Extract days from the first column (skip header rows)
    days = data_df.iloc[3:, 0].dropna().astype(int).unique()

    # Loop through the columns to extract the data
    for rep_col, g_col, r_col, t_col in zip(replicate_columns, g_columns, r_columns, t_columns):
        replicate = data_df.iloc[1, rep_col]
        for day in days:
            day_data = data_df.loc[data_df.iloc[:, 0] == day]
            if not day_data.empty:
                extracted_data['Replicate'].append(replicate)
                extracted_data['Day'].append(day)
                extracted_data['g'].append(day_data.iloc[0, g_col])
                extracted_data['r'].append(day_data.iloc[0, r_col])
                extracted_data['t'].append(day_data.iloc[0, t_col])

    # Convert the dictionary to a DataFrame
    extracted_df = pd.DataFrame(extracted_data)

    pd.set_option('display.max_rows', None)

    # Normalize the extracted values based on the 't' value of each replicate at Day 1
    normalized_data = {'Replicate': [], 'Day': [], 'g_normalized': [], 'r_normalized': [], 't_normalized': []}

    # Extract the baseline 't' values for normalization
    baseline_t_values = extracted_df[extracted_df['Day'] == 0].set_index('Replicate')['t']

    # Loop through the extracted data and normalize
    for _, row in extracted_df.iterrows():
        replicate = row['Replicate']
        day = row['Day']
        baseline_t = baseline_t_values.get(replicate, None)
        if baseline_t is not None:
            g_normalized = row['g'] / baseline_t
            r_normalized = row['r'] / baseline_t
            t_normalized = row['t'] / baseline_t

            normalized_data['Replicate'].append(replicate)
            normalized_data['Day'].append(day)
            normalized_data['g_normalized'].append(g_normalized)
            normalized_data['r_normalized'].append(r_normalized)
            normalized_data['t_normalized'].append(t_normalized)
        else:
            print(f"Warning: Baseline t value missing for replicate {replicate}")

    # Convert the normalized data to a DataFrame
    normalized_df = pd.DataFrame(normalized_data)

    return normalized_df

# Function to identify samples based on replicates using the dictionary
def identify_sample(replicate):
    # Remove the prefix from the replicate name if it starts with '1'
    if replicate.startswith('1'):
        replicate = replicate.lstrip('1')
    if replicate.startswith('2'):
        replicate = replicate.lstrip('2')
    for sample, replicates in sample_groups.items():
        if replicate in replicates:
            return sample
    return np.nan

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


def plot_plate_data_with_treatment(combined_df, plate_number):
    # Get unique replicates
    replicates = combined_df['Replicate'].unique()

    # Create a mapping from replicate to subplot position
    replicate_to_position = {replicate: (i // 12, i % 12) for i, replicate in enumerate(replicates)}

    fig, axes = plt.subplots(8, 12, figsize=(24, 16))
    fig.suptitle(f'Plate {plate_number} Data', fontsize=16)

    for replicate in replicates:
        row, col = replicate_to_position[replicate]
        ax = axes[row, col]

        replicate_data = combined_df[combined_df['Replicate'] == replicate]

        ax.plot(replicate_data['Day'], replicate_data['g_normalized'], 'g-', label='g_normalized')
        ax.plot(replicate_data['Day'], replicate_data['r_normalized'], 'r-', label='r_normalized')
        ax.plot(replicate_data['Day'], replicate_data['t_normalized'], 'k-', label='t_normalized')

        ax.axhline(y=1, color='gray', linestyle='--')

        # Set x-axis limit based on the first NaN value in the data
        nan_rows = replicate_data[replicate_data.isna().any(axis=1)]
        first_nan_row = nan_rows.iloc[0]
        value_of_third_column = first_nan_row['Day']

        ax.set_xlim([-2, value_of_third_column - 1])  # Set x-axis limit from -2 to 4
        ax.set_yscale('log')  # Set y-axis to logarithmic scale
        ax.set_ylim([0.02, 4])

        # Add background colors based on the treatment value
        for _, row in replicate_data.iterrows():
            if row['treatment'] == 2:
                ax.axvspan(row['Day'] - 0.5, row['Day'] + 0.5, color='green', alpha=0.3)
            elif row['treatment'] == 1:
                ax.axvspan(row['Day'] - 0.5, row['Day'] + 0.5, color='yellow', alpha=0.3)
            elif row['treatment'] in [0, 3, 4, 5]:
                ax.axvspan(row['Day'] - 0.5, row['Day'] + 0.5, color='gray', alpha=0.3)

        ax.set_title(replicate)

        # Remove x and y ticks for clarity
        #ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(len(replicates), 96):
        row, col = i // 12, i % 12
        axes[row, col].axis('off')

    # Create a single legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()

def plot_kaplan_meier_pooled(df1, df2):
    # Determine the latest day without NaN values
    latest_day1 = df1.dropna(subset=['g_normalized', 'r_normalized', 't_normalized', 'treatment'])['Day'].max()
    latest_day2 = df2.dropna(subset=['g_normalized', 'r_normalized', 't_normalized', 'treatment'])['Day'].max()
    latest_day = max(latest_day1, latest_day2)

    # Initialize events and durations lists
    events = []
    durations = []
    samples = []

    # Function to append events and durations
    def append_events_and_durations(df):
        for sample in df['Sample'].unique():
            sample_data = df[df['Sample'] == sample]
            for replicate in sample_data['Replicate'].unique():
                replicate_data = sample_data[sample_data['Replicate'] == replicate]
                death_day = replicate_data[replicate_data['treatment'] == 3]['Day'].min()
                dropout_day = replicate_data[replicate_data['treatment'].isin([0, 4, 5])]['Day'].min()

                if pd.notna(death_day) and pd.notna(dropout_day):
                    if death_day < dropout_day:
                        events.append(1)
                        durations.append(death_day)
                    else:
                        events.append(0)
                        durations.append(dropout_day)
                elif pd.notna(death_day):
                    events.append(1)
                    durations.append(death_day)
                elif pd.notna(dropout_day):
                    events.append(0)
                    durations.append(dropout_day)
                else:
                    events.append(0)
                    durations.append(latest_day)

                samples.append(sample)

    # Apply the function to identify samples
    df1['Sample'] = df1['Replicate'].apply(identify_sample)
    df2['Sample'] = df2['Replicate'].apply(identify_sample)

    # Append events and durations for both dataframes
    append_events_and_durations(df1)
    append_events_and_durations(df2)

    # Kaplan-Meier Plot
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(12, 8))

    # Define the events
    unique_samples = set(samples)
    sample_colors = {
        'No treatment': 'blue',
        'Maximum-tolerable-dose': 'orange',
        'Adaptive Therapy 100': 'green',
        'Adaptive Therapy 80': 'red',
        'Adaptive Therapy 50': 'purple',
        'Resistant Control No Treatment': 'brown',
        'Resistant Control Maximum-tolerable-dose': 'pink',
        'Intermittent': 'gray'
    }

    sample_durations_dict = {}
    sample_events_dict = {}

    for sample in unique_samples:
        sample_events = [event for i, event in enumerate(events) if samples[i] == sample]
        sample_durations = [duration for i, duration in enumerate(durations) if samples[i] == sample]

        sample_durations_dict[sample] = sample_durations
        sample_events_dict[sample] = sample_events

        kmf.fit(sample_durations, event_observed=sample_events, label=f"{sample} (n={len(sample_durations)})")
        ax = kmf.plot_survival_function(ci_show=False, color=sample_colors.get(sample, 'black'))

        # Add ticks for censored data (dropouts)
        censored_times = [sample_durations[i] for i in range(len(sample_durations)) if sample_events[i] == 0]
        censored_survival = kmf.survival_function_at_times(censored_times).values.flatten()
        ax.plot(censored_times, censored_survival, '|', color=sample_colors.get(sample, 'black'), markersize=10, markeredgewidth=1)

    plt.title('Kaplan-Meier Plot for Pooled Treatments')
    plt.xlabel('Days')
    plt.ylabel('Survival Probability')
    plt.legend(loc='lower left')
    plt.xlim([0, latest_day + 5])
    plt.show()

    # Statistical tests between each pair of survival curves, excluding specified treatments
    excluded_samples = ['No treatment', 'Resistant Control No Treatment', 'Resistant Control Maximum-tolerable-dose']
    p_values = []
    for i, sample1 in enumerate(unique_samples):
        if sample1 in excluded_samples:
            continue
        for j, sample2 in enumerate(unique_samples):
            if sample2 in excluded_samples:
                continue
            if i < j:
                result = logrank_test(sample_durations_dict[sample1], sample_durations_dict[sample2],
                                      event_observed_A=sample_events_dict[sample1],
                                      event_observed_B=sample_events_dict[sample2])
                p_values.append((sample1, sample2, result.p_value))

    # Print significant differences
    significant_threshold = 0.05
    significant_differences = [(s1, s2, p) for s1, s2, p in p_values if p < significant_threshold]

    if significant_differences:
        print("Significant differences found between the following groups in the Kaplan-Meier plot:")
        for s1, s2, p in significant_differences:
            print(f"{s1} vs {s2}: p-value = {p:.5f}")
    else:
        print("No significant differences found between the groups in the Kaplan-Meier plot.")


def plot_treatment_violinplots(df1, df2):
    # Create copies of the dataframes to avoid modifying the original dataframes
    df1 = df1.copy()
    df2 = df2.copy()

    # Add a prefix to the 'Replicate' column in each dataframe
    df1['Replicate'] = df1['Replicate'].apply(lambda x: '1' + x)
    df2['Replicate'] = df2['Replicate'].apply(lambda x: '2' + x)

    # Combine the two dataframes
    combined_df = pd.concat([df1, df2])

    # Filter out rows that contain a NaN value
    combined_df2 = combined_df.dropna()

    # Filter out replicates that have a 4 or 5 in their 'treatment' column at any point
    replicates_to_exclude = combined_df2[combined_df2['treatment'].isin([4.0, 5.0])]['Replicate'].unique()
    filtered_df = combined_df2[~combined_df2['Replicate'].isin(replicates_to_exclude)]

    # Filter out rows where the 'treatment' column is 3 or 0
    filtered_df = filtered_df[~filtered_df['treatment'].isin([3.0, 0.0])]

    # Find the maximum day for each replicate
    max_day_df = filtered_df.groupby('Replicate')['Day'].max().reset_index()

    # Filter for rows where treatment is 1.0
    filtered_df2 = combined_df[combined_df['treatment'] == 1.0]

    # Group by 'Replicate' and 'treatment', and count the number of days
    grouped_df = filtered_df2.groupby(['Replicate', 'treatment'])['Day'].count().reset_index()

    # Identify the sample for each replicate
    grouped_df['Sample'] = grouped_df['Replicate'].apply(identify_sample)

    # Merge the grouped_df with max_day_df on the 'Replicate' column
    grouped_df = pd.merge(grouped_df, max_day_df, on='Replicate', how='inner')

    # Rename the 'Day' column to 'Max_Day'
    grouped_df.rename(columns={'Day': 'Max_Day'}, inplace=True)

    # Exclude specified samples
    excluded_samples = ['No treatment', 'Resistant Control No Treatment', 'Resistant Control Maximum-tolerable-dose']
    grouped_df = grouped_df[~grouped_df['Sample'].isin(excluded_samples)]

    # Create a new column 'normalized_Day_x' in the dataframe
    grouped_df['normalized_Day_x'] = (grouped_df['Day_x'] - 1) / grouped_df['Day_y']

    # Calculate the median for each group
    grouped_df['median'] = grouped_df.groupby('Sample')['normalized_Day_x'].transform('median')

    # Sort the DataFrame by the median values
    sorted_df = grouped_df.sort_values('median')

    # Create box plots of the number of days for each sample, separated by treatment
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Sample', y='normalized_Day_x', hue='treatment', data=sorted_df, order=sorted_df['Sample'].unique())
    plt.title('Violin Plot of Cumulative Treatment Times for All Treatment Groups')
    plt.ylabel('Fraction of days with treatment')
    plt.ylim(0.35, 1.1)
    plt.show()

def plot_scatterplot(df1, df2):

    df1 = df1.copy()
    df2 = df2.copy()

    # Add a prefix to the 'Replicate' column in each dataframe
    df1['Replicate'] = df1['Replicate'].apply(lambda x: '1' + x)
    df2['Replicate'] = df2['Replicate'].apply(lambda x: '2' + x)

    # Combine the two dataframes
    combined_df = pd.concat([df1, df2])

    # Filter out rows that contain a NaN value
    combined_df = combined_df.dropna()

    # Filter out replicates that have a 0, 4 or 5 in their 'treatment' column at any point
    replicates_to_exclude = combined_df[combined_df['treatment'].isin([4.0, 5.0, 0.0])]['Replicate'].unique()
    filtered_df = combined_df[~combined_df['Replicate'].isin(replicates_to_exclude)]

    # Filter out rows where the 'treatment' column is 3 or 0
    filtered_df = filtered_df[~filtered_df['treatment'].isin([3.0, 0.0])]

    # Group by 'Replicate' and count the number of days where 'treatment' is 1.0
    treatment_days_df = filtered_df[filtered_df['treatment'] == 1.0].groupby('Replicate')['Day'].count().reset_index()

    # Find the maximum day for each replicate
    max_day_df = filtered_df.groupby('Replicate')['Day'].max().reset_index()

    # Merge the two dataframes
    merged_df = pd.merge(treatment_days_df, max_day_df, on='Replicate')

    # Group the combined_df by 'Replicate' and get the most frequent sample type for each replicate
    sample_type_df = combined_df.groupby('Replicate')['Sample'].agg(lambda x: x.mode()[0]).reset_index()

    # Merge this dataframe with the merged_df on the 'Replicate' column
    merged_df = pd.merge(merged_df, sample_type_df, on='Replicate')

    # Create a new column 'normalized_Day_x' in the dataframe
    merged_df['normalized_Day_x'] = (merged_df['Day_x'] - 1) / merged_df['Day_y']

    # Create a dictionary that maps each unique sample to a unique number
    sample_to_number = {sample: number for number, sample in enumerate(merged_df['Sample'].unique())}

    # Create a new column 'color' in the dataframe that contains the mapped values
    merged_df['color'] = merged_df['Sample'].map(sample_to_number)

    # Create a scatter plot with the maximum day on the x-axis and the cumulative number of treatment days on the y-axis
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab10', len(sample_to_number))
    # Add jitter to the x and y coordinates
    jittered_x = merged_df['Day_y'] + np.random.normal(0, 0.2, size=len(merged_df))
    jittered_y = merged_df['normalized_Day_x'] + np.random.normal(0, 0.02, size=len(merged_df))
    scatter = plt.scatter(jittered_x, jittered_y, c=merged_df['color'], cmap=cmap, s=15)
    patches = [mpatches.Patch(color=scatter.cmap(scatter.norm(sample_to_number[sample])), label=sample) for sample in sample_to_number]
    plt.title('Scatter Plot of Treatment Failure Day vs Fraction of days with treatment for each replicate')
    plt.xlabel('Treatment Failure Day')
    plt.ylabel('Fraction of days with treatment')
    plt.legend(handles=patches, loc='lower left')
    plt.ylim(0.4, 1.1)
    plt.xlim([0, None])
    plt.show()

def correlate_red_growth(df1, df2):

    df1 = df1.copy()
    df2 = df2.copy()

    # Add a prefix to the 'Replicate' column in each dataframe
    df1['Replicate'] = df1['Replicate'].apply(lambda x: '1' + x)
    df2['Replicate'] = df2['Replicate'].apply(lambda x: '2' + x)

    # Combine the two dataframes
    df = pd.concat([df1, df2])

    # Calculate the cumulative sum of 'treatment' for each 'Replicate'
    df['cumulative_treatment'] = df.groupby((df['treatment'] != df['treatment'].shift(1)).cumsum()).apply(
        lambda x: (x['treatment'] == 1.0).cumsum()).reset_index(drop=True)

    # Shift the 'cumulative_treatment' column by one day
    df['cumulative_treatment'] = df.groupby('Replicate')['cumulative_treatment'].shift(fill_value=0)

    # Calculate the relative change in 'r_normalized' for each 'Replicate'
    df['relative_change'] = df.groupby('Replicate')['r_normalized'].pct_change(fill_method=None) + 1

    # Keep only the necessary columns
    df = df[['Replicate', 'Day', 'r_normalized', 't_normalized', 'g_normalized', 'treatment', 'cumulative_treatment', 'relative_change']]

    # Remove rows where 'treatment' is 3.0, 4.0, or 5.0
    df = df[~df['treatment'].isin([0.0, 3.0, 4.0, 5.0])]

    # Remove rows with NaN values in the 'relative_change' column
    df = df.dropna(subset=['relative_change'])

    # Exclude rows with 'relative_change' greater than 2 or less than -2 # Remove outliers??
    df = df[df['relative_change'].between(0.75, 2)]

    # Create separate DataFrames for the treatment statuses 1.0 and 2.0
    df_treatment_1 = df[df['treatment'] == 1.0]
    df_treatment_2 = df[df['treatment'] == 2.0]

    # Map the treatment values to the new labels
    df['treatment'] = df['treatment'].map({1.0: 'Treatment the day before', 2.0: 'No treatment the day before'})

    # Calculate the medians
    median_treatment = df[df['treatment'] == 'Treatment the day before']['relative_change'].median()
    median_no_treatment = df[df['treatment'] == 'No treatment the day before']['relative_change'].median()

    # Convert to percentages
    median_treatment_percent = round((median_treatment * 100) - 100, 2)
    median_no_treatment_percent = round((median_no_treatment * 100) - 100, 2)

    # Print the medians
    print(f"Median of 'relative_change' for 'Treatment the day before': {median_treatment_percent}%")
    print(f"Median of 'relative_change' for 'No treatment the day before': {median_no_treatment_percent}%")

    # Create box plots for each treatment status in one plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='treatment', y='relative_change', data=df[df['treatment'].isin(['Treatment the day before', 'No treatment the day before'])])
    plt.title('Relative Growth rate of red intensity in a day')
    plt.ylabel('Relative Change')
    plt.ylim([0.75, 1.25])  # Set y-axis limits

    plt.tight_layout()
    plt.show()

    # Perform the Mann-Whitney U test
    stat, p = mannwhitneyu(df_treatment_1['relative_change'], df_treatment_2['relative_change'])
    print('Statistics=%.3f, p=%.2E' % (stat, p))

    # Interpretation
    alpha = 0.05
    if p > alpha:
        print('No difference seen between both distributions')
    else:
        print('Both growth distributions are significantly different')

    # Create a scatter plot with 'cumulative_treatment' on the x-axis and 'relative_change' on the y-axis
    plt.figure(figsize=(12, 6))

    # Add jittering to the data
    jittered_cumulative_treatment = df['cumulative_treatment'] + np.random.normal(0, 1, size=len(df))
    jittered_relative_change = df['relative_change'] + np.random.normal(0, 0.1, size=len(df))

    scatter = plt.scatter(jittered_cumulative_treatment, jittered_relative_change, c=df['Day'], cmap='viridis', s=5)
    plt.colorbar(scatter, label='Day')
    plt.xlabel('Length of consecutive treatment before the day of measurement')
    plt.ylabel('Relative Growth rate of red intensity in a day')
    plt.title('Relative Change vs Cumulative Treatment')
    plt.ylim([0.6, 2])  # Set y-axis limits

    # Fit a line to the jittered data
    m, b = np.polyfit(jittered_cumulative_treatment, jittered_relative_change, 1)
    plt.plot(jittered_cumulative_treatment, m * jittered_cumulative_treatment + b, color='red')

    # Calculate the Pearson correlation coefficient
    corr, _ = pearsonr(jittered_cumulative_treatment, jittered_relative_change)
    print('Pearsons correlation: %.3f' % corr)

    plt.show()

    # Create a scatter plot with 'g_normalized' on the x-axis and 'relative_change' on the y-axis
    plt.figure(figsize=(12, 6))

    # Add jittering to the data
    jittered_t_normalized = df['g_normalized'] + np.random.normal(0, 0.1, size=len(df))
    jittered_relative_change = df['relative_change'] + np.random.normal(0, 0.1, size=len(df))

    scatter = plt.scatter(jittered_t_normalized, jittered_relative_change, c=df['Day'], cmap='viridis', s=5)
    plt.colorbar(scatter, label='Day')
    plt.xlabel('g_normalized')
    plt.ylabel('Relative Growth rate of red intensity in a day')
    plt.title('Relative Change vs g_normalized')
    plt.ylim([0.6, 2])  # Set y-axis limits

    # Fit a line to the jittered data
    m, b = np.polyfit(jittered_t_normalized, jittered_relative_change, 1)
    plt.plot(jittered_t_normalized, m * jittered_t_normalized + b, color='red')

    # Calculate the Pearson correlation coefficient
    corr, _ = pearsonr(jittered_t_normalized, jittered_relative_change)
    print('Pearsons correlation: %.3f' % corr)

    plt.show()

    return df

##### Remove artifacts from the data #########

def extract_and_normalize_remove_artifacts(data_df):
    # Manually adjusts the data to remove artifacts, TAKE CARE, VERY CRUDE

    # Initialize a dictionary to hold the extracted data
    extracted_data = {'Replicate': [], 'Day': [], 'g': [], 'r': [], 't': []}

    # Extract the replicate names and the relevant columns for 'g', 'r', and 't'
    replicate_columns = range(3, data_df.shape[1], 5)  # Updated to every fifth column
    g_columns = range(3, data_df.shape[1], 5)
    r_columns = range(5, data_df.shape[1], 5)
    t_columns = range(6, data_df.shape[1], 5)

    # Extract days from the first column (skip header rows)
    days = data_df.iloc[3:, 0].dropna().astype(int).unique()

    # Loop through the columns to extract the data
    for rep_col, g_col, r_col, t_col in zip(replicate_columns, g_columns, r_columns, t_columns):
        replicate = data_df.iloc[1, rep_col]
        for day in days:
            if day == 29 or day == 15 or day == 27:  # Skip day 29 and day 15 and day 27
                continue
            day_data = data_df.loc[data_df.iloc[:, 0] == day]
            if not day_data.empty:
                g_value = day_data.iloc[0, g_col]
                r_value = day_data.iloc[0, r_col]
                if day in range(-2, 6):  # For days -2 to 5
                    g_value -= 40000000  # Subtract 35000000 from 'g' value
                    r_value -= 40000000  # Subtract 35000000 from 'r' value
                elif day == 6:  # For day 6
                    g_value -= 5000000  # Subtract 5000000 from 'g' value
                    r_value -= 5000000  # Subtract 5000000 from 'r' value
                # Subtract r'_normalized / 6 from g
                g_value -= r_value / 12
                extracted_data['Replicate'].append(replicate)
                extracted_data['Day'].append(day)
                extracted_data['g'].append(g_value)
                extracted_data['r'].append(r_value)
                extracted_data['t'].append(day_data.iloc[0, t_col])

    # Convert the dictionary to a DataFrame
    extracted_df = pd.DataFrame(extracted_data)

    pd.set_option('display.max_rows', None)

    # Normalize the extracted values based on the 't' value of each replicate at Day 1
    normalized_data = {'Replicate': [], 'Day': [], 'g_normalized': [], 'r_normalized': [], 't_normalized': []}

    # Extract the baseline 't' values for normalization
    baseline_t_values = extracted_df[extracted_df['Day'] == 0].set_index('Replicate')['t']

    # Loop through the extracted data and normalize
    for _, row in extracted_df.iterrows():
        replicate = row['Replicate']
        day = row['Day']
        baseline_t = baseline_t_values.get(replicate, None)
        if baseline_t is not None:
            g_normalized = row['g'] / baseline_t
            r_normalized = row['r'] / baseline_t
            t_normalized = row['t'] / baseline_t

            normalized_data['Replicate'].append(replicate)
            normalized_data['Day'].append(day)
            normalized_data['g_normalized'].append(g_normalized)
            normalized_data['r_normalized'].append(r_normalized)
            normalized_data['t_normalized'].append(t_normalized)
        else:
            print(f"Warning: Baseline t value missing for replicate {replicate}")

    # Convert the normalized data to a DataFrame
    normalized_df = pd.DataFrame(normalized_data)

    return normalized_df

def process_replicate_data(normalized_df, treatment_df):
    # Initialize an empty list to store the results
    results = []

    # Iterate over each unique replicate
    for replicate in normalized_df['Replicate'].unique():
        # Filter the dataframe for the current replicate
        replicate_df = normalized_df[normalized_df['Replicate'] == replicate]

        # Define the condition for two consecutive days
        condition = ((replicate_df['t_normalized'] > 5) & (replicate_df['t_normalized'].shift(1) > 5)) | \
                    ((replicate_df['r_normalized'] > 0.1) & (replicate_df['r_normalized'].shift(1) > 0.1))

        # Find the first day where the condition is met for two consecutive days
        first_day = replicate_df[condition]['Day'].min()

        # Append the result to the list
        results.append({'Replicate': replicate, 'First_Day': first_day})

    # Convert the results list to a new dataframe
    result_df = pd.DataFrame(results)

    # Extract relevant days for each replicate in treatment_df
    relevant_days = {}
    for replicate in treatment_df.columns[2:]:  # Skip the first two columns
        for index, row in treatment_df.iterrows():
            if row[replicate] in [0, 4, 5]:
                relevant_days[replicate] = row['Unnamed: 0']  # Assuming 'Unnamed: 0' is the day column
                break

    # Convert the relevant days to a DataFrame
    relevant_days_df = pd.DataFrame(list(relevant_days.items()), columns=['Replicate', 'First_Relevant_Day'])

    # Merge the relevant days into the first DataFrame
    result_df = result_df.merge(relevant_days_df, left_on='Replicate', right_on='Replicate', how='left')

    return result_df

def combine_replicate_data(df1, df2):
    # Add a prefix to the 'Replicate' column in each dataframe
    df1['Replicate'] = df1['Replicate'].apply(lambda x: '1' + x)
    df2['Replicate'] = df2['Replicate'].apply(lambda x: '2' + x)

    # Combine the two dataframes
    combined_df = pd.concat([df1, df2])

    return combined_df

def plot_kaplan_meier(normalized_df_plate1, normalized_df_plate2, treatment_df_plate1, treatment_df_plate2, sample_groups, sample_colors):
    # Process replicate data
    result_df1 = process_replicate_data(normalized_df_plate1, treatment_df_plate1)
    result_df2 = process_replicate_data(normalized_df_plate2, treatment_df_plate2)

    # Combine replicate data
    combined_result_df = combine_replicate_data(result_df1, result_df2)

    # Convert 'First_Day' and 'First_Relevant_Day' to numeric, forcing errors to NaN
    combined_result_df['First_Day'] = pd.to_numeric(combined_result_df['First_Day'], errors='coerce')
    combined_result_df['First_Relevant_Day'] = combined_result_df['First_Relevant_Day'].str.extract(r'(\d+)').astype(float)

    # Determine the event day and event type
    combined_result_df['Event_Day'] = combined_result_df[['First_Day', 'First_Relevant_Day']].min(axis=1)
    combined_result_df['Event_Type'] = combined_result_df.apply(
        lambda row: 1 if pd.isna(row['First_Day']) or row['First_Day'] <= row['First_Relevant_Day'] else 0,
        axis=1
    )

    # Filter out rows with NaN values in 'Event_Day' or 'Event_Type'
    filtered_df = combined_result_df.dropna(subset=['Event_Day', 'Event_Type'])

    # Invert the Event_Type values using .loc to avoid SettingWithCopyWarning
    filtered_df.loc[:, 'Event_Type'] = filtered_df['Event_Type'].apply(lambda x: 1 if x == 0 else 0)

    # Initialize the plot
    plt.figure(figsize=(15, 10))

    # Exclude specified groups
    excluded_groups = ['No treatment', 'Resistant Control No Treatment', 'Resistant Control Maximum-tolerable-dose']

    # Plot Kaplan-Meier curves for each sample group on the same plot
    for group, replicates in sample_groups.items():
        if group not in excluded_groups:
            kmf = KaplanMeierFitter()
            group_df = filtered_df[filtered_df['Replicate'].str[1:].isin(replicates)]
            kmf.fit(group_df['Event_Day'], event_observed=group_df['Event_Type'], label=group)
            kmf.plot_survival_function(ci_show=False, color=sample_colors[group])

    # Set plot title and labels
    plt.title('Kaplan-Meier Plot for Selected Groups')
    plt.xlabel('Days')
    plt.ylabel('Survival Probability')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def add_fallapart_info(combined_df, fallapart_df):
    # Extract the relevant columns from the fallapart dataframe
    fallapart_days = fallapart_df.iloc[:, 0].dropna().unique()
    replicate_columns = fallapart_df.columns[2:]

    # Create a dictionary to map (Day, Replicate) to fallapart value
    fallapart_dict = {}
    for index, row in fallapart_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in fallapart_days:
            continue
        day = int(day_str[1:])  # Assuming the day values are formatted like 'd-2', 'd-1', etc.
        for replicate in replicate_columns:
            fallapart_dict[(day, replicate)] = row[replicate]

    # Add the new 'fallapart_info' column to the combined_df dataframe
    combined_df['fallapart_info'] = combined_df.apply(
        lambda row: 6 if fallapart_dict.get((row['Day'], row['Replicate']), 0) == 6 else 0, axis=1
    )

    return combined_df

def plot_treatment_failure_vs_fallapart(df1, df2, sample_groups, sample_colors):
    # Combine the two dataframes
    df1 = df1.copy()
    df2 = df2.copy()
    df1['Replicate'] = df1['Replicate'].apply(lambda x: '1' + x)
    df2['Replicate'] = df2['Replicate'].apply(lambda x: '2' + x)
    combined_df = pd.concat([df1, df2])

    # Filter out rows that contain a NaN value
    combined_df = combined_df.dropna()

    # Filter out replicates that have a 0, 4 or 5 in their 'treatment' column at any point
    replicates_to_exclude = combined_df[combined_df['treatment'].isin([4.0, 5.0, 0.0])]['Replicate'].unique()
    filtered_df = combined_df[~combined_df['Replicate'].isin(replicates_to_exclude)]

    # Filter out rows where the 'treatment' column is 3 or 0
    filtered_df = filtered_df[~filtered_df['treatment'].isin([3.0, 0.0])]

    # Find the maximum day for each replicate
    max_day_df = filtered_df.groupby('Replicate')['Day'].max().reset_index()

    # Find the first day where fallapart_info is 6 for each replicate
    fallapart_day_df = filtered_df[filtered_df['fallapart_info'] == 6].groupby('Replicate')['Day'].min().reset_index()
    fallapart_day_df.rename(columns={'Day': 'Fallapart_Day'}, inplace=True)

    # Merge the two dataframes on 'Replicate'
    merged_df = pd.merge(max_day_df, fallapart_day_df, on='Replicate', how='inner')

    # Adjust the sample groups to account for the added prefix in the replicate numbers
    adjusted_sample_groups = {}
    for sample, replicates in sample_groups.items():
        adjusted_sample_groups[sample] = ['1' + rep for rep in replicates] + ['2' + rep for rep in replicates]

    # Map replicates to sample groups
    replicate_to_sample = {replicate: sample for sample, replicates in adjusted_sample_groups.items() for replicate in replicates}
    merged_df['Sample'] = merged_df['Replicate'].map(replicate_to_sample)

    # Map samples to colors
    merged_df['Color'] = merged_df['Sample'].map(sample_colors)

    # Plot the data
    plt.figure(figsize=(12, 8))
    for sample, color in sample_colors.items():
        sample_data = merged_df[merged_df['Sample'] == sample]
        plt.scatter(sample_data['Fallapart_Day'], sample_data['Day'], c=color, label=sample,
                    s=15)  # Swapped x and y axes

        # Calculate the average and standard deviation for x and y values
        avg_x = sample_data['Fallapart_Day'].mean()
        avg_y = sample_data['Day'].mean()
        std_x = sample_data['Fallapart_Day'].std()
        std_y = sample_data['Day'].std()

        # Determine the size of the circle based on the standard deviation
        size = std_x + std_y * 100  # Adjust the factor as needed

        # Draw a semitransparent circle at the average location
        plt.scatter(avg_x, avg_y, c=color, s=size, alpha=0.5, edgecolors='w')

    # Add a dotted line for y = x
    plt.plot([0, 60], [0, 60], linestyle='dotted', color='black')

    plt.title('First Day of Destruction vs Treatment Failure Day')
    plt.xlabel('First Day of Destruction')
    plt.ylabel('Treatment Failure Day')
    plt.legend(loc='best')
    plt.xlim(0, 60)
    plt.ylim(0, 60)
    plt.show()

def correlate_red_growth_with_fallapart(df1, df2):

    df1 = df1.copy()
    df2 = df2.copy()

    # Add a prefix to the 'Replicate' column in each dataframe
    df1['Replicate'] = df1['Replicate'].apply(lambda x: '1' + x)
    df2['Replicate'] = df2['Replicate'].apply(lambda x: '2' + x)

    # Combine the two dataframes
    df = pd.concat([df1, df2])

    # Calculate the relative change in 'r_normalized' for each 'Replicate'
    df['relative_change'] = df.groupby('Replicate')['r_normalized'].pct_change(fill_method=None) + 1

    # Keep only the necessary columns
    df = df[['Replicate', 'Day', 'r_normalized', 't_normalized', 'g_normalized', 'treatment', 'relative_change', 'fallapart_info']]

    # Remove rows where 'treatment' is 3.0, 4.0, or 5.0
    df = df[~df['treatment'].isin([0.0, 3.0, 4.0, 5.0])]

    # Remove rows with NaN values in the 'relative_change' column
    df = df.dropna(subset=['relative_change'])

    # Exclude rows with 'relative_change' greater than 2 or less than -2 # Remove outliers??
    df = df[df['relative_change'].between(0.75, 2)]

    # Initialize the 'adjusted_day' column to NaN
    df['adjusted_day'] = np.nan

    # Iterate over each unique replicate
    for replicate in df['Replicate'].unique():
        # Filter the dataframe for the current replicate
        replicate_df = df[df['Replicate'] == replicate]

        # Find the first day where 'fallapart_info' is 6
        fallapart_day = replicate_df[replicate_df['fallapart_info'] == 6]['Day'].min()

        if pd.notna(fallapart_day):
            # Adjust the days based on the fallapart day
            df.loc[df['Replicate'] == replicate, 'adjusted_day'] = df.loc[df[
                                                                              'Replicate'] == replicate, 'Day'] - fallapart_day
        else:
            # If no fallapart day is found, use the last day of the replicate
            last_day = replicate_df['Day'].max()
            df.loc[df['Replicate'] == replicate, 'adjusted_day'] = df.loc[df['Replicate'] == replicate, 'Day'] - last_day

    # Adjust the sample groups to account for the added prefix in the replicate numbers
    adjusted_sample_groups = {}
    for sample, replicates in sample_groups.items():
        adjusted_replicates = ['1' + rep if rep[0].isdigit() else '2' + rep for rep in replicates]
        adjusted_sample_groups[sample] = adjusted_replicates

    # Exclude replicates from 'Resistant Control No Treatment' and 'Resistant Control Maximum-tolerable-dose'
    excluded_samples = ['Resistant Control No Treatment', 'Resistant Control Maximum-tolerable-dose']
    excluded_replicates = [rep for sample in excluded_samples for rep in adjusted_sample_groups[sample]]
    df = df[~df['Replicate'].isin(excluded_replicates)]

    # Map replicates to sample groups
    replicate_to_sample = {replicate: sample for sample, replicates in adjusted_sample_groups.items() for replicate
                           in replicates}
    df['Sample'] = df['Replicate'].map(replicate_to_sample)

    # Map samples to colors
    df['Color'] = df['Sample'].map(sample_colors)

    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    for sample, color in sample_colors.items():
        if sample not in excluded_samples:
            sample_df = df[df['Sample'] == sample]
            plt.scatter(sample_df['adjusted_day'], sample_df['relative_change'], alpha=0.6, edgecolors='w', s=30, label=sample, color=color)


    # Add titles and labels
    plt.title('Relative Change vs Adjusted Day')
    plt.xlabel('Adjusted Day')
    plt.ylabel('Relative Change')
    plt.legend(title='Sample Groups')

    # Fit and plot the correlation line
    m, b = np.polyfit(df['adjusted_day'], df['relative_change'], 1)
    plt.plot(df['adjusted_day'], m * df['adjusted_day'] + b, color='red', alpha=0.5)

    plt.show()

    # Plot individual scatter plots for each sample type in one figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    plot_idx = 0

    for sample, color in sample_colors.items():
        if sample not in excluded_samples:
            sample_df = df[df['Sample'] == sample]
            ax = axes[plot_idx]
            ax.scatter(sample_df['adjusted_day'], sample_df['relative_change'], alpha=0.6, edgecolors='w', s=30,
                       label=sample, color=color)
            ax.set_title(f'Relative Change vs Adjusted Day for {sample}')
            ax.set_xlabel('Adjusted Day')
            ax.set_ylabel('Relative Change')
            ax.set_xlim([-30, 40])
            ax.set_ylim([0.7, 2.0])
            ax.legend(title='Sample Groups')

            # Fit and plot the correlation line for the individual plot
            m, b = np.polyfit(sample_df['adjusted_day'], sample_df['relative_change'], 1)
            ax.plot(sample_df['adjusted_day'], m * sample_df['adjusted_day'] + b, color='red', alpha=0.5)

            plot_idx += 1

    plt.tight_layout()
    plt.show()

    # Bin relative_change values based on adjusted_day
    before_day_0 = df[df['adjusted_day'] < 0]['relative_change']
    after_day_0 = df[df['adjusted_day'] >= 0]['relative_change']

    # Combine the data into a single DataFrame
    combined_data = pd.DataFrame({
        'relative_change': pd.concat([before_day_0, after_day_0], axis=0),
        'Time Period': ['Before Day 0'] * len(before_day_0) + ['After Day 0'] * len(after_day_0)
    })

    # Create the violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Time Period', y='relative_change', data=combined_data, bw_method=0.2, density_norm='count')
    plt.title('Distribution of Relative Change Before, At, and After Day 0')
    plt.xlabel('Time Period')
    plt.ylabel('Relative Change')
    plt.ylim([0.7, 1.7])
    plt.show()


########################## MAIN ##################################

# Sample groups definition
sample_groups = {
    'No treatment': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
    'Maximum-tolerable-dose': ['A7', 'A8', 'A9', 'A10', 'A11', 'A12'],
    'Adaptive Therapy 100': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11',
              'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11'],
    'Adaptive Therapy 80': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11',
             'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11'],
    'Adaptive Therapy 50': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11',
             'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11'],
    'Resistant Control No Treatment': ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'],
    'Resistant Control Maximum-tolerable-dose': ['H7', 'H8', 'H9', 'H10', 'H11', 'H12'],
    'Intermittent': ['B12', 'C12', 'D12', 'E12', 'F12', 'G12']
}

# Define colors for the plots
colors = {'g_normalized_mean': 'green', 'r_normalized_mean': 'red', 't_normalized_mean': 'black'}
sample_colors = {
    'No treatment': 'blue',
    'Maximum-tolerable-dose': 'orange',
    'Adaptive Therapy 100': 'green',
    'Adaptive Therapy 80': 'red',
    'Adaptive Therapy 50': 'purple',
    'Resistant Control No Treatment': 'brown',
    'Resistant Control Maximum-tolerable-dose': 'pink',
    'Intermittent': 'gray'
}


# Load the Excel file
file_path = r'Y:\Members\MaxE\Thesis\Data for Figures\Figure_7\ME_A2-3 Data.xlsx'
xls = pd.ExcelFile(file_path)

# Load the data from the correct sheets for each plate
data_df_plate1 = pd.read_excel(xls, 'Plate 1')
data_df_plate2 = pd.read_excel(xls, 'Plate 2')

# Load the treatment data from the correct sheets for each plate
treatment_df_plate1 = pd.read_excel(xls, 'Plate 1 Treatment')
treatment_df_plate2 = pd.read_excel(xls, 'Plate 2 Treatment')

# Extract and normalize data for both plates
normalized_df_plate1 = extract_and_normalize(data_df_plate1)
normalized_df_plate2 = extract_and_normalize(data_df_plate2)

# Combine normalized data with treatment data for both plates
combined_df_plate1 = combine_normalized_with_treatment(normalized_df_plate1, treatment_df_plate1)
combined_df_plate2 = combine_normalized_with_treatment(normalized_df_plate2, treatment_df_plate2)

# Extract and normalize data for both plates and remove artifacts
normalized_df_plate1_deartifacted = extract_and_normalize_remove_artifacts(data_df_plate1)
normalized_df_plate2_deartifacted = extract_and_normalize_remove_artifacts(data_df_plate2)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Combine normalized data with treatment data for both plates and removed artifacts
combined_df_plate1_deartifacted = combine_normalized_with_treatment(normalized_df_plate1_deartifacted, treatment_df_plate1)
combined_df_plate2_deartifacted = combine_normalized_with_treatment(normalized_df_plate2_deartifacted, treatment_df_plate2)

# Plot the data for the provided dataframe with treatment backgrounds
plot_plate_data_with_treatment(combined_df_plate1_deartifacted, 1)
plot_plate_data_with_treatment(combined_df_plate2_deartifacted, 2)

### Plot without deartification ###
#plot_plate_data_with_treatment(combined_df_plate1, 1)
#plot_plate_data_with_treatment(combined_df_plate2, 2)

# Process and plot data for both dataframes separately
log_scale = False

# Plot the Kaplan-Meier diagram for the pooled data
plot_kaplan_meier_pooled(combined_df_plate1_deartifacted, combined_df_plate2_deartifacted)

# Plot a different Kaplan Meier. Failure here is if t_normalized > 5 or r_normalized > 0.1
plot_kaplan_meier(normalized_df_plate1_deartifacted, normalized_df_plate2_deartifacted, treatment_df_plate1, treatment_df_plate2, sample_groups, sample_colors)

# Call the function with the combined dataframes as arguments
plot_treatment_violinplots(combined_df_plate1_deartifacted, combined_df_plate2_deartifacted)

# Call the function with the combined dataframes as arguments
plot_scatterplot(combined_df_plate1_deartifacted, combined_df_plate2_deartifacted)

# Research the correlation between growth of the red population and treatment
# Use the function on your DataFrame
correlate_red_growth(combined_df_plate1_deartifacted, combined_df_plate2_deartifacted)


##### Do new plots  to study red growth rate before and after spheroid destruction #####

# Load the data from the correct sheets for each plate
df_fallapart1 = pd.read_excel(xls, 'Plate 1 Treatment_fallapart')
df_fallapart2 = pd.read_excel(xls, 'Plate 2 Treatment_fallapart')

# Add the fallapart_info column to the combined_df_plate1_deartifacted
combined_df_plate1_deartifacted_fallapart = add_fallapart_info(combined_df_plate1_deartifacted, df_fallapart1)
combined_df_plate2_deartifacted_fallapart = add_fallapart_info(combined_df_plate2_deartifacted, df_fallapart2)

# Plot the treatment failure vs Day of destruction
plot_treatment_failure_vs_fallapart(combined_df_plate1_deartifacted_fallapart, combined_df_plate2_deartifacted_fallapart, sample_groups, sample_colors)

# Plot Red growth rate against Day of destruction
correlate_red_growth_with_fallapart(combined_df_plate1_deartifacted_fallapart, combined_df_plate2_deartifacted_fallapart)