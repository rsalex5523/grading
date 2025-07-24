import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
import statsmodels.formula.api as smf
import math


# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================

def load_and_prepare_data(file_path, price_cap, num_price_groups, grading_ref_cat, price_ref_cat_num):
    """
    Loads data, filters by price, and engineers features with dynamic controls.

    Args:
        file_path (str): Path to the CSV file.
        price_cap (int): Maximum base price (BP) to include.
        num_price_groups (int): The number of quantiles to create for price groups.
        grading_ref_cat (str): The reference category for 'mediaGrading' and 'coverGrading' (e.g., 'EX').
        price_ref_cat_num (int): The number of the price group to use as the reference (e.g., 3 for 'Q3').

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The prepared DataFrame.
            - list: The dynamically generated list of price group labels (e.g., ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']).
    """
    print(f"--- Loading data from {file_path} ---")
    df = pd.read_csv(file_path, sep=";", decimal=",")
    print(f"Initial record count: N = {len(df)}")

    # Filter by price cap
    df = df[df['BP'] < price_cap].copy()
    print(f"Record count after filtering for BP < {price_cap}: N = {len(df)}\n")

    # --- Feature Engineering ---
    # Set dynamic reference category for Gradings
    all_grading_cats = ['NM', 'EX', 'VG', 'G' ,'F', 'P']
    all_grading_cats.remove(grading_ref_cat)
    model_grading_order = [grading_ref_cat] + all_grading_cats
    grading_dtype = CategoricalDtype(categories=model_grading_order, ordered=True)
    df['mediaGrading'] = df['mediaGrading'].astype(grading_dtype)
    df['coverGrading'] = df['coverGrading'].astype(grading_dtype)
    print(f"Set '{grading_ref_cat}' as reference for Grading. Order: {model_grading_order}")

    # Dynamically create Price Groups
    price_group_labels = [f'Q{i+1}' for i in range(num_price_groups)]
    df['PriceGroup'] = pd.qcut(df['BP'], q=num_price_groups, labels=price_group_labels, duplicates='drop')
    print(f"Created {num_price_groups} price groups: {price_group_labels}")

    # Set dynamic reference category for Price Groups
    price_ref_cat = f'Q{price_ref_cat_num}'
    model_price_order = list(price_group_labels) # Make a copy
    model_price_order.remove(price_ref_cat)
    model_price_order.insert(0, price_ref_cat)
    df['PriceGroup'] = df['PriceGroup'].cat.reorder_categories(model_price_order, ordered=True)
    print(f"Set '{price_ref_cat}' as reference for PriceGroup. Order: {model_price_order}")

    print("\nData preparation complete.")
    return df, price_group_labels

# ==============================================================================
# 2. DESCRIPTIVE TABLES
# ==============================================================================

def print_descriptive_tables(df, price_group_labels):
    """
    Calculates and prints various descriptive statistics tables.

    Args:
        df (pandas.DataFrame): The prepared DataFrame.
        price_group_labels (list): The list of price group labels for ordering.
    """
    # Define orders for display purposes
    plot_grading_order = ['NM', 'EX', 'VG', 'G', 'F', 'P']
    plot_grading_order_reversed = plot_grading_order[::-1]

    print("\n--- Overall Status Counts (Bought vs. Not Bought) ---")
    print(df['Status'].value_counts())

    print("\n--- Summary of Price Ranges for each Price Group ---")
    price_summary = df.groupby('PriceGroup', observed=False)['BP'].describe()
    print(price_summary.reindex(price_group_labels))

    print("\n--- Summary of Prices (BP) for each Media Grading ---")
    media_price_summary = df.groupby('mediaGrading', observed=False)['BP'].describe()
    print(media_price_summary.reindex(plot_grading_order))

    print("\n--- Summary of Prices (BP) for each Cover Grading ---")
    cover_price_summary = df.groupby('coverGrading', observed=False)['BP'].describe()
    print(cover_price_summary.reindex(plot_grading_order))

    print("\n--- Media Grading Counts within each Status ---")
    media_counts = pd.crosstab(df.mediaGrading, df.Status, margins=True, margins_name="Total")
    print(media_counts.reindex(plot_grading_order + ['Total']))

    print("\n--- Cover Grading Counts within each Status ---")
    cover_counts = pd.crosstab(df.coverGrading, df.Status, margins=True, margins_name="Total")
    print(cover_counts.reindex(plot_grading_order + ['Total']))

    print("\n--- Proportion Bought per Media Grading ---")
    print(df.groupby('mediaGrading', observed=False)['Status'].mean().reindex(plot_grading_order).round(3))

    print("\n--- Proportion Bought per Cover Grading ---")
    print(df.groupby('coverGrading', observed=False)['Status'].mean().reindex(plot_grading_order).round(3))

    print("\n--- Proportion Bought for Media Grading within each Price Group ---")
    media_prop_table = df.groupby(['mediaGrading', 'PriceGroup'], observed=False)['Status'].mean().unstack()
    print(media_prop_table.reindex(plot_grading_order)[price_group_labels].round(3))

    print("\n--- Proportion Bought for Cover Grading within each Price Group ---")
    cover_prop_table = df.groupby(['coverGrading', 'PriceGroup'], observed=False)['Status'].mean().unstack()
    print(cover_prop_table.reindex(plot_grading_order)[price_group_labels].round(3))

    print("\n--- Media Grading Distribution within each Price Group ---")
    media_dist_by_price = pd.crosstab(df.PriceGroup, df.mediaGrading, normalize='index')
    print(media_dist_by_price.reindex(price_group_labels)[plot_grading_order_reversed].round(3))

    print("\n--- Cover Grading Distribution within each Price Group ---")
    cover_dist_by_price = pd.crosstab(df.PriceGroup, df.coverGrading, normalize='index')
    print(cover_dist_by_price.reindex(price_group_labels)[plot_grading_order_reversed].round(3))




# ==============================================================================
# 3. PLOTTING FUNCTIONS
# ==============================================================================

def plot_price_distribution(df):
    """Plots the histogram of the Base Price."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='BP')
    plt.title('Distribution of Base Price', fontsize=16)
    plt.xlabel('Base Price ($)')
    plt.ylabel('Count')
    plt.show()

def plot_sold_proportion_by_grading(df, grading_col):
    """
    Plots a stacked bar chart of sold vs. not sold proportions for a grading column.
    """
    plot_grading_order = ['P', 'F', 'G', 'VG', 'EX', 'NM']
    title_text = 'Media' if grading_col == 'mediaGrading' else 'Cover'
    prop = df.groupby(grading_col, observed=False)['Status'].mean().reset_index().rename(columns={'Status': 'Proportion Sold'})
    prop['Proportion Not Sold'] = 1 - prop['Proportion Sold']
    prop = prop.set_index(grading_col).reindex(plot_grading_order)
    prop[['Proportion Sold', 'Proportion Not Sold']].plot(kind='barh', stacked=True, color=['#4CAF50', '#E0E0E0'], figsize=(8, 6), width=0.8)
    plt.title(f'Proportion Sold vs. Not Sold by {title_text} Grading', fontsize=16)
    plt.xlabel('Proportion')
    plt.ylabel(f'{title_text} Grading')
    plt.legend(['Sold', 'Not Sold'], loc='lower right')
    plt.show()

def plot_sold_proportion_by_price_group(df, grading_col, price_group_labels):
    """
    Plots proportions sold by price group for each category in a grading column.
    This version dynamically creates the correct number of subplots.
    """
    plot_grading_order = ['P', 'F', 'G', 'VG', 'EX', 'NM']
    title_text = 'Media' if grading_col == 'mediaGrading' else 'Cover'
    
    # Filter for grades present in the data to avoid errors with empty pivots
    grades_in_data = [g for g in plot_grading_order if g in df[grading_col].unique()]
    
    pivot = df.groupby([grading_col, 'PriceGroup'], observed=False)['Status'].value_counts(normalize=True).unstack(fill_value=0)
    if 1 not in pivot.columns: pivot[1] = 0
    if 0 not in pivot.columns: pivot[0] = 0
    pivot = pivot.rename(columns={1: 'Proportion Sold', 0: 'Proportion Not Sold'})
    pivot = pivot.reindex(grades_in_data, level=grading_col)

    # --- DYNAMIC SUBPLOT CREATION (THE FIX) ---
    n_grades = len(grades_in_data)
    n_cols = 2 # Let's keep 2 columns for a nice layout
    n_rows = math.ceil(n_grades / n_cols) # Calculate required rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 5), sharex=True, sharey=True)
    fig.suptitle(f'Proportion Sold by Price Group by {title_text} Grading', fontsize=20, y=1.02)
    axes = axes.flatten()

    for i, grade in enumerate(grades_in_data):
        ax = axes[i]
        # Check if the grade exists in the pivot index to avoid errors
        if grade in pivot.index.get_level_values(grading_col):
            data_to_plot = pivot.loc[grade]
            data_to_plot[['Proportion Sold', 'Proportion Not Sold']].reindex(price_group_labels).plot(
                kind='barh', stacked=True, ax=ax, color=['#4CAF50', '#E0E0E0'], width=0.8, legend=False
            )
        ax.set_title(f'{title_text} Grade: {grade}')
        ax.set_ylabel('Price Group')
        ax.set_xlabel('Proportion')

    # Hide any unused subplots if the number of grades isn't perfectly even
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ['Not Sold', 'Sold'], loc='upper right') # Corrected legend order
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()




def plot_overall_grading_distribution(df, plot_grading_order, color_map):
    """Generates pie charts for overall media and cover grading distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Overall Grading Distribution, N = {len(df)}', fontsize=20, y=1.02)
    for i, col in enumerate(['mediaGrading', 'coverGrading']):
        counts = df[col].value_counts().reindex(plot_grading_order)
        axes[i].pie(counts.dropna(), labels=counts.dropna().index, autopct='%1.1f%%',
                    startangle=90, colors=[color_map.get(g) for g in counts.dropna().index])
        axes[i].set_title(col.replace('Grading', ' Grading'), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_stacked_barh_with_labels(data, title, plot_grading_order):
    """Creates a stacked horizontal bar chart from pre-computed crosstab data."""
    ax = data[plot_grading_order].plot(
        kind='barh', stacked=True, figsize=(10, 5), colormap='viridis_r', width=0.8)
    for c in ax.containers:
        labels = [f'{w*100:.1f}%' if w > 0.03 else '' for w in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=10, fontweight='bold')
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel('Proportion', fontsize=12)
    plt.ylabel('Price Group', fontsize=12)
    plt.legend(title='Grade', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# ==============================================================================
# 3. STATISTICAL MODELING (NEWLY ADDED FUNCTIONS)
# ==============================================================================

def run_logit_model(df, formula, model_name):
    """
    Runs a logistic regression model and prints the summary.
    The formula should already include the reference category specifications.
    """
    print(f"\n--- {model_name.upper()} ---")
    try:
        model = smf.logit(formula=formula, data=df)
        result = model.fit(disp=0, cov_type='HC1') # Use robust standard errors
        print(result.summary())
        return result
    except Exception as e:
        print(f"--- MODEL FAILED: {e} ---")
        return None

def plot_interaction_effect(result, df, hue_col, price_group_labels):
    """
    Visualizes the interaction from a fitted model by plotting predicted probabilities.
    
    Args:
        result: The fitted statsmodels result object.
        df (pd.DataFrame): The original dataframe, used to get category levels.
        hue_col (str): The name of the grading column ('mediaGrading' or 'coverGrading').
        price_group_labels (list): The ordered list of price group labels.
    """
    if not result:
        print(f"--- Skipping interaction plot for {hue_col} because model failed. ---")
        return

    title_str = hue_col.replace('Grading', ' Grading')
    plot_grading_order = ['NM', 'EX', 'VG', 'G']
    
    # Create a grid of all combinations to predict on
    grid = pd.MultiIndex.from_product(
        [price_group_labels, df[hue_col].cat.categories],
        names=['PriceGroup', hue_col]
    ).to_frame(index=False)

    # Get predicted probabilities for the grid
    grid['PredictedProbability'] = result.predict(grid)

    # Plot the results
    g = sns.catplot(
        x='PriceGroup', y='PredictedProbability', hue=hue_col, data=grid,
        kind='point', height=6, aspect=1.8,
        order=price_group_labels, hue_order=plot_grading_order,
    )
    g.fig.suptitle(f'Interaction: Probability of Sale by Price Group and {title_str}', fontsize=16, y=1.03)
    g.set_axis_labels("Price Group", "Predicted Probability of Sale")
    g.ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def run_continuous_interaction_model(df, grading_col, grading_ref_cat):
    """
    Runs a logit model with a continuous price interaction term (BP * C(Grading)).
    This model is necessary for simulating price adjustments.
    """
    model_name = f"Simulation Model: Status ~ BP * {grading_col}"
    formula = f"Status ~ BP * C({grading_col}, Treatment(reference='{grading_ref_cat}'))"
    print(f"\n\n--- {model_name.upper()} ---")
    print("Building a new model with continuous price for simulation...")
    try:
        model = smf.logit(formula=formula, data=df)
        result = model.fit(disp=0, cov_type='HC1')
        print(result.summary())
        return result
    except Exception as e:
        print(f"--- SIMULATION MODEL FAILED: {e} ---")
        return None

