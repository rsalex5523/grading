import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype

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
    all_grading_cats = ['NM', 'EX', 'VG', 'G']
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
    plot_grading_order = ['NM', 'EX', 'VG', 'G']
    
    print("\n--- Summary of Price Ranges for each Quintile ---")
    price_summary = df.groupby('PriceGroup', observed=False)['BP'].describe()
    print(price_summary.reindex(price_group_labels))

    print("\n--- Summary of Prices (BP) for each Media Grading ---")
    media_price_summary = df.groupby('mediaGrading', observed=False)['BP'].describe()
    print(media_price_summary.reindex(plot_grading_order))

    print("\n--- Proportion Bought for Media Grading within each Price Group ---")
    media_prop_table = df.groupby(['mediaGrading', 'PriceGroup'], observed=False)['Status'].mean().unstack()
    print(media_prop_table.reindex(plot_grading_order)[price_group_labels].round(3))

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
    plot_grading_order = ['G', 'VG', 'EX', 'NM']
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
    """
    plot_grading_order = ['G', 'VG', 'EX', 'NM']
    title_text = 'Media' if grading_col == 'mediaGrading' else 'Cover'
    pivot = df.groupby([grading_col, 'PriceGroup'], observed=False)['Status'].value_counts(normalize=True).unstack(fill_value=0)
    pivot = pivot.rename(columns={1: 'Proportion Sold', 0: 'Proportion Not Sold'})
    pivot = pivot.reindex(plot_grading_order, level=grading_col)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle(f'Proportion Sold by Price Group by {title_text} Grading', fontsize=20, y=1.02)
    axes = axes.flatten()
    for i, grade in enumerate(plot_grading_order):
        ax = axes[i]
        data_to_plot = pivot.loc[grade]
        data_to_plot[['Proportion Sold', 'Proportion Not Sold']].reindex(price_group_labels).plot(
            kind='barh', stacked=True, ax=ax, color=['#4CAF50', '#E0E0E0'], width=0.8, legend=False
        )
        ax.set_title(f'{title_text} Grade: {grade}')
        ax.set_ylabel('Price Group')
        ax.set_xlabel('Proportion')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ['Sold', 'Not Sold'], loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
