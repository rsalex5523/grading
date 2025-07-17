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
    # Define orders for display purposes
    plot_grading_order = ['NM', 'EX', 'VG', 'G']
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



def plot_overall_grading_distribution(df, plot_grading_order, color_map):
    """
    Generates and displays two pie charts for the overall distribution of
    media and cover gradings.

    Args:
        df (pd.DataFrame): The input dataframe.
        plot_grading_order (list): The order to display grades.
        color_map (dict): A dictionary mapping grades to colors.
    """
    print("\n--- Generating Overall Grading Distribution Pie Charts ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Overall Grading Distribution', fontsize=20, y=1.02)

    # Media Grading Pie Chart
    media_counts = df['mediaGrading'].value_counts().reindex(plot_grading_order)
    axes[0].pie(
        media_counts.dropna(),
        labels=media_counts.dropna().index,
        autopct='%1.1f%%',
        startangle=90,
        colors=[color_map.get(g) for g in media_counts.dropna().index]
    )
    axes[0].set_title('Media Grading', fontsize=16)
    axes[0].axis('equal')

    # Cover Grading Pie Chart
    cover_counts = df['coverGrading'].value_counts().reindex(plot_grading_order)
    axes[1].pie(
        cover_counts.dropna(),
        labels=cover_counts.dropna().index,
        autopct='%1.1f%%',
        startangle=90,
        colors=[color_map.get(g) for g in cover_counts.dropna().index]
    )
    axes[1].set_title('Cover Grading', fontsize=16)
    axes[1].axis('equal')

    plt.tight_layout()
    plt.show()


def plot_grading_by_status(df, grading_col, color_map, save_path=None):
    """
    Generates pie charts comparing grading distribution for 'Bought' vs. 'Not Bought' items.

    Args:
        df (pd.DataFrame): The input dataframe.
        grading_col (str): The grading column to analyze ('mediaGrading' or 'coverGrading').
        color_map (dict): A dictionary mapping grades to colors.
        save_path (str, optional): Path to save the figure. Defaults to None.
    """
    print(f"\n--- Generating {grading_col} Distribution by Purchase Status ---")
    df_bought = df[df['Status'] == 1]
    df_not_bought = df[df['Status'] == 0]

    bought_counts = df_bought[grading_col].value_counts().sort_index()
    not_bought_counts = df_not_bought[grading_col].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    title_str = ' '.join(word.capitalize() for word in grading_col.replace('Grading', ' Grading').split())
    fig.suptitle(f'Composition of {title_str} for Bought vs. Not Bought Groups', fontsize=16)

    axes[0].pie(bought_counts, labels=bought_counts.index, autopct='%1.1f%%', startangle=90, colors=[color_map.get(g) for g in bought_counts.index])
    axes[0].set_title('"Bought" Group')

    axes[1].pie(not_bought_counts, labels=not_bought_counts.index, autopct='%1.1f%%', startangle=90, colors=[color_map.get(g) for g in not_bought_counts.index])
    axes[1].set_title('"Not Bought" Group')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_stacked_barh_with_labels(data, title, plot_grading_order, colormap='viridis'):
    """
    Creates a stacked horizontal bar chart with percentage labels inside each segment.

    Args:
        data (pd.DataFrame): Crosstab/pivot table of proportions.
        title (str): The title for the plot.
        plot_grading_order (list): The order for grading categories.
        colormap (str, optional): Matplotlib colormap. Defaults to 'viridis'.
    """
    ax = data[plot_grading_order].plot(
        kind='barh',
        stacked=True,
        figsize=(14, 8),
        colormap=colormap,
        width=0.8
    )

    # Add percentage labels to each segment
    for c in ax.containers:
        # Each container is a set of bars for a given grade
        labels = [f'{w*100:.1f}%' if w > 0.03 else '' for w in c.datavalues] # Only label segments > 3%
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=10, fontweight='bold')

    plt.title(title, fontsize=18, pad=20)
    plt.xlabel('Proportion', fontsize=12)
    plt.ylabel('Price Group', fontsize=12)
    plt.legend(title='Grade', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_interaction_effect(df, x_col, y_col, hue_col, order, hue_order, title):
    """
    Visualizes the interaction between two categorical variables on a continuous outcome
    using a point plot.

    Args:
        df (pd.DataFrame): The input dataframe.
        x_col (str): The column for the x-axis.
        y_col (str): The column for the y-axis (the outcome).
        hue_col (str): The column for the hue variable (the moderator).
        order (list): The order of categories for the x-axis.
        hue_order (list): The order of categories for the hue.
        title (str): The title for the plot.
    """
    print(f"\n--- Visualizing Interaction for {hue_col} ---")
    g = sns.catplot(
        x=x_col, y=y_col, hue=hue_col, data=df,
        kind='point', height=6, aspect=1.8,
        order=order, errorbar=None,
        hue_order=hue_order
    )
    g.fig.suptitle(title, fontsize=16, y=1.03)
    g.set_axis_labels("Price Group ($)", "Probability of being Bought")
    g.ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# ==============================================================================
# STATISTICAL MODELING FUNCTIONS
# ==============================================================================

def run_logit_model(formula, data, model_name, plot_grading_order):
    """
    Runs a logistic regression model, prints the summary, and predicted probabilities.

    Args:
        formula (str): The model formula for statsmodels.
        data (pd.DataFrame): The dataframe to use.
        model_name (str): A descriptive name for the model.
        plot_grading_order (list): The order for displaying grades in predictions.

    Returns:
        statsmodels.results.results.ResultWrapper: The fitted model result.
    """
    print(f"\n--- {model_name.upper()} ---")
    model = smf.logit(formula=formula, data=data)
    result = model.fit(disp=0, cov_type='HC1') # Use robust standard errors
    print(result.summary())

    # --- Predict and display probabilities ---
    predictor_col = formula.split('~')[1].strip().replace('C(', '').replace(')', '')
    print(f"\n--- Predicted Probability of Sale by {predictor_col} (from {model_name}) ---")
    grades_to_predict = pd.DataFrame({predictor_col: data[predictor_col].cat.categories})
    pred_prob = result.predict(grades_to_predict)
    grades_to_predict['PredictedProbability'] = pred_prob
    print(grades_to_predict.set_index(predictor_col).reindex(plot_grading_order).round(3))

    return result


def run_interaction_logit_model(formula, data, model_name, grading_col, price_col, plot_grading_order, quintile_labels):
    """
    Runs a logistic regression model with an interaction term.

    Args:
        formula (str): The model formula for statsmodels.
        data (pd.DataFrame): The dataframe to use.
        model_name (str): A descriptive name for the model.
        grading_col (str): The name of the grading column.
        price_col (str): The name of the price group column.
        plot_grading_order (list): Order for rows in the probability table.
        quintile_labels (list): Order for columns in the probability table.

    Returns:
        statsmodels.results.results.ResultWrapper or None: The fitted model result, or None if it fails.
    """
    print(f"\n--- {model_name.upper()} ---")
    model = smf.logit(formula=formula, data=data)
    try:
        result = model.fit(disp=0, cov_type='HC1')
        print(result.summary())

        # --- Predict and display probabilities ---
        print(f"\n--- Predicted Probability of Sale by Grade and Price (from {model_name}) ---")
        all_combinations = pd.MultiIndex.from_product(
            [data[grading_col].cat.categories, data[price_col].cat.categories],
            names=[grading_col, price_col]
        ).to_frame(index=False)
        pred_prob = result.predict(all_combinations)
        all_combinations['PredictedProbability'] = pred_prob
        prob_table = all_combinations.pivot(index=grading_col, columns=price_col, values='PredictedProbability')
        print(prob_table.reindex(plot_grading_order)[quintile_labels].round(3))

        return result

    except Exception as e:
        print(f"\n--- MODEL FAILED TO CONVERGE OR ERRORED: {e} ---")
        return None

