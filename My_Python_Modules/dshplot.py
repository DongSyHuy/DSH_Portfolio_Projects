"""Functions for Data Visualization"""

import matplotlib.pyplot as plt
import seaborn as sns

# !pip install plottable
import plottable as ptt
import plottable.plots as pttp
import plottable.formatters as pttf
import time, re, math

sns.set_style("darkgrid")
import numpy as np
import pandas as pd
from IPython.display import display


# Timing functions
def timing__decorator(func):
    """
    https://towardsdatascience.com/python-decorators-for-data-science-6913f717669a
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        running_time = round((end_time - start_time), 2)
        (
            print(f"\nExecution time: {running_time} sec.\n")
            if running_time < 60
            else print(
                f"\nExecution time:  {running_time//60} min and {math.ceil(running_time%60)} sec.\n"
            )
        )
        return result

    return wrapper


#
def background_gradient(styler):
    styler.background_gradient(
        cmap=sns.diverging_palette(220, 10, center="dark", as_cmap=True), axis=None
    )
    styler.map(lambda x: "text-align: center")
    return styler


def my_format_table(
    df: pd.DataFrame,
    background_gradient=False,
    high_light: list = None or ["max", 0] or ["min", 0],
    format_dict: dict = None or "{:.2f}",
    float_precision: int = 1,
):
    """Format table
    {'col1': '{:.2f}', 'col2': '£ {:.1f}'}
    high_light = ["max", 0]: max value, axis = 0 (max value in each column)
    """

    if background_gradient is True:
        return (
            df.copy()
            .style.pipe(background_gradient)
            .format(format_dict, precision=float_precision)
        )
    if high_light is not None:
        if high_light[0] == "max":
            return (
                df.copy()
                .style.highlight_max(color="indianred", axis=high_light[1])
                .format(format_dict, precision=float_precision)
            )
        elif high_light[0] == "min":
            return (
                df.copy()
                .style.highlight_min(color="indianred", axis=high_light[1])
                .format(format_dict, precision=float_precision)
            )
    else:
        return df.copy().style.format(format_dict, precision=float_precision)


# KDE plot for numerical variable
@timing__decorator
def my_kdeplot_multivars_vs_1cat(
    df: pd.DataFrame,
    x_col: list,
    hue_col: list = None,
    subplot_col: int = 2,
    figsize: tuple = (4, 6),
):
    """KDE plot for numerical variable"""
    df1 = df[x_col].copy()
    sns.set_palette("colorblind")
    cols = subplot_col
    rows = int(np.ceil(len(x_col) / cols))
    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(rows * figsize[0], rows * figsize[1]),
        # constrained_layout=True,
    )
    axes_flat = ax.flatten()
    for i in range(rows):
        for j in range(cols):
            ax_index = cols * i + j
            if ax_index >= len(x_col):
                break
            if rows == 1:
                h = hue_col
                g = sns.kdeplot(
                    data=df1,
                    x=x_col[ax_index],
                    hue=hue_col,
                    ax=ax[j],
                    fill=True,
                )
            else:
                g = sns.kdeplot(
                    data=df1,
                    x=x_col[ax_index],
                    hue=hue_col,
                    ax=ax[i, j],
                    fill=True,
                )
            g.set_xlabel('"' + x_col[ax_index] + '" variable')
            g.set_ylabel("")
            g.tick_params(axis="both")

    # Remove any extra empty subplots if the number of variables is less than coloumns
    if len(x_col) < rows * cols:
        for jj in range(len(x_col), rows * cols):
            fig.delaxes(axes_flat[jj])
    # plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()


# Count plot for categorical variable
@timing__decorator
def my_countplot_multicats_vs_1cat(
    df: pd.DataFrame,
    x_col: list,
    y_col: list = None,
    hue_col: list = None,
    bar_format={"count": "%g"} or {"proportion": "{:.2%}"},
    subplot_col: int = 2,
    figsize: tuple = (4, 6),
):
    """Count plot for categorical variable"""
    df1 = df.copy()
    for k, v in bar_format.items():
        k, v
    len_x_col = len(x_col) if x_col is not None else len(y_col)
    # sns.set_palette("colorblind")
    cols = subplot_col
    rows = int(np.ceil(len_x_col / cols))
    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(rows * figsize[0], rows * figsize[1]),
        constrained_layout=True,
    )
    axes_flat = ax.flatten()
    for i in range(rows):
        for j in range(cols):
            ax_index = cols * i + j

            if ax_index >= len_x_col:
                break
            if rows == 1:
                if hue_col == None:
                    h = x_col[ax_index] if x_col is not None else y_col[ax_index]
                else:
                    h = hue_col
                # h = x_col[ax_index] if hue_col == None else hue_col
                g = sns.countplot(
                    data=df1,
                    x=x_col[ax_index] if x_col is not None else None,
                    y=None if y_col is None else y_col[ax_index],
                    hue=h,
                    # orient='v',
                    stat=k,
                    palette="colorblind",
                    ax=ax[j],
                    width=0.5,
                )
            else:
                if hue_col == None:
                    h = x_col[ax_index] if x_col is not None else y_col[ax_index]
                else:
                    h = hue_col
                # h = x_col[ax_index] if hue_col == None else hue_col
                g = sns.countplot(
                    data=df1,
                    x=x_col[ax_index] if x_col is not None else None,
                    y=None if y_col is None else y_col[ax_index],
                    hue=h,
                    # orient='v',
                    stat=k,
                    palette="colorblind",
                    ax=ax[i, j],
                    width=0.5,
                )
            l = x_col[ax_index] if x_col is not None else y_col[ax_index]
            g.set_xlabel(f"'{l}' variable")
            g.set_ylabel("")
            g.tick_params(axis="both")

            for c in range(len(g.containers)):
                g.bar_label(g.containers[c], fmt=v, weight="bold")
            # if hue_col != None:
            #     g.bar_label(g.containers[1], weight="bold")
            # g.bar_label(g.containers[2], weight='bold', fontsize=40)

    # Remove any extra empty subplots if the number of variables is less than coloumns
    if len_x_col < rows * cols:
        for jj in range(len_x_col, rows * cols):
            fig.delaxes(axes_flat[jj])
    plt.show()


# @timing__decorator
def my_vertical_bar_plot(df, col_x, col_y):
    _, ax = plt.subplots(figsize=(6, 4))
    color = sns.color_palette(palette="colorblind")

    _ = sns.barplot(
        data=df, x=col_x, y=col_y, hue=None, palette="colorblind", ax=ax, width=0.7
    )
    _.bar_label(_.containers[0])


#
def my_barplot(
    df: pd.DataFrame,
    x,
    y,
    hue,
    errorbar=None,
    palette="colorblind",
    ax=None,
    figsize: tuple = (10, 8),
    title_size: list = ["Bar plot", "center", 12, 6.0],
    x_label: list = ["x", 8, 90, None],
    y_label: list = ["y", 8, 0, None],
    hide_ticklabels=False or "x" or "y",
    bar_label: list = ["edge", 0, "%g", 7, "bold"],
):
    """
    {:.2%}
    x_label: list = [x-axis label, tick size, tick rotation, (xlim)],\n
    title_size= ["title_name", loc, font_size, pad],\n
    bar_label=[label_type='edge', padding, format, font_size, font_weight]\n
    """
    if ax is None:
        pl, _ = plt.subplots(figsize=figsize)

    pl = sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        errorbar=errorbar,
        palette=palette,
        # width=0.5,
        ax=ax,
    )
    for c in range(len(pl.containers)):
        pl.bar_label(
            pl.containers[c],
            label_type=bar_label[0],
            padding=bar_label[1],
            fmt=bar_label[2],
            fontsize=bar_label[3],
            weight=bar_label[4],
        )
    pl.tick_params(axis="x", labelrotation=x_label[2], labelsize=x_label[1])
    pl.tick_params(axis="y", labelrotation=y_label[2], labelsize=y_label[1])
    pl.set(xlim=x_label[3], ylabel=y_label[0], xlabel=x_label[0], ylim=y_label[3])
    if hide_ticklabels == "x":
        pl.set(xticklabels=[])
    elif hide_ticklabels == "y":
        pl.set(yticklabels=[])
    pl.set_title(
        title_size[0],
        loc=title_size[1],
        size=title_size[2],
        pad=title_size[3],
        y=1,
    )


#
# @timing__decorator
def my_plottable_bars_stars_progress(
    df: pd.DataFrame,
    plot: str,
    cols: list,
    is_pct=False,
    palette: str = "colorblind",
    figsize: tuple = (8, 4),
):
    """
    plot_fn=percentile_bars for percentile bars\n
    plot_fn=percentile_stars for a start review\n
    plot_fn=progress_donut for a donut progress]\n
    Ex:\n
        "is_pct": True. This means that our values are in percentage.\n
        "formatter": "{:.0%}" percentage formatting
    """

    # Define a colormap from matplotlib
    cmap = sns.set_palette(palette=palette)
    # Init a figure
    fig, ax = plt.subplots(figsize=figsize)

    col_defs = (
        [
            ptt.ColumnDefinition(
                i, plot_fn=plot, plot_kw={"cmap": cmap, "is_pct": is_pct}
            )
            for i in cols
        ]
        if plot != pttp.progress_donut
        else [
            ptt.ColumnDefinition(
                i, plot_fn=plot, plot_kw={"is_pct": is_pct, "formatter": "{:.0%}"}
            )
            for i in cols
        ]
    )
    # Create the Table() object
    tab = ptt.Table(
        df.copy(),
        cell_kw={"linewidth": 0, "edgecolor": "k"},
        textprops={"ha": "center"},
        column_definitions=col_defs,
    )

    # Display the output
    plt.show()


# Box plot for num
@timing__decorator
def my_boxplot_multinums_vs_1cat(
    df: pd.DataFrame,
    x_col: list,
    y_cat_col: list = None,
    hue_col: list = None,
    subplot_col: int = 2,
    figsize: tuple = (6, 4),
):
    """Box plot for numerical variable"""
    df1 = df[x_col].copy()

    sns.set_palette("colorblind")
    cols = subplot_col
    rows = int(np.ceil(len(x_col) / cols))
    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(rows * figsize[0], rows * figsize[1]),
        # constrained_layout=True,
    )
    axes_flat = ax.flatten()
    for i in range(rows):
        for j in range(cols):
            ax_index = cols * i + j
            if ax_index >= len(x_col):
                break
            if rows == 1:
                g = sns.boxplot(
                    data=df1,
                    x=x_col[ax_index],
                    y=y_cat_col[ax_index] if y_cat_col is not None else None,
                    ax=ax[j],
                    hue=hue_col[ax_index] if hue_col is not None else None,
                    # color='white',
                    width=0.3,
                    medianprops={"linestyle": "-", "color": "white", "linewidth": 2},
                )
            else:
                g = sns.boxplot(
                    data=df1,
                    x=x_col[ax_index],
                    y=y_cat_col[ax_index] if y_cat_col is not None else None,
                    hue=hue_col[ax_index] if hue_col is not None else None,
                    # color='white',
                    ax=ax[i, j],
                    width=0.3,
                    medianprops={"linestyle": "-", "color": "white", "linewidth": 2},
                )
            g.set_xlabel('"' + x_col[ax_index] + '" variable')
            g.tick_params(axis="both")
    # Remove any extra empty subplots if the number of variables is less than coloumns
    if len(x_col) < rows * cols:
        for jj in range(len(x_col), rows * cols):
            fig.delaxes(axes_flat[jj])
    # plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()


# Plot continuous
@timing__decorator
def my_kde_boxplot_cont_vs_cat(
    df: pd.DataFrame, cont_vars: list, cat_var: str, figsize: tuple = (6, 4)
):
    """Plot 1 kde with 1 boxplot"""

    for i in cont_vars:
        sns.set_palette("colorblind")
        df1 = df.copy()
        df1[cat_var] = df1[cat_var].astype("category")

        fig, (ax1, ax2) = plt.subplots(
            2, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": (0.7, 0.3)}
        )
        sns.kdeplot(
            data=df1, x=i, hue=cat_var, fill=True, alpha=0.5, legend=True, ax=ax1
        )
        sns.boxplot(x=i, y=cat_var, hue=cat_var, data=df1, legend=False, ax=ax2)
        # ax1.set_ylabel("")
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        # ax2.get_legend().remove()
        ax2.set_ylabel("")

        plt.tight_layout()
        plt.show()


#
@timing__decorator
def my_plot_2vars_vs_3cats(
    df: pd.DataFrame,
    plot: list,
    row_cat_var: str,
    col_cat_var: str,
    x_var: str,
    y_var: str,
    hue_cat_var: str,
    tick: list = ["x", "both", 0],
    x_y_label: list = None,
):
    """
    sns.kdeplot, sns.countplot, sns.scatterplot, sns.stripplot for 2 continous/discrete variables vs 3 categorical variables\n
    Ex:\n
        plot = [plot, palette, alpha, fill]\n
            plot=[sns.countplot, "colorblind", 0.5, True]\n
        tick = tick_params = (axis='x', which=t'both', rotation=0)\n
    """
    sns.set_palette(palette=plot[1])

    fac = sns.FacetGrid(
        df.copy(), row=row_cat_var, col=col_cat_var, hue=hue_cat_var, margin_titles=True
    )
    if plot[0] in [sns.kdeplot, sns.countplot, sns.barplot]:
        fac.map_dataframe(
            plot[0],
            data=df,
            x=x_var,
            y=y_var,
            alpha=plot[2] if plot[2] is not None else None,
            fill=plot[3] if plot[3] is not None else None,
        )
    elif plot[0] in [sns.scatterplot, sns.stripplot]:
        fac.map_dataframe(
            plot[0],
            data=df,
            x=x_var,
            y=y_var,
            alpha=plot[2] if plot[2] is not None else None,
        )
    # fac.map_dataframe(plot[0], data=df, x= num_var_x, y=num_var_y )
    fac.tick_params(axis=tick[0], which=tick[1], rotation=tick[2])
    fac.set_axis_labels(x_y_label[0], x_y_label[1]) if x_y_label is not None else None
    fac.add_legend()


# Heatmap for correlation
# @timing__decorator
def my_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    interpretations: dict = None,
    figsize: tuple = (10, 8),
    title_and_num_size: list = [12, 10],
    x_label: list = [10, 90, "right"],
    y_label: list = [10, 0, "right"],
):
    """
    Heatmap for correlation\n
    method = {'pearson', 'kendall', 'spearman'}\n
    title_and_num_size: list = [12, 10],\n
    x_label: list = [fontsize, rotation, "right" | "center" | "left"],\n
    y_label: list = [fontsize, rotation, "right" | "center" | "left"],\n
    interpretations={'cardio':['age', 'gender']}
    """
    # Generate a mask for the upper triangle
    corr_df = df.corr(method=method.lower(), min_periods=1, numeric_only=False)

    mask = np.triu(np.ones_like(corr_df))

    pl, ax = plt.subplots(figsize=figsize)
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    pl = sns.heatmap(
        corr_df,
        cmap=colormap,
        square=True,
        cbar_kws={"shrink": 1},
        ax=ax,
        annot=True,
        linewidths=0.1,
        vmax=1.0,
        linecolor="white",
        annot_kws={"fontsize": title_and_num_size[1]},
        mask=mask,
    )
    pl.set_xticklabels(
        pl.get_xticklabels(),
        fontsize=x_label[0],
        rotation=x_label[1],
        ha=x_label[2],
        rotation_mode="anchor",
    )
    pl.set_yticklabels(
        pl.get_yticklabels(),
        fontsize=y_label[0],
        rotation=y_label[1],
        ha=y_label[2],
        rotation_mode="anchor",
    )
    plt.title(
        method.capitalize() + " Correlation of Features",
        y=1,
        size=title_and_num_size[0],
    )

    plt.show()

    if interpretations is not None:
        for k, v in interpretations.items():
            for i in v:
                if corr_df.loc[i, k] >= -1 and corr_df.loc[i, k] < -0.7:
                    print(
                        f"There is a 'Very strong negative correlation' between '{i}' and '{k}'."
                    )
                elif corr_df.loc[i, k] >= -0.7 and corr_df.loc[i, k] < -0.5:
                    print(
                        f"There is a 'Strong negative correlation' between '{i}' and '{k}'."
                    )
                elif corr_df.loc[i, k] >= -0.5 and corr_df.loc[i, k] < -0.3:
                    print(
                        f"There is a 'Moderate negative correlation' between '{i}' and '{k}'."
                    )
                elif corr_df.loc[i, k] >= -0.3 and corr_df.loc[i, k] < 0:
                    print(
                        f"There is a 'Weak negative correlation' between '{i}' and '{k}'."
                    )
                elif corr_df.loc[i, k] == 0:
                    print(f"There is 'No correlation' between '{i}' and '{k}'.")
                elif corr_df.loc[i, k] > 0 and corr_df.loc[i, k] <= 0.3:
                    print(
                        f"There is a 'Weak positive correlation' between '{i}' and '{k}'."
                    )
                elif corr_df.loc[i, k] > 0.3 and corr_df.loc[i, k] <= 0.5:
                    print(
                        f"There is a 'Moderate positive correlation' between '{i}' and '{k}'."
                    )
                elif corr_df.loc[i, k] > 0.5 and corr_df.loc[i, k] <= 0.7:
                    print(
                        f"There is a 'Strong positive correlation' between '{i}' and '{k}'."
                    )
                elif corr_df.loc[i, k] > 0.7 and corr_df.loc[i, k] <= 1:
                    print(
                        f"There is a 'Very strong positive correlation' between '{i}' and '{k}'."
                    )


# def plot_pie_barchart(df, var, title=""):
#     plt.figure(figsize=(12, 8))

#     plt.subplot(1, 2, 1)
#     label_list = list(df[var].value_counts().index)
#     colors = sns.color_palette("colorblind")
#     plt.pie(
#         df[var].value_counts(),
#         autopct="%1.1f%%",
#         colors=colors,
#         startangle=60,
#         labels=label_list,
#         wedgeprops={"linewidth": 2, "edgecolor": "white"},
#         shadow=False,
#         textprops={"fontsize": 25},
#     )
#     plt.title("Distribution of " + var + " variable " + title, fontsize=25)

#     plt.subplot(1, 2, 2)
#     bar_plot = sns.countplot(x=var, data=df, palette="colorblind")
#     bar_plot.set_title(
#         "Count of " + var + " cases " + title,
#         loc="center",
#         fontdict={
#             "fontsize": 25
#             # , 'fontweight': 'bold'
#         },
#     )
#     bar_plot.bar_label(bar_plot.containers[0], fontsize=20)
#     plt.ylabel("Count", fontsize=20)
#     plt.xlabel(var, fontsize=20)
#     plt.tick_params(axis="both", which="major", labelsize=15)
#     plt.show()
