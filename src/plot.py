import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
from matplotlib.lines import Line2D
sns.set_style("whitegrid")

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message="DataFrameGroupBy.apply")

# Some parameters
country_order = ['United States', 'United Kingdom', 'France', 'Canada', 'India']
product_order = ['PRO', 'GURU', 'BUSINESS']

country_order_dict = {country: index for index, country in enumerate(country_order)}
product_order_dict = {product: index for index, product in enumerate(product_order)}

palette = [
    'lightblue',
    '#370031ff',
    'darkblue',
    '#ce8964ff',
    '#eaf27cff',
]
color_dict_country = dict(zip(country_order, palette))

palette = sns.cubehelix_palette(n_colors=3)
color_dict_product = dict(zip(product_order, palette))
colors_pie_product = [color_dict_product[product] for product in product_order]


# Funcs
def plot_compare(df_active_users, df_transactions, df_amount, x_col, hue_col, title='',
                 hue_order=None, hue_color_dict=None, b_detail_period=False,
                 figsize=(15, 3), dpi=100):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    if hue_order is None:
        hue_order = df_active_users[hue_col].unique()
    if hue_color_dict is None:
        hue_color_dict = dict(zip(hue_order, sns.color_palette(n_colors=len(hue_order))))

    # Left plot
    if b_detail_period:
        mask_period = (df_transactions['period'] == 1)
        sns.lineplot(data=df_transactions.loc[mask_period], x=x_col, y='n', hue=hue_col,
                 palette=hue_color_dict, hue_order=hue_order, ax=ax1, ls='--')

        mask_period = (df_transactions['period'] == 12)
        sns.lineplot(data=df_transactions.loc[mask_period], x=x_col, y='n', hue=hue_col,
                 palette=hue_color_dict, hue_order=hue_order, ax=ax1, ls='-')
    else:
        sns.lineplot(data=df_transactions, x=x_col, y='n', hue=hue_col,
                     palette=hue_color_dict, hue_order=hue_order, ax=ax1)
    ax1.set_title(f'Transactions')
    ax1.set_ylabel('n')
    ax1.set_xlabel('')

    # Middle plot
    if b_detail_period:
        mask_period = (df_active_users['period'] == 1)
        sns.lineplot(data=df_active_users.loc[mask_period], x=x_col, y='n', hue=hue_col,
                     palette=hue_color_dict, hue_order=hue_order, ax=ax2, ls='--')

        mask_period = (df_active_users['period'] == 12)
        sns.lineplot(data=df_active_users.loc[mask_period], x=x_col, y='n', hue=hue_col,
                     palette=hue_color_dict, hue_order=hue_order, ax=ax2, ls='-')
    else:
        sns.lineplot(data=df_active_users, x=x_col, y='n', hue=hue_col,
                     palette=hue_color_dict, hue_order=hue_order, ax=ax2)
    ax2.set_title(f'Count of active subscriptions')
    ax2.set_ylabel('n')
    ax2.set_xlabel('')

    # Right plot (amount)
    if b_detail_period:
        mask_period = (df_amount['period'] == 1)
        sns.lineplot(data=df_amount.loc[mask_period], x=x_col, y='amount', hue=hue_col,
                     palette=hue_color_dict, hue_order=hue_order, ax=ax3, ls='--')

        mask_period = (df_active_users['period'] == 12)
        sns.lineplot(data=df_amount.loc[mask_period], x=x_col, y='amount', hue=hue_col,
                     palette=hue_color_dict, hue_order=hue_order, ax=ax3, ls='-')
    else:
        sns.lineplot(data=df_amount, x=x_col, y='amount', hue=hue_col,
                     palette=hue_color_dict, hue_order=hue_order, ax=ax3)
    ax3.set_title(f'MRR')
    ax3.set_ylabel('Amount')
    ax3.set_xlabel('')

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)

    # Custom legend for both product and period
    if b_detail_period:
        combined_legend = []
        for hue in hue_order:
            combined_legend.append(Line2D([0], [0], color=hue_color_dict[hue], ls='-', label=f'{hue} (Anual)'))
            combined_legend.append(Line2D([0], [0], color=hue_color_dict[hue], ls='--', label=f'{hue} (Monthly)'))
        handles = combined_legend
    else:
        handles, labels = ax3.get_legend_handles_labels()

    fig.legend(handles=handles, title=hue_col.capitalize(), loc='upper left', bbox_to_anchor=(1.02, 0.91))

    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()

    plt.tight_layout()
    plt.suptitle(f'{title}', fontsize=12, y=1.05)
    plt.show()


def format_ax_date(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    ax.grid(which='minor', linestyle=':', linewidth='0.2', color='gray')

    return ax


def custom_sort(row):
    return (country_order_dict.get(row['billingCountry'], len(country_order)),
            product_order_dict.get(row['product'], len(product_order)))


def get_contrasting_color_pie(bg_color):
    # get contrasting color for pie chart
    r, g, b = bg_color[:3]
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return 'black' if luminance > 0.5 else 'white'


def get_contrasting_color_bar(bg_color):
    # get contrasting color for bar
    rgb = plt.cm.colors.to_rgb(bg_color)
    brightness = np.sqrt(0.299 * rgb[0]**2 + 0.587 * rgb[1]**2 + 0.114 * rgb[2]**2)
    return 'white' if brightness < 0.5 else 'black'


def plot_anual_vs_monthly_by_product(df, df_transactions, year, product_order, color_dict_product):
    fig, axs = plt.subplots(2, 3, figsize=(15, 6), dpi=100)

    plot_configs = [
        ('Transactions', df_transactions, 'datetime', 'n'),
        ('Active users', df, 'mrr_month', 'n'),
        ('Revenue', df, 'mrr_month', 'mrr')
    ]

    for col, (title, data_source, x_col, y_col) in enumerate(plot_configs):
        for row, period_value in enumerate([12, 1]):
            df_plot = (
                data_source.loc[
                    (data_source['period'] == period_value) &
                    (data_source[x_col].dt.year == year)
                ]
                .set_index(x_col)
            )

            if y_col == 'n':
                df_plot = df_plot.groupby([pd.Grouper(freq='MS'), 'product']).size().reset_index(name='n')
            else:
                df_plot = df_plot[['product', y_col]].groupby([pd.Grouper(freq='MS'), 'product']).sum().reset_index()

            ax = axs[row, col]
            ls = '-'
            if period_value == 1:
                ls = '--'
            sns.lineplot(data=df_plot, x=x_col, y=y_col, ax=ax, hue='product',
                         palette=color_dict_product, hue_order=product_order, ls=ls)

            ax.set_title(f'{"Annual" if period_value == 12 else "Monthly"} - {title}')
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('')
            ax.get_legend().remove()

    plt.tight_layout()
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles[:len(product_order)], labels[:len(product_order)],
               title='Product', loc='center right', bbox_to_anchor=(1.1, 0.5))

    fig.suptitle(f'Comparison of Transactions, Active Users, and Revenue for {year}', y=1.02)
    plt.show()


def plot_simple_pie_chart(counts, title='', color_palette=None):
    n = len(counts)

    if not color_palette:
        palette = sns.cubehelix_palette(n_colors=n, start=.5, rot=-.75)
    else:
        palette = color_palette

    fig, ax = plt.subplots(figsize=(6, 3))
    wedges, texts, autotexts = plt.pie(counts.values, labels=counts.index, colors=palette, autopct='%1.1f%%', startangle=90)

    for wedge, autotext in zip(wedges, autotexts):
        text_color = get_contrasting_color_pie(wedge.get_facecolor())
        autotext.set_color(text_color)

    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_single_baseline_pred(baseline_predictions, baseline_future_predictions, product, palette=palette, b_plot_ci=False):
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create a date range for the x-axis
    historical_len = len(baseline_predictions[product]['Actual'])
    future_len = len(next(iter(baseline_future_predictions[product].values()))['forecast'])
    date_range = pd.date_range(end='2018-03-01', periods=historical_len + future_len, freq='ME')

    # Plot actual data
    ax.plot(date_range[:historical_len],
            baseline_predictions[product]['Actual'],
            label='Actual', linestyle='--', linewidth=2, color=palette[0])

    for i, (model, predictions) in enumerate(baseline_predictions[product].items(), start=1):
        if model != 'Actual':
            # Plot historical predictions
            ax.plot(date_range[:historical_len], predictions,
                    label=model, linewidth=1.5, color=palette[i - 1])

            # Plot future predictions
            future_pred = baseline_future_predictions[product][model]
            full_prediction = np.concatenate([predictions[-1:], future_pred['forecast']])
            ax.plot(date_range[historical_len - 1:], full_prediction,
                    linestyle='--', linewidth=1.5, color=palette[i - 1])

            # Plot confidence intervals
            if b_plot_ci:
                ci_dates = date_range[historical_len:]
                ax.fill_between(ci_dates, future_pred['lower_ci'], future_pred['upper_ci'],
                                color=palette[i - 1], alpha=0.2)

    ax.set_title(f'{product} - Actual vs Predictions with Confidence Intervals', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax = format_ax_date(ax=ax)
    plt.tight_layout()
    plt.show()


def plot_baseline_predictions(baseline_predictions, baseline_future_predictions, product_order, palette=palette):
    for product in product_order:
        plot_single_baseline_pred(baseline_predictions, baseline_future_predictions, product, palette=palette)


def plot_single_sarima_pred(sarima_observed, sarima_forecast, product):

    data = sarima_observed[product]
    dict_forecast = sarima_forecast[product]

    forecast = dict_forecast['forecast']
    lower_ci = dict_forecast['lower_ci']
    upper_ci = dict_forecast['upper_ci']

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label='Observed')
    forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=len(forecast), freq='MS')
    plt.plot(forecast_index, forecast, color='r', label='Forecast')
    plt.fill_between(forecast_index,
                     lower_ci,
                     upper_ci,
                     color='pink', alpha=0.3)
    plt.title(f'{product} - SARIMAX Forecast')
    plt.xlabel('Date')
    plt.ylabel('MRR')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_sarima_predictions(sarima_observed, sarima_forecasts):
    for product in product_order:
        plot_single_sarima_pred(sarima_observed, sarima_forecasts, product)


def plot_prophet_predictions(prophet_data, prophet_forecasts):
    plt.figure(figsize=(15, 5 * len(prophet_data)))

    for i, (product, forecast) in enumerate(prophet_forecasts.items(), 1):
        plt.subplot(len(prophet_data), 1, i)
        plt.plot(prophet_data[product]['ds'], prophet_data[product]['y'], label='Actual')
        plt.plot(forecast['ds'], forecast['yhat'], color='red', label='Forecast')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)
        plt.title(f'{product} MRR Forecast')
        plt.legend()

    plt.tight_layout()
    plt.show()
