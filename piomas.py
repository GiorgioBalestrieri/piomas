import requests
import pandas as pd
import io
import gzip
import re
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# %% FUNCTIONS
def dayofyear_todatetime(year, serial_day):
    return pd.Timestamp(str(year)) + pd.Timedelta(days=serial_day-1)

def main():
    # %% GET FILE
    r = requests.get('http://psc.apl.uw.edu/wordpress/wp-content/uploads/schweiger/ice_volume/PIOMAS.vol.daily.1979.2018.Current.v2.1.dat.gz')

    assert r.ok, f"Query failed with status code {response.status_code}" + \
                 f"\nFull message: {response.text}"

    # %% PARSE TABLE
    buf = io.BytesIO(r.content)
    gzip_f = gzip.GzipFile(fileobj=buf)
    content = gzip_f.read()
    str_data = content.decode("utf-8")
    str_data = re.sub(' +', ' ', str_data)
    csv = io.StringIO(str_data)
    df = pd.read_csv(csv, sep=' ')

    # %% EXTRACT SERIES
    idx = [dayofyear_todatetime(d, y) for (d, y) in zip(df['Year'], df['#day'])]
    series = pd.Series(df['Vol'].values, index=idx, name='Vol')
    series = series.resample('D').interpolate()                                     # adding missing data (31-12 of leap years) through linear interpolation

    roll_mean = series.rolling(365).mean()
    day_mean = df.groupby('#day')['Vol'].mean().reset_index()
    seasonality = pd.merge(df[['Year', '#day']], day_mean, 'right').sort_values(['Year', '#day'])
    seasonality = pd.Series(seasonality['Vol'].values, index=idx)
    seasonality = seasonality.resample('D').interpolate()
    trend = series - seasonality

    trend.plot(figsize=(25,5))

    # from pandas.plotting import autocorrelation_plot
    # pd.plotting.autocorrelation_plot(trend)

    forecast = ARIMA(trend, order=(5,1,0))
    forecast_fit = forecast.fit(disp=0)

    forecast_fit.summary()

    next_day = (trend.index[-1].dayofyear + 1) % 365
    build_index = lambda x: list(range(x, 366)) + list(range(1, x))
    next_year_index = build_index(next_day)

    next_year = forecast_fit.forecast(365)

    next_year_avg = pd.Series(next_year[0], index=next_year_index)
    next_year_min = pd.Series(list(zip(*next_year[2]))[0], index=next_year_index)
    next_year_max = pd.Series(list(zip(*next_year[2]))[1], index=next_year_index)

    day_mean.set_index('#day', inplace=True)
    day_mean = day_mean.squeeze()
    sorted_day_mean = day_mean.reindex(next_year_index)

    next_year_min += sorted_day_mean
    next_year_avg += sorted_day_mean
    next_year_max += sorted_day_mean

    series_min = series.append(next_year_min, ignore_index=True)
    series_centr = series.append(next_year_avg, ignore_index=True)
    series_max = series.append(next_year_max, ignore_index=True)

    final_index = pd.date_range(start=series.index[0], end=series.index[-1]+pd.Timedelta(days=365), freq='D')

    series_min.index = final_index
    series_centr.index = final_index
    series_max.index = final_index

    # %%
    my_slice = slice(len(series_centr)-730, len(series_centr))
    # my_slice = slice(0, len(series_centr))
    plt.figure(figsize=(20, 5))
    plt.plot(series_centr[my_slice], color='red')
    plt.fill_between(series_min[my_slice].index, series_min[my_slice].values, series_max[my_slice].values, alpha=0.5)

    # %%
    plt.figure(figsize=(20, 5))
    seasonality.plot()

    # %%
    plt.figure(figsize=(20, 5))
    plt.plot(series - seasonality)

    # %% RESAMPLES
    def pivot_months(my_series):
        df = pd.DataFrame(my_series)
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        return df.pivot_table(values=my_series.name, index='Year', columns='Month')

    monthly_min = series.resample('MS').min()
    monthly_max = series.resample('MS').max()
    yearly_min = series.resample('YS').min()
    yearly_mean = series.resample('YS').mean()
    yearly_max = series.resample('YS').max()
    yearly_min.index, yearly_mean.index, yearly_max.index = [yearly_min.index.year] * 3

    df_min_month = pivot_months(monthly_min)
    df_max_month = pivot_months(monthly_max)

    # %%
    plt.figure(figsize=[20, 5])
    plt.fill_between(monthly_min.index, monthly_min.values, monthly_max.values)
    plt.show()

    # %%
    plt.figure(figsize=[20, 5])
    plt.plot(df_min_month)
    labels = list(range(1, 13))
    plt.legend(labels)
    plt.show()

    # %%
    jan = 1
    apr = 4
    sep = 9
    plt.figure(figsize=[20, 5])
    plt.fill_between(df_min_month.index, df_min_month[jan], df_max_month[jan])
    plt.fill_between(df_min_month.index, df_min_month[apr], df_max_month[apr])
    plt.fill_between(df_min_month.index, df_min_month[sep], df_max_month[sep])
    plt.show()

    # %%
    plt.figure(figsize=[20, 5])
    plt.fill_between(yearly_min.index, yearly_min, yearly_max)
    plt.plot(yearly_mean, 'y')
    plt.show()

    # %% SAVE FORECAST
    today = pd.datetime.today()
    series_min.tail(365).to_csv(f'/Users/Giulio/python/piomas/min {today.year}-{today.month}.csv')
    series_centr.tail(365).to_csv(f'/Users/Giulio/python/piomas/central {today.year}-{today.month}.csv')
    series_max.tail(365).to_csv(f'/Users/Giulio/python/piomas/max {today.year}-{today.month}.csv')
    
if __name__ == '__main__':
    main()
