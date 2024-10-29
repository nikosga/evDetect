import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import detrend
import statsmodels.formula.api as smf
sns.set()

class Detector():
    
    def __init__(self):
        self.event_type=None
        self.results=None
        self.detected=False
        self.detected_event=None
        self.formula=None
        self.coef_label=None
        self.series=pd.DataFrame()
        self.elambda=None
        self.metric_label = 'metric'
        self.time_label = 'time'

    def estimate_event_type(self):
        self.event_type='constant'
        return self.event_type
    
    def estimate_lambda(self):
        return 0.12
    
    def fit_from_formula(self, series, formula, coef_label):
        min_periods = 120
        self.elambda=self.estimate_lambda()
        l=self.elambda
        results = {'time':[], 'coef':[], 'pval':[], 'rsquared':[]}
        for t in range(1, len(series)+1):
            if t>min_periods and t<len(series)-min_periods:
                tmp = series[(series[self.time_label] >= t - min_periods) & (series[self.time_label] < t + min_periods)].copy()
                tmp['event'] = np.where(tmp[self.time_label]>=t, 1, 0)
                tmp['time_from_event'] = np.where(tmp[self.time_label]>t, tmp[self.time_label]-t, 0)

                model = smf.ols(formula=formula, data=tmp).fit()
                results[self.time_label].append(t)
                results['coef'].append(model.params[coef_label])
                results['pval'].append(model.pvalues[coef_label])
                results['rsquared'].append(model.rsquared)

        results = pd.DataFrame(results)

        results['weighted_coef'] = (1 - results['pval'])*results['coef']*results['rsquared']

        for col in ['coef', 'pval', 'weighted_coef', 'rsquared']:
            results[col] = results[col].round(2)

        maxindex = results['weighted_coef'].idxmax()
        opt_res = results.iloc[[maxindex]]

        return True, results, opt_res

    def detrend(self, series):
        #model = smf.ols(formula=f'{self.metric_label} ~ {self.time_label}', data=series).fit()
        #series[f'detrend_{self.metric_label}'] = series[self.metric_label] - model.predict(series)
        series[f'detrend_{self.metric_label}'] = detrend(series[self.metric_label], type='linear')
        return series

    def fit(self, series, metric_label='metric', time_label='time', event_type=None, detrend=False):
        self.metric_label = metric_label
        self.time_label = time_label
        #if detrend:
        #    series = self.detrend(series)
        #    self.metric_label=f'detrend_{self.metric_label}'

        if event_type is not None:
            self.event_type=event_type

        if self.event_type=='diminishing':
            formula=f'{self.metric_label} ~ event:np.exp(-l*time_from_event)'
            coef_label='event:np.exp(-l * time_from_event)'
        else:
            formula=f'{self.metric_label} ~ event'
            coef_label='event'

        if detrend:
            formula = formula+' + time'

        self.detected, self.results, self.detected_event = self.fit_from_formula(series, formula, coef_label)

        self.series=series
        self.formula=formula
        self.coef_label=coef_label
        return self
    
    def summary(self):
        return {
            'detected':self.detected,
            'analysis_data':self.results,
            'detected_event':self.detected_event
        }

    def predict(self):
        if self.detected:
            ms = self.series.copy()
            evdate = self.detected_event.time.values[0]
            l=self.elambda
            ms['event'] = np.where(ms.time>=evdate, 1, 0)
            ms['time_from_event'] = np.where(ms.time>evdate, ms.time-evdate, 0)
            model = smf.ols(formula=self.formula, data=ms).fit()
            self.series[f'fitted_{self.metric_label}'] = model.predict(ms)
            print(model.summary())

    def plot(self):
        plt.figure(figsize=(10,6))
        sns.lineplot(x='time', y=self.metric_label, data=self.series)
        sns.lineplot(x='time', y=f'fitted_{self.metric_label}', data=self.series)
        plt.show()