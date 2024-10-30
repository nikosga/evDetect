import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import detrend
import statsmodels.formula.api as smf
sns.set()

# learn event_type

class Detector():
    
    def __init__(self, min_periods=30, event_type='constant'):
        self.min_periods=min_periods
        self.event_type=event_type
        self.results=None
        self.detected=False
        self.detected_event=None
        self.formula=None
        self.coef_label=None
        self.series=pd.DataFrame()
        self.Lambda=0
        self.metric_label = 'metric'
        self.time_label = 'time'
    
    def fit_from_formula(self, series, formula, coef_label):
        l=self.Lambda
        results = {'time':[], 'coef':[], 'trend':[], 'pval':[], 'trend_pval':[], 'rsquared':[]}
        for t in range(1, len(series)+1):
            if t>self.min_periods and t<len(series)-self.min_periods:
                tmp = series[(series[self.time_label] >= t - self.min_periods) & (series[self.time_label] < t + self.min_periods)].copy()
                tmp['event'] = np.where(tmp[self.time_label]>=t, 1, 0)
                tmp['time_from_event'] = np.where(tmp[self.time_label]>t, tmp[self.time_label]-t, 0)

                model = smf.ols(formula=formula, data=tmp).fit()
                results[self.time_label].append(t)
                results['coef'].append(model.params[coef_label])
                results['trend'].append(model.params[self.time_label])
                results['pval'].append(model.pvalues[coef_label])
                results['trend_pval'].append(model.pvalues[self.time_label])
                results['rsquared'].append(model.rsquared)

        results = pd.DataFrame(results)

        results['weighted_coef'] = (1 - results['pval'])*results['coef']*results['rsquared']

        for col in ['coef', 'pval', 'weighted_coef', 'rsquared']:
            results[col] = results[col].round(2)

        maxindex = results['weighted_coef'].idxmax()
        opt_res = results.iloc[[maxindex]]

        return True, results, opt_res

    def fit(self, series, metric_label='metric', time_label='time'):
        self.metric_label = metric_label
        self.time_label = time_label

        # build regression formula
        if self.event_type=='diminishing':
            formula=f'{self.metric_label} ~ event:np.exp(-l*time_from_event)'
            coef_label='event:np.exp(-l * time_from_event)'
        else:
            formula=f'{self.metric_label} ~ event'
            coef_label='event'

        formula = formula+' + time'

        # greedy-search Lambda
        if self.event_type=='diminishing':
            weighted_coefs = {'lambda':[], 'coef':[]}
            for l in np.arange(0, 1, 0.05):
                self.Lambda=l
                self.detected, self.results, self.detected_event = self.fit_from_formula(series, formula, coef_label)
                weighted_coefs['coef'].append(self.detected_event['weighted_coef'].values[0])
                weighted_coefs['lambda'].append(l)
            weighted_coefs=pd.DataFrame(weighted_coefs)
            id = weighted_coefs['coef'].idxmax()
            self.Lambda=weighted_coefs.loc[id, 'lambda']
        
        self.detected, self.results, self.detected_event = self.fit_from_formula(series, formula, coef_label)

        self.series=series
        self.formula=formula
        self.coef_label=coef_label
        return self
    
    def summary(self):
        return {
            'detected':self.detected,
            'analysis_data':self.results,
            'detected_event':self.detected_event,
            'event_time':None,
            'event_duraton':None,
            'event_type':None
        }

    def predict(self):
        if self.detected:
            ms = self.series.copy()
            evdate = self.detected_event.time.values[0]
            l=self.Lambda
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