import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import detrend
import statsmodels.formula.api as smf
sns.set()


class Detector():
    
    def __init__(self, min_periods=30):
        self.min_periods=min_periods
        self.results=None
        self.detected=False
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

        for col in ['coef', 'trend', 'pval', 'trend_pval', 'weighted_coef', 'rsquared']:
            results[col] = results[col].round(2)

        maxindex = results['weighted_coef'].idxmax()
        opt_res = results.iloc[[maxindex]]
        if opt_res['weighted_coef'].values[0]>0.01*series[self.metric_label].mean():
            self.detected=True
        else:
            self.detected=False
        results['detected_event'] = np.where((results.index==maxindex) & (self.detected), 1, 0)
        return results

    def fit(self, series, metric_label='metric', time_label='time'):
        self.metric_label = metric_label
        self.time_label = time_label

        # build regression formula
        formula=f'{self.metric_label} ~ event:np.exp(-l*time_from_event)'
        coef_label='event:np.exp(-l * time_from_event)'
        formula = formula+' + time'

        # greedy-search Lambda
        weighted_coefs = {'lambda':[], 'coef':[]}
        for l in np.arange(0, 1, 0.05):
            self.Lambda=l
            self.results = self.fit_from_formula(series, formula, coef_label)
            event_row = self.results[self.results.detected_event==1]
            weighted_coefs['coef'].append(event_row['weighted_coef'].values[0])
            weighted_coefs['lambda'].append(l)
        weighted_coefs=pd.DataFrame(weighted_coefs)
        id = weighted_coefs['coef'].idxmax()
        self.Lambda=weighted_coefs.loc[id, 'lambda']

        # fit optimal
        self.results = self.fit_from_formula(series, formula, coef_label)

        self.series=series
        self.formula=formula
        self.coef_label=coef_label
        return self
    
    def summary(self):
        if self.detected:
            return {
                'detected':self.detected,
                'optimal_results':self.results[self.results.detected_event==1].to_dict('records'),
                'event_time':self.results[self.results.detected_event==1][self.time_label].values[0],
                'event_halflife':np.log(2)/self.Lambda if self.Lambda!=0 else np.inf,
                'decay_lambda':self.Lambda
            }
        else:
            return {}

    def predict(self):
        if self.detected:
            ms = self.series.copy()
            evdate = self.results[self.results.detected_event==1][self.time_label].values[0]
            l=self.Lambda
            ms['event'] = np.where(ms[self.time_label]>=evdate, 1, 0)
            ms['time_from_event'] = np.where(ms[self.time_label]>evdate, ms[self.time_label]-evdate, 0)
            model = smf.ols(formula=self.formula, data=ms).fit()
            self.series[f'fitted_{self.metric_label}'] = model.predict(ms)
            print(model.summary())

    def plot(self):
        plt.figure(figsize=(10,6))
        sns.lineplot(x=self.time_label, y=self.metric_label, data=self.series)
        sns.lineplot(x=self.time_label, y=f'fitted_{self.metric_label}', data=self.series)
        plt.show()