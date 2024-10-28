import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
                tmp = series[(series.time >= t - min_periods) & (series.time < t + min_periods)].copy()
                tmp['event'] = np.where(tmp.time>=t, 1, 0)
                tmp['time_from_event'] = np.where(tmp.time>t, tmp.time-t, 0)

                model = smf.ols(formula=formula, data=tmp).fit()
                results['time'].append(t)
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

    def fit(self, series, event_type=None):
        if event_type is not None:
            self.event_type=event_type

        if self.event_type=='diminishing':
            formula='metric ~ event:np.exp(-l*time_from_event)'
            coef_label='event:np.exp(-l * time_from_event)'
        else:
            formula='metric ~ event'
            coef_label='event'

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
            self.series['fitted_metric'] = model.predict(ms)
            print(model.summary())

    def plot(self):
        plt.figure(figsize=(10,6))
        sns.lineplot(x='time', y='metric', data=self.series)
        sns.lineplot(x='time', y='fitted_metric', data=self.series)
        plt.show()