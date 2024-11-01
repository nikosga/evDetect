import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing
import statsmodels.api as sm
sns.set()

def time_scan(t, series, l, time_label, min_periods, metric_label):
    tmp = series[(series[time_label] >= min_periods)].copy()
    tmp['event'] = np.where(tmp[time_label]>=t, 1, 0)
    tmp['time_from_event'] = np.where(tmp[time_label]>t, tmp[time_label]-t, 0)
    tmp['event_exp_lt'] = tmp['event']*np.exp(-l*tmp['time_from_event'])
    tmp['Intercept']=1
    X = tmp[['event_exp_lt', time_label, 'Intercept']]
    y = tmp[metric_label]
    model = sm.OLS(y, X).fit()
    return [t, model.params['event_exp_lt'], model.params[time_label], 
            model.pvalues['event_exp_lt'], model.pvalues[time_label],
            model.rsquared, 
            (1 - model.pvalues['event_exp_lt'])*model.params['event_exp_lt']*model.rsquared]

class Detector():
    
    def __init__(self, min_periods=50):
        self.min_periods=min_periods
        self.results=None
        self.detected=False
        self.series=pd.DataFrame()
        self.Lambda=0
        self.metric_label = 'metric'
        self.time_label = 'time'
        self.model=None

    def fit_from_formula(self, series, n_cores=1):

        results = Parallel(n_jobs=n_cores)(delayed(time_scan)(
            t, series, self.Lambda, self.time_label, 
            self.min_periods, self.metric_label
            ) for t in range(self.min_periods+1, len(series)))

        cols = [self.time_label, 'coef', 'trend', 'pval', 'trend_pval', 'rsquared', 'weighted_coef']
        results = pd.DataFrame(results, columns=cols)
        for col in cols:
            if col!=self.time_label:
                results[col] = results[col].round(2)

        maxindex = results['weighted_coef'].idxmax()
        opt_res = results.iloc[[maxindex]]
        if opt_res['weighted_coef'].values[0]>0.01*series[self.metric_label].mean():
            self.detected=True
        else:
            self.detected=False
        results['detected_event'] = np.where((results.index==maxindex) & (self.detected), 1, 0)
        return results

    def __predict(self):
        if self.detected:
            ms = self.series.copy()
            evdate = self.results[self.results.detected_event==1][self.time_label].values[0]
            l=self.Lambda
            ms['event'] = np.where(ms[self.time_label]>=evdate, 1, 0)
            ms['time_from_event'] = np.where(ms[self.time_label]>evdate, ms[self.time_label]-evdate, 0)
            ms['event_exp_lt'] = ms['event']*np.exp(-l*ms['time_from_event'])
            ms['Intercept']=1
            X = ms[['event_exp_lt', self.time_label, 'Intercept']]
            y = ms[self.metric_label]
            model = sm.OLS(y, X).fit()
            self.series[f'fitted_{self.metric_label}'] = model.predict(X)
            return model
        else:
            return None

    def fit(self, series, metric_label='metric', time_label='time', parallel=True):
        if parallel:
            n_cores = min(multiprocessing.cpu_count(),7)
            print(f'Estimating on {n_cores} cores')
        else:
            n_cores=1

        self.metric_label = metric_label
        self.time_label = time_label

        # greedy-search Lambda
        weighted_coefs = {'lambda':[], 'coef':[]}
        for l in np.arange(0, 1, 0.05):
            self.Lambda=l
            self.results = self.fit_from_formula(series, n_cores=n_cores)
            event_row = self.results[self.results.detected_event==1]
            weighted_coefs['coef'].append(event_row['weighted_coef'].values[0])
            weighted_coefs['lambda'].append(l)
        weighted_coefs=pd.DataFrame(weighted_coefs)
        id = weighted_coefs['coef'].idxmax()
        self.Lambda=weighted_coefs.loc[id, 'lambda']

        # fit optimal
        self.results = self.fit_from_formula(series, n_cores=n_cores)
        self.series=series
        self.model = self.__predict()
        return self.model
    
    def summary(self):
        if self.detected:
            return {
                'detected':self.detected,
                'event_time':float(self.results[self.results.detected_event==1][self.time_label].values[0]),
                'event_halflife':float(round(np.log(2)/self.Lambda,1) if self.Lambda!=0 else np.inf),
                'decay_lambda':float(self.Lambda),
                'amplitude':float(self.results[self.results.detected_event==1]['coef'].values[0]),
                'trend':float(self.results[self.results.detected_event==1]['trend'].values[0]),
                'rsquared':float(self.results[self.results.detected_event==1]['rsquared'].values[0])
            }

        else:
            return {}

    def plot(self, save_plot=False):
        plt.figure(figsize=(10,6))
        sns.lineplot(x=self.time_label, y=self.metric_label, data=self.series)
        sns.lineplot(x=self.time_label, y=f'fitted_{self.metric_label}', data=self.series)
        plt.tight_layout()
        if save_plot:
            plt.savefig('figures/plot.png', bbox_inches='tight')
        plt.show()