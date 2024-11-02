import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing
import statsmodels.api as sm
sns.set()


class Detector():

    def __init__(self, min_periods=50, n_cores=7):
        """The evDetect main class for parametric event detection & inference 

        Args:
            min_periods (int, optional): Minimum number of periods to start scanning for the event time. Defaults to 50.
            n_cores (int, optional): Number of cores for the event time search. Defaults to 7.
        """
        self.min_periods=min_periods
        if n_cores>multiprocessing.cpu_count():
            self.n_cores=multiprocessing.cpu_count()
        else:
            self.n_cores=n_cores
        self.results=None
        self.detected=False
        self.series=pd.DataFrame()
        self.Lambda=0
        self.metric_label = 'metric'
        self.time_label = 'time'
        self.model=None

    def fit_model(self, t, l, store_model=False, store_fitted=False):
        """This function fits a single model given the event time and lambda parameters

        Args:
            t (_type_): _description_
            l (_type_): _description_
            store_model (bool, optional): Whether you would like to store the model within the detector object. Defaults to False.
            store_fitted (bool, optional): Whether you would like to store the predictions in the data as a new column. Defaults to False.

        Returns:
            list: Results from model fitting process
        """
        tmp = self.series.copy()
        tmp['event'] = np.where(tmp[self.time_label]>=t, 1, 0)
        tmp['time_from_event'] = np.where(tmp[self.time_label]>t, tmp[self.time_label]-t, 0)
        tmp['event_exp_lt'] = tmp['event']*np.exp(-l*tmp['time_from_event'])
        tmp['Intercept']=1
        X = tmp[['event_exp_lt', self.time_label, 'Intercept']]
        y = tmp[self.metric_label]
        model = sm.OLS(y, X).fit()
        if store_model:
            self.model = model
        if store_fitted:
            self.series[f'fitted_{self.metric_label}'] = model.predict(X)

        return [t, model.params['event_exp_lt'], model.params[self.time_label], 
                model.pvalues['event_exp_lt'], model.pvalues[self.time_label],
                model.rsquared, 
                (1 - model.pvalues['event_exp_lt'])*model.params['event_exp_lt']*model.rsquared]

    def model_scan(self):
        """This function scans in parallel the space of time periods to identify the period where
        the event would most likely appear

        Returns:
            pd.DataFrame: Pandas dataframe with the results from all runs
        """
        results = Parallel(n_jobs=self.n_cores)(delayed(self.fit_model)(
            t, self.Lambda
            ) for t in range(self.min_periods+1, len(self.series)))

        cols = [self.time_label, 'coef', 'trend', 'pval', 'trend_pval', 'rsquared', 'weighted_coef']
        results = pd.DataFrame(results, columns=cols)
        for col in cols:
            if col!=self.time_label:
                results[col] = results[col].round(2)

        maxindex = results['weighted_coef'].idxmax()
        opt_res = results.iloc[[maxindex]]
        if opt_res['weighted_coef'].values[0]>0.01*self.series[self.metric_label].mean():
            self.detected=True
        else:
            self.detected=False
        results['detected_event'] = np.where((results.index==maxindex) & (self.detected), 1, 0)
        return results

    def fit(self, series, metric_label='metric', time_label='time'):
        """The main function used for model fitting and parameter estimation

        Args:
            series (_type_): the data used for fitting the model
            metric_label (str, optional): The label of the metric in the data. Defaults to 'metric'.
            time_label (str, optional): The label of the time dimension in the data. Defaults to 'time'.

        Returns:
            statsmodels.regression.linear_model.OLS: The optimal model
        """
        self.metric_label = metric_label
        self.time_label = time_label
        self.series=series

        # greedy-search Lambda
        weighted_coefs = {'lambda':[], 'coef':[]}
        for l in np.arange(0, 1, 0.05):
            self.Lambda=l
            self.results = self.model_scan()
            event_row = self.results[self.results.detected_event==1]
            weighted_coefs['coef'].append(event_row['weighted_coef'].values[0])
            weighted_coefs['lambda'].append(l)
        weighted_coefs=pd.DataFrame(weighted_coefs)
        id = weighted_coefs['coef'].idxmax()
        self.Lambda=weighted_coefs.loc[id, 'lambda']

        # scan event given optimal lambda
        self.results = self.model_scan()

        # fit optimal model
        event_t = self.results[self.results.detected_event==1][self.time_label].values[0]
        self.fit_model(event_t, self.Lambda, store_model=True, store_fitted=True)
        return self.model
    
    def summary(self):
        """Returns results from the fitting process

        Returns:
            dict: Python dictionary with the results
        """
        if self.detected:
            t=float(self.results[self.results.detected_event==1][self.time_label].values[0])
            hl=float(round(np.log(2)/self.Lambda,1) if self.Lambda!=0 else np.inf)
            a=float(self.results[self.results.detected_event==1]['coef'].values[0])
            trend=float(self.results[self.results.detected_event==1]['trend'].values[0])
            r2=float(self.results[self.results.detected_event==1]['rsquared'].values[0])

            return {
                'detected':self.detected,
                'event_time':t,
                'event_halflife':hl,
                'decay_lambda':float(self.Lambda),
                'amplitude':a,
                'trend':trend,
                'rsquared':r2
            }

        else:
            return {}

    def plot(self, save_plot=False):
        """Plots the times series along with the fitted model

        Args:
            save_plot (bool, optional): Whether you would like to save the model. Defaults to False.
        """
        plt.figure(figsize=(10,6))
        sns.lineplot(x=self.time_label, y=self.metric_label, data=self.series)
        sns.lineplot(x=self.time_label, y=f'fitted_{self.metric_label}', data=self.series)
        plt.tight_layout()
        if save_plot:
            plt.savefig('figures/plot.png', bbox_inches='tight')
        plt.show()