import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def geometric_adstock(series, decay, event_time, max_effect_time):
    adstock_series = np.zeros_like(series, dtype=float)
    for t in range(len(series)):
        adstock_series[t] = series[t] + (decay * adstock_series[t-1] if t >= event_time and t <max_effect_time else 0)    
    return adstock_series

class Scenario:
    
    def __init__(self, scenario_type='constant', n_records=500, std=0.2, effect_time=200):
        """Data generation class for the examples

        Args:
            scenario_type (str, optional): The type of pattern you would like to generate. Defaults to 'constant'.
            n_records (int, optional): Number of time periods. Defaults to 500.
            std (float, optional): Noise measured in standard deviation of normal distribution. Defaults to 0.2.
            effect_time (int, optional): The time the event should appear. Defaults to 200.
        """
        self.scenario_type=scenario_type
        self.n_records=n_records
        self.std=std
        self.effect_time=effect_time
        self.data=self.gen_data()
    
    def gen_data(self):
        """The method for data generation

        Raises:
            ValueError: If a scenarion that is not available is requested

        Returns:
            pd.DataFrame: The dataframe with the synthetic data
        """

        if self.scenario_type=='constant':
            y = np.random.normal(scale=self.std, size=self.n_records)
            t = np.arange(1, len(y)+1)

            y = np.where(t>=self.effect_time, y+5, y)
            s = pd.DataFrame({'time':t, 'metric':y})

        elif self.scenario_type=='constant_with_trend':
            y = np.random.normal(scale=self.std, size=self.n_records)
            t = np.arange(1, len(y)+1)
            b = 0.01
            y = y + b*t

            y = np.where(t>=self.effect_time, y+5, y)
            s = pd.DataFrame({'time':t, 'metric':y})

        elif self.scenario_type=='diminishing':
            y = np.random.normal(loc=0, scale=self.std, size=self.n_records)
            t = np.arange(1, len(y)+1)

            max_effect_time=self.effect_time+50
            decay = 0.9
            y = np.where(t==self.effect_time, y+5, y)
            y = geometric_adstock(y, decay, self.effect_time, max_effect_time)
            s = pd.DataFrame({'time':t, 'metric':y})

        elif self.scenario_type=='diminishing_with_trend':
            y = np.random.normal(loc=0, scale=self.std, size=self.n_records)
            t = np.arange(1, len(y)+1)

            max_effect_time=self.effect_time+50
            decay = 0.9
            y = np.where(t==self.effect_time, y+5, y)
            y = geometric_adstock(y, decay, self.effect_time, max_effect_time)
            b = 0.01
            y = y + b*t
            s = pd.DataFrame({'time':t, 'metric':y})
        
        else:
            raise ValueError('Not implemented scenario')
        
        return s
    
    def plot(self):
        """Plots the synthetic data
        """
        plt.figure(figsize=(10,6))
        sns.lineplot(x='time', y='metric', data=self.data)
        plt.show()