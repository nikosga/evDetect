# evDetect
*Parametric event detection & inference library*

---

## What is evDetect?

evDetect is a minimal python library for:
1. Automatic detection of events (e.g. spikes) in any metric
2. Parametric modeling of these events as an exponential decay process while accounting for trends
3. Reporting of detected event with metrics like amplitude and half-life.

The underlying assumption is that every event can be approximated by an exponential decay process:

$$y=Ae^{-\lambda t}$$

Because the library has to scan through every time period and tune the optimal parameter $\lambda$ we have included the functionality for multiprocessing to accelerate the computation time.

## Install

```sh
pip install evdetect
```

## How to use

**Code Example**

```python
from evdetect.evdetector import Detector
from evdetect.gen_data import Scenario

s = Scenario()
d=Detector()
d.fit(s.data)
print(d.summary())
d.plot()
```

**Summary Example**

```python
{
    'detected': True, 
    'event_time': 200.0, 
    'event_halflife': 1.7, 
    'decay_lambda': 0.4, 
    'amplitude': 6.32, 
    'trend': 0.01, 
    'rsquared': 0.95
}
```

**Charts Examples**

No trend and infinite half-life event
![Example1](figures/plot_1.png)

Trend with non infinite half-life event
![Example2](figures/plot_2.png)

For more examples see the tutorial in the notebooks folder.

## Author

[Nick Gavriil](https://www.nickgavriil.com/)