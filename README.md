# evDetect
Parametric event detection & inference library

## Install

```
pip install evdetect
```

## How to use

**Example**

```python
from evdetect.evdetector import Detector
from evdetect.gen_data import Scenario

s = Scenario()
d=Detector()
d.fit(s.data)
print(d.summary())
d.predict()
d.plot()
```

**Examples**
No trend and infinite half-life event
![Example1](figures/plot_1.png)

Trend with non infinite half-life event
![Example2](figures/plot_2.png)

For parallel processing use `fit(...,parallel=True)`

For more examples see the tutorial in the notebooks folder.