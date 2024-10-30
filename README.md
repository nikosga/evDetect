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

For more examples see the tutorial in the notebooks folder.