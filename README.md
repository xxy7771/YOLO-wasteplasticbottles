YOLO-wasteplasticbottles re-implementation using PyTorch

### Installation

```
pip install -r requirements.txt
```

### Train

* Configure your dataset path in `train.py` for training
* Run `yolo train model=$.pt data=$.yaml epochs=200 imgsz=640 batch=16` for training

### Val

* Configure your dataset path in `val.py` for testing
* Run `yolo val model = $.pt data = $.yaml` for valing

`$` represents the name of the yaml file

The model can autonomously choose between $.pt.

### 3D Location

* Run `python 3D Location.py  model = YOLO("$.pt")` for 3D Location

### Reference

* https://github.com/ultralytics/ultralytics
