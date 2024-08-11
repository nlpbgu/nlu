## UNLI Commonsense Reasoning Augmentation ACL 2024

This repository hosts the code for the following paper:
 * Tongfei Chen*, Zhengping Jiang*, Adam Poliak, Keisuke Sakaguchi, Benjamin Van Durme (2020): 
   Uncertain natural language inference. In _Proceedings of ACL_.

### Prerequisites
 * Python >= 3.6

### Running

In order to run the code, first clone this repository to your location.





To prepares SNLI and u-SNLI datasets (automatically downloads data) , run:


```python 
python py_tapes/data.py --root_dir your location to repository root dir
```

To trains baseline UNLI without augmentation , run:


```python 
python py_tapes/regression.py --root_dir --out_dir dir_store_the_learning_weights
```

To augmentation UNLI with Comet model , run:


```python 
python py_tapes/regression.py --root_dir --out_dir dir_store_the_learning_weights --augmentation comet --threshold 
```

To augmentation UNLI with baseline bart  , run:


```python 
python py_tapes/regression.py --root_dir --out_dir dir_store_the_learning_weights --augmentation bart --threshold
```


To trains the regression model with the augmentation data , run:


```python 
python py_tapes/regression.py --root_dir --out_dir dir_store_the_learning_weights --training_augmentation
```



