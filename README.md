## UNLI Commonsense Reasoning Augmentation ACL 2024

This repository hosts the code for the following paper:
 * Tongfei Chen*, Zhengping Jiang*, Adam Poliak, Keisuke Sakaguchi, Benjamin Van Durme (2020): 
   Uncertain natural language inference. In _Proceedings of ACL_.

### Prerequisites
 * Python >= 3.6

### Running

In order to run the code, first clone this repository to your location.

Arguments explanation:
```
| root_dir - your location to  the root directory of the project repository                                        |
| out_dir - dir to store the learning weights after the training done                                              |
| augmentation - model to augmentation the unli data                                                               | 
| threshold - the threshold of the residual you decide to robust the unli                                          |
| nli - Optional. values {CON,NEU,ENT}. nessecary to use with nli1 and nli2 , when you want to augmentation NEU    | 
| nli1 - Optional. values [0,1]. when you want to augmentation CON                                                 |
| nli2 -  Optional. values [0,1]. when you want to augmentation ENT                                                | 
| training_augmentation - after augmentation ,training the model with new augmentation data                        |
| dir_augmentation - the name of the directory where the new dataset augmented stored                              | 

```


### Data

To prepares SNLI and u-SNLI datasets (automatically downloads data) , run:



```python 
python py_tapes/data.py --root_dir
```

### Training
To trains baseline UNLI without augmentation , run:


```python 
python py_tapes/regression.py --root_dir --out_dir 
```

To augmentation UNLI with Comet model , run:


```python 
python py_tapes/regression.py --root_dir --out_dir --augmentation comet --threshold --nli1 --nli2 --nli
```

To augmentation UNLI with baseline bart  , run:


```python 
python py_tapes/regression.py --root_dir --out_dir --augmentation bart --threshold --nli1 --nli2 --nli
```


To trains the regression model with the augmentation data , run:


```python 
python py_tapes/regression.py --root_dir --out_dir --training_augmentation  --dir_augmentation
```



