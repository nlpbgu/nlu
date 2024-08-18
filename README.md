## UNLI Commonsense Reasoning Augmentation 2024


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
| nli  -   Optional. values {CON,NEU,ENT}. nessecary to use with nli1 and nli2 , when you want to augmentation NEU  | 
| nli1 -  Optional. values [0,1]. UNLI range. Considers all values ​​smaller than it. use to augmentation CON        |
|          use only nli1 , without nli and nli2                                                                    |
| nli2 -  Optional. values [0,1]. UNLI range. Considers all values ​​bigger than it. use to augmentation ENT         |
|          use only nli2 , without nli and nli1                                                                    |
| training_augmentation - after augmentation ,training the model with new augmentation data                        |
| dir_augmentation - the name of the directory where the new augmentation data stored                              | 

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


### Retraining Data Augmentation

To trains the regression model with the augmentation data , run:


```python 
python py_tapes/regression.py --root_dir --out_dir --training_augmentation  --dir_augmentation
```



