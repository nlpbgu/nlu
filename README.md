## UNLI Commonsense Reasoning Augmentation ACL 2024

This repository contains the code for the paper titled "UNLI Commonsense Reasoning Augmentation," presented at ACL 2024. The paper introduces an enhancement to the UNLI model by integrating commonsense reasoning using the COMET-ATOMIC 2020 knowledge graph. Our approach identifies instances with high prediction residuals during training and augments them with commonsense reasoning to improve the model's performance. This method demonstrates that incorporating commonsense knowledge significantly enhances the model's ability to capture nuanced inferences in natural language tasks.

### Prerequisites
 * Python >= 3.6

### Running

In order to run the code, first clone this repository to your location.

Arguments explanation:
```
| root_dir - your location to  the root directory of the project repository                   |
| out_dir - dir to store the learning weights after the training done                         |
| augmentation - model to augmentation the unli data                                          | 
| threshold - the threshold of the residual you decide to robust the unli                     | 
| training_augmentation - after augmentation ,training the model with new augmentation data   | 
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
python py_tapes/regression.py --root_dir --out_dir --augmentation comet --threshold 
```

To augmentation UNLI with baseline bart  , run:


```python 
python py_tapes/regression.py --root_dir --out_dir --augmentation bart --threshold
```


To trains the regression model with the augmentation data , run:


```python 
python py_tapes/regression.py --root_dir --out_dir --training_augmentation
```



