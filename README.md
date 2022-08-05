# iHMnBS

The source code of paper 'Identifying modifications on DNA-bound histones with joint deep learning of multiple binding sites in DNA sequence'

## Description of the data

npz files can be loaded as:

```
import numpy as np
np.load(file, allow_pickle=True)
```

Each data entry includes five parts, the key and the meaning of the corresponding value are as follows:

- dna: dna sequence consisting of four bases (A\T\C\G)
- dnase: dnase values corresponding to the bases in the dna sequence
- tlabel: labelling of data under the iHMnBS setting
- label: labelling of data under the DeepHistone setting
- peaks: the location of peaks in ChIP-seq data, and the encoding for histone modification that occurs in this region

## Description of the code running

You can get the model up and running simply by running the `main.py` file with the following command:

```
python main.py
```

For the data input, you just need to fill the first argument of the `get_fn` function in the `main` function with your data path in the `main.py` file. 

The subsequent code in `main` function deals with the slicing of the data when we train the model. As the name implies, `train_valid_set` and `test_set` represent the training set and test set respectively, and later `train_set` and `valid_set` subdivide the training set into a set for training and a set for validation. This part of the code used for data segmentation can be commented out as appropriate when you test the code.

Any more questions please let me know: 20204227065@stu.suda.edu.cn
