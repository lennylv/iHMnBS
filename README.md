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
