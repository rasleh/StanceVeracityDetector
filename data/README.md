Data
====
This folder contains datasets for training and testing machine learning models and data for use in generating features
from said data, for use in said machine learning models. 

## Datasets
Contains datasets for training and testing machine learning models. Currently contrains two datasets; DAST and 
'twitter' (working title, as dataset is still being constructed), which are each placed in their own sub-folder. Folder
structure for the /datasets/ folder can be found below.

```
datasets
|
|___twitter
|   |
|   |___preprocessed
|   |   |
|   |   |___stance
|   |   |
|   |   |___veracity
|   |
|   |___raw
|
|___DAST
    |
    |___preprocessed
    |   |
    |   |___stance
    |   |
    |   |___veracity
    |
    |___raw
```

The /preprocessed/ folders contain data preprocessed for use in stance detection and veracity determination. For 
information regarding data structures in the preprocessed data, see README.md in the root folder. /raw/ folders contain
the non-preprocessed data for each dataset. /DAST/ contains several sub-folders in the /raw/ folder. For an in-depth
explanation of folder and dataset structure, see README in the /datasets/dast/raw/ folder.

## Featurization
The featurization sub-folder contains data used for extracting features from data, for used in machine learning models.
The folder is structured as follows:
```
Featurization
|   bow_dict.txt
|   ddo-synonyms.csv
|
|___Lexicon
|       README.md
|       negation_words.txt
|       negation_smileys.txt
|       positive_smileys.txt
|       swear_words.txt
|       swear_words_en.txt
|
|___word2vec
        README.md
```

See  README in word2vec folder on how to attain pre-trained word2vec embeddings for the DAST dataset.