Stance and Veracity detector
=======
This project contains models and scripts allowing determination of stance within a given text towards some other text, 
as well as the determination of the veracity of a given piece of text based on conversational structures stemming from
comment sections, connected to that text. The project currently supports use of reddit data in the form of the DAST
dataset, as well as tweet conversation trees. Data formatting guidelines are found in the README in the /data/ folder.

## Getting started
### Word embeddings
Download word2vec word embeddings trained on the DAST dataset and sentence data from [dsl](https://dsl.dk/), from the 
link below, and place these in the folder /data/featurization/word2vec/. Required embedding files:
* word2vec_dsl_sentences_reddit_sentences_300_cbow_negative.kv
* word2vec_dsl_sentences_reddit_sentences_300_cbow_negative.kv.vectors.npy

Embeddings are downloaded [here](https://figshare.com/articles/Danish_DSL_and_Reddit_word2vec_word_embeddings/8099927).

### Preprocessing data
Most preprocessed data is already included in the project, however a preprocessed version of the full DAST dataset for 
stance detection was found too large to include. This is generated by running the preprocess_stance.py script found in /src/
with all but one parameter set to default, as such:

>`-m preprocess_stance out_file_name=pure_lstm_subset.csv`

This data file is used as default for training and testing the stance detection model, and this functionality will thus
not function before the file has been generated.

### Python libraries
The project was developed using Python 3.6, and requires the following Python libraries to be available:
* matplotlib
* NumPy
* [AFINN](https://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010)
* NLTK
* sklearn
* Polyglot (see polyglot_pos.py for installation instructions)
* Gensim
* joblib
* hmmlearn
* Torch

### Running code
The project has a number of command-line interfaces, using which new data can be preprocessed for use in stance detection and
veracity prediction models, and new models can be trained, tested and saved as joblib files for future use.

#### preprocess_stance.py
Script for preprocessing data for use in stance detection. Default parameters are currently set to generate preprocessed
data for the lstm_stance model. Script is located in /src/.

The command-line interface calls the preprocess() method of preprocess_stance.py using user-defined input, or default values.
Default values are defined for all variables.

* <b>--database</b> Defines the database type, currently accepting 'twitter' or 'dast' as input, using 'dast' as default
* <b>--data_path</b> Defines the full path to the raw data to be preprocessed. If none is entered, a data path is generated based on the input to <b>--database</b>
* <b>--sub_sample</b> Implements sub-sampling by removing conversation structures containing only comments of the class "commenting", useful for the DAST
dataset, where this is the vast majority class. Set to True as default
* <b>--sdqc_parent</b> Include the stance label of the parent comment as feature, set to False as default
* <b>--text_features</b> Include a number of textual features, including but not limited to presence of punctuation and 
question-words, number of normalized features e.g. text length and ratio of capital letters to non-capital. Set to False as default.
* <b>--sentiment</b> Include sentiment of text as feature, set to False as default
* <b>--lexicon</b> Include lexical features e.g. use of smileys and swear words, set to False as default
* <b>--pos</b> Include part of speech tags as features, set to False as default
* <b>--word_ebs</b> Include features based on word embeddings, including e.g. cosine similarity to parent and to conversation,
set to False as default
* <b>--lstm_wembs</b> Include word embeddings formatted for the conditional LSTM, in the form of word embeddings for the 
text followed by word embeddings for the source text towards which stance is to be determined, set to True as default
* <b>--write_out</b> Whether the generated preprocessed data should be written to file, set to True as default
* <b>--out_file_name</b> Name of the out file, set to "preprocessed.csv" as default

Generating preprocessed DAST data for stance detection, without sub-sampling, including only sentiment and lexicon features, while not saving 
to file, is thus performed:
>`-m preprocess_stance sub_sample=False sentiment=True lexicon=True lstm_wembs=False write_out=False`

Generating preprocessed DAST data with sub-sampling and writing to file, for use in the conditional LSTM, is performed using:
>`-m preprocess_stance`

##### Input data format
Raw DAST data follows the dataset guidelines found in the README file in the folder /data/datasets/dast/raw/. Raw DAST 
data should be found in this folder.
 
Raw twitter data should be passed in a .txt file, as generated by the twitter conversation collecter found [here](https://github.com/rasleh/twitter-conversation-collecter).
The file should contain one twitter conversation tree at each line, the line starting with the tweet ID of the source
tweet, followed by a dictionary of tweet IDs referencing tweet .json objects, as generated when querying the twitter api.
These json objects have the following added variables:
* <b>SDQC_Submission</b> The SDQC value of the tweet towards a given rumour, "Underspecified" if yet unlabeled
* <b>SourceSDQC</b> The SDQC value of a source tweet towards a given rumour, "Underspecified" if yet unlabeled
* <b>SDQC_Parent</b> The SDQC value of the parent tweet in the conversational structure, "Underspecified" if yet unlabeled
* <b>Children</b> An array containing IDs of tweets commenting on this tweet, that is, children in the conversation tree

##### Output data format
A tab separated .csv file with a header containing "ID", "SDQC_Submission" and the names of all included features.
Each line contains a single tweet, including its ID, SDQC value (if any, else a blank space) and each of the included features,
in the corresponding row.

By default preprocessed data files are found in /data/datasets/dast/preprocessed/stance/ for DAST and 
/data/datasets/twitter/preprocessed/stance/ for Twitter data.

#### preprocess_veracity.py
Script for preprocessing data for use in veracity determination.

The command-line interface calls the preprocess() method of preprocess_veracity.py using user-defined input, or default values.
Default values are defined for all variables.

* <b>--database</b> Defines the database type, currently accepting 'twitter' or 'dast' as input, using 'dast' as default
* <b>--data_path</b> Defines the full path to the raw data to be preprocessed. If none is entered, a data path is generated based on the input to <b>--database</b>
* <b>--timestamps</b> Include timestamps normalized across current branch in the conversational tree as features, set to True as default
* <b>--write_out</b> Whether the generated preprocessed data should be written to file, set to True as default
* <b>--out_file_name</b> Name of the out file, set to "preprocessed.csv" as default

Generating preprocessed DAST data for veracity determination, without timestamps, while not saving, would be performed thus:
>`preprocess_veracity timestamps=False write_out=False`

##### Input data format
Raw DAST data follows the dataset guidelines found in the README file in the folder /data/datasets/twitter/raw/. Raw DAST 
data should be found in this folder.
 
Raw twitter data should be passed in a .txt file, as generated by the twitter conversation collecter found [here](https://github.com/rasleh/twitter-conversation-collecter).
See subsection <i>input data format</i> in the section <i>preprocess_stance.py</i> for further details.

To allow proper pre-processing, SDQC_Submission must be filled for raw twitter data, and a variable "TruthStatus" must 
be defined for all source tweets, and added to the dataset. No data is currently annotated for this dataset. 

##### Output data format
A tab separated .csv file, with a header containing "TruthStatus", and "SDQC_Labels". 
Each line contains a single branch in a conversational tree in the form of the veracity of the rumour the branch corresponds
to, followed by an array of SDQC labels. Each SDQC label is contained in an array, and if timestamps are included by features,
it is followed by the normalized timestamp

By default preprocessed data files are found in /data/datasets/dast/preprocessed/veracity/ for DAST and 
/data/datasets/twitter/preprocessed/veracity/ for Twitter data.

#### hmm_veracity.py
Script for initializing, training and saving an HMM-based model for veracity prediction.

The command-line interface calls the run_benchmark method of the hmm_veracity.py script, using the user-defined inputs or
default values, and saves the generated model. 

* <b>--data_path</b> Defines the full path to the data to be used for model training. If none is entered, a data path is generated based on the input to <b>--timestamps</b>
* <b>--timestamps</b> Include timestamps normalized across current branch in the conversational tree as features, set to True as default
* <b>--unverified_cast</b> How unverified rumours should be cast. Either 'true', 'false' or 'none', where 'none' leaves them as "unverified". 'false' used as default
* <b>--save_model</b> Whether the generated model should be saved as a joblib file, set to True as default
* <b>--model_name</b> The name of the generated joblib file containing the model, will be generated if not defined by user
* <b>--remove_commenting</b> Whether comments with "commenting" SDQC value should be ignored in predicting veracity. Has been
shown to increase performance in some cases, is false as default
* <b>--hmm_version</b> Which HMM implementation is to be used, Gaussian HMM by default, "multinomial" can be entered as alternative

Generating and saving a gaussian HMM veracity determination model without using timestamps as features is done thus:
>`-m hmm_veracity timestamps=False`

##### Input data format
Makes use of data preprocessed for veracity prediction, as performed by preprocess_veracity.py. See subsection <i>output 
data format</i> in the section <i>preprocess_veracity.py</i> for further details.

#### lstm_stance.py
Script for initializing, training and saving a conditional LSTM-based model for stance detection.

The command-line interface calls the load_dast_lstm function from the data_loader.py script located in /src/, and subsequently
calls the run_specific_benchmark function of the lstm_stance.py script, saving the generated model. Will save the strongest
model found, after running for the user-defined number of epochs.

A benchmark file containing model performance during training can be generated in /benchmarking/ if specified by user.

* <b>--data_path</b> Defines the full path to the data to be used for model training. Set to DAST path as default
* <b>--lstm_layers</b> Number of LSTM layers to be used in model, set to 3 as default
* <b>--lstm_dimensions</b> Number of dimensions in each LSTM layer, set to 200 as default
* <b>--relu_layers</b> Number of linear layers with ReLU activation function, set to 1 as default
* <b>--relu_dimensions</b> Number of dimensions in each linear layer with ReLU activation, set to 50 as default
* <b>--max_epochs</b> Maximal number of epochs for training, set to 200 as default
* <b>--bi-directional</b> Whether the model is to be bi-directional. Set to False as default, as this implementation is
still faulty for some cases
* <b>--save_model</b> Whether the model is to be saved as a joblib file, set to True as default
* <b>--model_name</b> Name for the saved joblib file, is generated based on hyperparameters and model performance if not specified
* <b>--benchmark_name</b> Name of the benchmark file, is generated based on hyperparameters and model performance if not specified

Default values of lstm_layers, lstm_dimensions, relu_layers and relu_dimensions set based on optimal hyperparameter combination.

Generating and saving a stance detection model with 1 LSTM layer, 500 LSTM dimensions and run for a max of 500 epochs:
>`-m lstm_stance lstm_layers=1 lstm_dimensions=500 max_epochs=500`

##### Input data format
Takes data preprocessed for stance detection, as defined in subsection <i>output data format</i> in the section <i>preprocess_stance.py</i>.

#### veracity.py
Script for running unseen branches of a conversation structure through a pre-trained stance detection model, determine stances
within the branch, and running those stances through a pre-trained veracity prediction model, to determine the veracity
of the branch source.

The command-line interface loads two pre-trained models and a given dataset, splits that dataset into branches, determines
stance followed by veracity for each branch and outputs results as print statements. The interface allows use of data already
in a database, or an ID reference to a tweet, for which veracity is to be determined. Give "new" as parameter to access
the "new data" interface, and "stored" to access the "stored data" interface.

Primary interface:
* <b>--stance_model_path</b> Path to pre-trained stance detection model, model using optimal hyperparameters used as default
* <b>--veracity_model_path</b> Path to pre-trained veracity determination model, strongest current HMM model is used as default,
either model trained with or without timestamps chosen based on user input for <b>--timestamps</b>
* <b>--timestamps</b> Whether timestamps are to be used as features. If True, a veracity model trained using timestamps as
features must be used as well. Set to True as default

"New data" interface variables:
* <b>--id</b> The ID of a tweet from the conversation, for which veracity will be determined

"Stored data" interface variables:
* <b>--data_path</b> Defines the full path to the raw data, generated based on <b>--data_type</b> if none is given
* <b>--data_type</b> Type of raw data, either 'twitter' or 'dast', 'twitter' is used as default

##### Input data format
Uses the same input data format as preprocess_stance.py. See subsection <i>input data format</i> in the section 
<i>preprocess_stance.py</i>.

#### Scripts without command-line interfaces
##### lstm_benchmark_visualization.py
/benchmarking/ contains benchmark files, as well as <i>lstm_benchmark_visualization.py</i>. This a script for visualizing
model performance for lstm_stance.py. The script is currently inflexible, and assumes a hyperparameter space of 27 combinations;
3 different values for number of LSTM layers, 3 for number of LSTM dimensions and 3 for number of dimensions in linear 
layers with a ReLU activation function. 

##### feature_extraction folder
Located at /src/feature_extraction/, the folder contains scripts with helper methods for feature extraction, used for 
preprocessing data for both the stance prediction and veracity determination models.  

##### data_loader.py and pheme_loader.py
Located at /src/, the scripts contains methods for loading raw and preprocessed data used for training and testing stance
detection and veracity determination models, as well as for loading new data directly into the veracity.py script. 

##### tweet_fetcher.py
Located at /src/, the script contains methods for accessing the Twitter API, scraping tweets and saving them to file in a
file format compatible with the methods in the <i>data_loader.py</i> script, as well as for merging data files of this format.
The script allows two different search types:
* popular_search - Searches for tweets made from Denmark in Danish, with at least 10 replies, navigates to the source of the
conversation tree in which each of these tweets are located, and scrapes said conversation tree
* specific_search - Searches for tweets given a specific query, and scrapes all results of this search, along with responses
to these tweets

##### live_veracity.py
Located at /src/, and makes use of <i>veracity.py</i> and <i>tweet_fetcher.py</i> to scrape popular tweets made
from Denmark in Danish, featurize these, and perform stance detection and subsequent veracity prediction for the scraped
tweets.

##### Dataset classes
Located at /src/dataset_classes/ is <i>DAST_datasets.py</i> and <i>datasets.py</i>. The DataSet class found in datasets.py
is used for representing collections of text data points, and performing operations on these collections. The Annotation 
class is likewise found in datasets.py, and is used for representing said text data points. The DastDataset and DastAnnotation
classes found in DAST_datasets.py inherit from the corresponding classes in datasets.py, with alterations allowing for
the use of data from the DAST dataset. 

##### labeling_doccano folder, annotation_gui.py and automation.sh
All located at /src/, and contain current efforts into automation of data scraping and annotation, as well as construction of
a GUI for annotation of scraped data for stance detection and veracity prediction. 

## Extending the project
The project is set up to be extendable in two dimensions; adding new models, both for stance detection and veracity prediction,
and adding new datasets to be included.

### Adding new models
New model classes should be added to the /src/models/ folder, and named using the naming convention [model_type]_[task].py, 
where task is the task which the model solves; 'stance' or 'veracity'. New model classes should allow generating a model, 
training it and saving it to a joblib file in the /pretrained_models/ folder. New models should take data in the format 
generated by preprocess_stance.py and preprocess_veracity.py; see subsection <i>output data format</i> in the sections 
<i>preprocess_stance.py</i> and <i>preprocess_veracity.py</i>.

Any additional feature extraction required for the new model should be implemented in a script in /src/feature_extraction/.
If possible, make use of methods already in feature_extractor.py and extend this script if necessary.

After implementation of a new model, veracity.py should be extended to allow model choice based on user input in the 
command-line interface.

### Adding new data
To add new data source to the project, a folder should be created in /data/datasets/. This folder should hold two folders;
/preprocessed/ and /raw/. /preprocessed/ should contain /stance/ and /veracity/, and /raw/ should contain the new data.

In data_loader.py, located in /src/, extend the switch in the load_raw_data method, to contain the new dataset. Create a new method named
load_raw_[datasourceName], which is called within the load_raw_data method. The data structure returned by load_raw_[datasource_name]
should correspond to that yielded by the other load_raw_[X] methods.

The yielded data structure should follow the one shown below. All text information is stored in dictionary form. For each 
source text, an array is created following the structure below. These arrays are added to a final array, which should be
returned from load_raw_[X].


```
[
    [
        {source text A},
        [ 
            {Reply text a}, 
            {Reply text b},      Conversation branch 1 - array of all texts, in dictionary form, in branch
            ...
            {Reply text x}
        ], 
        [ 
            {Reply text c}, 
            {Reply text d},      Conversation branch 2 - array of all texts, in dictionary form, in branch
            ... 
            {Reply text y}
        ], 
        ...
        [ 
            {Reply text e}, 
            {Reply text f},      Final Conversation branch for source - array of all texts in branch
            ... 
            {Reply text z}
        ] 
    ]
    [
        {Source text B},
        [Conversation branch 3],
        ...
        [Conversation branch Z]
    ]
    ...
    [  
        {Source text X},
        [Conversation branch 4}
        ...
        [Conversation branch x]
    ]
]
```

Create a datasource-specific dataset class in /srs/dataset_classes/, containing two inner classes; one representing
the texts in the new datasource, extending Annotaiton in datasets.py and one extending DataSet in datasets.py.
The annotation should hold logic to extract information regarding the texts in the new dataset, including but not limited
to which children a given text has in the overall conversational structure, the SDQC value of the text, the creation
date of the text and the word tokens in the text. The dataset class should hold logic to handle any discrepancies in
behaviour between the Annotation class in datasets.py and the newly created annotation class. See datasets.py and DAST_datasets.py,
both located in /src/dataset_classes/ for reference.

In preprocess_veracity.py and preprocess_stance.py, both located in /src/, extend the get_database_variables methods to
allow choosing the new dataset. This includes giving path specifications to raw and preprocessed data, an overview of
which rumours are included in the data source and the veracity of these rumours, and finally where to find the datasource
specific dataset class.

## Authors
* Rasmus Lehmann - <i>Primary contributor</i>
* Leon Derczynski - <i>Project supervisor</i>
* Anders Edelbo Lilie & Emil Refsgaard Middelboe - <i>Initial work</i> - [RumourResolution](href=https://github.com/danish-stance-detectors/RumourResolution)

## License
This project is licensed under the MIT License - see the [MIT License web-page](https://choosealicense.com/licenses/mit/) for details.

## Acknowledgements
* [DSL](https://dsl.dk/) The word embeddings have been trained on both sentence data from [dsl](https://dsl.dk/) and on 
reddit data from the danish stance dataset.
* [Afinn](https://github.com/fnielsen/afinn) The afinn sentiment is facilitated by the afinn sentiment library, which 
has been linked above. Further credits can be seen below. Finn Årup Nielsen, "A new ANEW: evaluation of a word list for 
sentiment analysis in microblogs", Proceedings of the ESWC2011 Workshop on 'Making Sense of Microposts': Big things come 
in small packages. Volume 718 in CEUR Workshop Proceedings: 93-98. 2011 May. Matthew Rowe, Milan Stankovic, Aba-Sah Dadzie, 
Mariann Hardey (editors)
* [Polyglot](https://polyglot.readthedocs.io/en/latest/POS.html) for POS tagging. See: "Al-Rfou, Rami  and  Perozzi, 
Bryan  and  Skiena, Steven, (2013), [Polyglot: Distributed Word Representations for Multilingual NLP](http://www.aclweb.org/anthology/W13-3520)"
* [Kochkina et al.](https://github.com/kochkinaelena/branchLSTM) for inspiration on LSTM implementation

 