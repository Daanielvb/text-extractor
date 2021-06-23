# Requirements
- Python 3.x and pip
- Gemsim, Numpy, NLTK, NLTK Trainer, Spacy, Sklearn, Pandas, Pyphen, Pyspellchecker 

# Configuration
- _It's highly recommended creating a virtualenv before installing the dependencies_

- Dependencies
```console
pip3 install virtualenv
virtualenv <YOU_NAME_IT>
source <THE_NAME_ABOVE>/bin/activate
pip install -r requirements.txt
sh setup.sh
```

- NLTK setup (Within a python terminal)
```python
import nltk
nltk.download('punkt')
nltk.download('mac_morpho')
nltk.download('stopwords')
```
_The step above should install the dependencies in your nltk_data folder (~/nltk_data)_


#Usage
- TBD

# ML text-extractor

- Extract textual document content from different sources (PDF, Docs and text files)
- Convert textual document into [stylometric features](http://www2.tcs.ifi.lmu.de/~ramyaa/publications/stylometry.pdf)
- Contains Random Forest and Simple Neural Network classifiers over the data described in the next section

## The data

- There are two main types of data set inside the data/parsed-data folder:
-- Regular data files, with textual content and masked author name
-- Stylometric data files, that represent the conversion of the raw text into stylometric features (~50)

*PS: Each data set has two versions of it, 'selected' means that samples with less than 3 per author were removed, 'data' is the complete data set with no exclusions*