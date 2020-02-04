# ML text-extractor

- Extract textual document content from different sources (PDF, Docs and text files)
- Convert textual document into [stylometric features](http://www2.tcs.ifi.lmu.de/~ramyaa/publications/stylometry.pdf)
- Contains Random Forest and Simple Neural Network classifiers over the data described in the next section

## The data

- There are two main types of data set inside the data/parsed-data folder:
-- Regular data files, with textual content and masked author name
-- Stylometric data files, that represent the conversion of the raw text into stylometric features (~50)

*PS: Each data set has two versions of it, 'selected' means that samples with less than 3 per author were removed, 'data' is the complete data set with no exclusions*