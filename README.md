This program seeks to classify the language of a document deterministically. 

Components of experiment:
1. [Samples](data/languages) of 1000 of the most common words for a given language
2. [Documents](data/samplesDocs) of which the language may not be known. The samples are included in this data for better visualization

Procedure:
1. Process the samples: Read the all the samples of common words into a single dictionary 
2. Vectorize a document: Score the documents based on how frequent a tokenized/stemmed word from the samples dictionary appears in the content of the document. This output will be in vector form of n-dimensions, where n is the number of languages we want to look for in a document.
3. Principal Component Analysis (PCA): From the vectors of the individual documents/raw text, create a matrix A, find its covariance matrix B = cov(A), and plot the eigen vectors B.

NOTE: The [notebook](notebook.ipynb) and its outputs has been exporeted to an [HTML](notebook.html) for viewing