import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


display = pd.options.display
display.max_columns = 1000
display.max_rows = 1000
display.max_colwidth = 199
display.width = 1000

def preprocess(input: str) -> str:
    input = input.lower()
    input = input.translate(str.maketrans('','',string.punctuation))
    return input


corpus = [
    'Mr Jeremy put on a macintosh, and a pair of shiny shoes; he took his fishing rod and basket, and set off with enormous hops to the place where he kept his boat. The boat was round and green, and very like the other lily-leaves. It was tied to a water-plant in the middle of the pond.',
    'Peter never stopped running or looked behind him till he got home to the big fir-tree. He was so tired that he flopped down upon the nice soft sand on the floor of the rabbit-hole and shut his eyes. His mother was busy cooking; she wondered what he had done with his clothes. It was the second little jacket and pair of shoes that Peter had lost in a week!'
]

# First we pre-process the input
processed_corpus = [preprocess(x) for x in corpus]

# Now we generate the two TF-IDF vectors and display them
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_corpus)

print("----------------------------")
print("TF-IDF Vectors")
print("----------------------------")

dataframe = pd.DataFrame(data=np.transpose(X.toarray()), index=vectorizer.get_feature_names_out(), columns=['Document 1','Document 2'])

print(dataframe)

print("----------------------------")
print("Cosine of Similarity")
print("----------------------------")

similarity = cosine_similarity(X)
print(similarity)
