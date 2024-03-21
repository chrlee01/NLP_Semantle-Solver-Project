from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import gensim.downloader
import time
import json
import datetime
import pytz

# load model
time_first = time.perf_counter()
print("loading word2vec model")
model = gensim.downloader.load('word2vec-google-news-300')
print(type(model))
print("model loaded")
print("took {} seconds to load".format(time.perf_counter() - time_first))

# get the number of the day
start_date = datetime.date(2022, 1, 29)
today = datetime.datetime.now(pytz.utc).date()
semantle_number = (today - start_date).days
print("semantle day", semantle_number)

# get target for the day
with open('words.json', 'r') as file:
    words = json.load(file)
    words = words['words']
    target = words[semantle_number]
target_vector = model[target]
target_vector_comparison = target_vector.reshape(1, -1)

url = f"https://semantle.com/model2/{target}/"

guess = ""

# game itself
while guess != target:
    # guess logic
    guess = input("enter a guess: ")    
    calculated_vector = model[guess]
    calculated_vector_comparison = calculated_vector.reshape(1, -1)

    # query website to see if returned vector and calculated vector are the same
    with requests.get(url + guess) as response:
        vector = np.array(response.json()['vec'])
        if cosine_similarity(vector.reshape(1, -1), calculated_vector_comparison) == 1:
            print("correct model")
        else:
            print("wrong model")

    # calculate cosine similarity to compare to the website
    similarity = cosine_similarity(target_vector_comparison, calculated_vector_comparison)
    similarity = similarity[0][0]
    print("cosine similarity:", similarity*100)

print("won, the word was {}".format(target))