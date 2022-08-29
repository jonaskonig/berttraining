from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from tqdm.auto import tqdm
import pandas as pd
import torch
from os.path import exists

FILENAME = "output/topi_label.csv"
datafile = ["output/trumptrain.csv","output/trumptest.csv","output/trumpval.csv"]
#dataset = load_dataset("csv",datafile=datafile)
dataset = load_dataset("jonaskoenig/trump_administration_statement", split="train[:2]")

print(dataset)
topiclist = ["work","news","sports","music","movies","politics","phones"," self-driving cars","family","cars","climate change","languages","business","health","science","style","opinion","economy","history","technology","affair","development","mobility"]

pipe = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
df= pd.DataFrame()
counter = 0
alltimecounter = 0
datdict = []
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, candidate_labels=topiclist, multi_label=True):
    myDict = {k: v for k, v in zip(out["labels"], out["scores"])}
    myDict["text"] =out["sequence"]
    myDict = {key: val for key, val in sorted(myDict.items(), key = lambda ele: ele[0])}
    datdict.append(myDict)
    counter += 1
    #print(counter)
    if counter > 15:
        if exists(FILENAME):
            pd.DataFrame(datdict).to_csv(FILENAME, mode='a', index=False, header=False)
        else:
            pd.DataFrame(datdict).to_csv(FILENAME, mode='w', index=False)
        datdict = []
        counter = 0
        alltimecounter += 16
        print(alltimecounter)

