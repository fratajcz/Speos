from gears import PertData, GEARS

# get data
print("initializing data")
pert_data = PertData('/mnt/storage/gears/')
print("Done")
# load dataset in paper: norman, adamson, dixit.
print("loading data")
pert_data.load(data_name = 'norman')
print("Done")

# specify data split
pert_data.prepare_split(split = 'simulation', seed = 1)
# get dataloader with batch size
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

gears_model = GEARS(pert_data, device = 'cuda:0')
#gears_model.model_initialize(hidden_size = 64)

#gears_model.train(epochs = 20)

# save/load model
#gears_model.save_model('gears')
gears_model.load_pretrained('gears')

import pandas as pd
hsps = pd.read_csv("/home/ubuntu/speos/hsps/uc.txt", header=None).iloc[:, 0].tolist()

import json

with open("/mnt/storage/speos/results/uc_film_nohetioouter_results.json", "r") as file:
    results = [key for key, value in json.load(file)[0].items() if value >= 11]

with open("/mnt/storage/speos/results/uc_film_nohetioouter_results.json", "r") as file:
    weakcore = [key for key, value in json.load(file)[0].items() if value < 11]

mendelians = pd.read_csv("/home/ubuntu/speos/extensions/uc_only_genes.tsv", sep="\t")["HGNC"].tolist()
results += mendelians


# first make the correct lists and shuffle them

coregenes = list(set(results[:]).intersection(set(gears_model.pert_list)))

allowed_hsps = list(set(hsps).intersection(set(gears_model.pert_list)))

background_peripherals = list(set(gears_model.pert_list).difference(set(weakcore)).difference(set(results)).difference(set(hsps)))

print(len(coregenes))
print(len(allowed_hsps))
print(len(background_peripherals))

from itertools import combinations
import os

def test_combination(genea, geneb, model, group):
    if os.path.isfile("/mnt/storage/gears/gi/{}_{}.tsv".format(genea, geneb)):
        return
    data = model.GI_predict([genea, geneb], GI_genes_file=None)
    del data["ts"]
    df = pd.DataFrame.from_dict({key: [value] for key, value in data.items()})
    df["Group"] = group
    df["Combination"] = [(genea, geneb)]
    df.to_csv("/mnt/storage/gears/gi/{}_{}.tsv".format(genea, geneb), sep="\t")


for (i, j) in combinations(allowed_hsps, 2):
    test_combination(i, j, gears_model, group="HSPxHSP")

print("Done with HSPxHSP")

from tqdm import tqdm
from random import seed, shuffle
seed(1)

to_use_combinations = list(combinations(coregenes, 2))
shuffle(to_use_combinations)

for (i, j) in tqdm(to_use_combinations):
    test_combination(i, j, gears_model, group="CorexCore")

print("Done with CorexCore")

to_use_combinations = []
for i in allowed_hsps:
    for j in coregenes:
        to_use_combinations.append((i, j))

seed(1)
shuffle(to_use_combinations)
for (i, j) in tqdm(to_use_combinations):
    test_combination(i, j, gears_model, group="HSPxCore")

print("Done with HSPxCore")

to_use_combinations = []
for i in allowed_hsps:
    for j in background_peripherals:
        to_use_combinations.append((i, j))

seed(1)
shuffle(to_use_combinations)
for (i, j) in tqdm(to_use_combinations):
        test_combination(i, j, gears_model, group="HSPxPeri")

print("Done with HSPxPeri")

to_use_combinations = list(combinations(background_peripherals, 2))
shuffle(to_use_combinations)
for (i, j) in tqdm(to_use_combinations):
    test_combination(i, j, gears_model, group="PerixPeri")

print("Done with PerixPeri")

for i in coregenes:
    for j in background_peripherals:
        to_use_combinations.append((i, j))

shuffle(to_use_combinations)

for (i, j) in tqdm(to_use_combinations):
    test_combination(i, j, gears_model, group="CorexPeri")

print("Done with CorexPeri")