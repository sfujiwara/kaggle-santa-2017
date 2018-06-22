import numpy as np
import pandas as pd
from tqdm import tqdm


print("loading data...")
child_wishlist = pd.read_csv("data/child_wishlist_v2.csv", header=None).drop(0, axis=1).values
gift_goodkids = pd.read_csv("data/gift_goodkids_v2.csv", header=None).drop(0, axis=1).values

print("computing child happiness matrix...")
child_happiness_mat = np.full((1000000, 1000), -10, dtype=np.int16)
val = np.arange(100, 0, -1) * 20 + 10
for cid in tqdm(range(1000000)):
    child_happiness_mat[cid, child_wishlist[cid]] += val

print("computing gift happiness matrix...")
gift_happiness_mat = np.full((1000000, 1000), -1, dtype=np.int16)
val = np.arange(1000, 0, -1) * 2 + 1
for gid in tqdm(range(1000)):
    gift_happiness_mat[gift_goodkids[gid], gid] += val

print("loading submit file")
# submit = np.loadtxt("results/relaxation_v2_1_0.csv", delimiter=",", dtype=int, skiprows=1)
# submit = np.loadtxt("results/ultimate_final3.csv", delimiter=",", dtype=int, skiprows=1)[:, :2]
submit = np.loadtxt("submission.csv", delimiter=",", dtype=int, skiprows=1)[:, :2]
child_happiness = 0
gift_happiness = 0
child_happiness_arr = np.zeros(1000000, dtype=np.int)
gift_happiness_arr = np.zeros(1000000, dtype=np.int)
for i, row in tqdm(enumerate(submit)):
    cid, gid = row
    child_happiness += child_happiness_mat[cid, gid]
    gift_happiness += gift_happiness_mat[cid, gid]
    child_happiness_arr[i] = child_happiness_mat[cid, gid]
    gift_happiness_arr[i] = gift_happiness_mat[cid, gid]

child_happiness /= 2000000000.
gift_happiness /= 2000000000.

print("Child Happiness: {}".format(child_happiness))
print("Gift Happiness: {}".format(gift_happiness))

print("Score: {}".format(gift_happiness**3 + child_happiness**3))
