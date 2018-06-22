# Run about 1 hour

from __future__ import print_function
from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
from tqdm import tqdm
import collections

gl = [
    0, 15, 19, 21, 23, 37, 38, 40, 47, 52, 58, 66, 69, 71, 72, 75, 82, 85, 89,
    101, 108, 114, 116, 124, 142, 148, 161, 162, 174, 182, 189, 192, 194,
    203, 204, 207, 210, 218, 223, 227, 233, 246, 248, 276, 279, 288, 291, 293, 296, 299,
    305, 307, 308, 315, 316, 317, 318, 326, 330, 340, 341, 343, 351, 353, 368, 370, 386, 387,
    404, 422, 423, 445, 454, 461, 462, 469, 476, 478, 483, 491, 492, 497, 499,
    504, 511, 530, 541, 553, 568, 570, 579, 588, 589, 596,
    600, 620, 621, 625, 633, 635, 637, 642, 647, 654, 665, 668, 689, 693, 698,
    700, 707, 712, 717, 722, 727, 730, 736, 747, 748, 752, 761, 778, 779, 793, 798,
    804, 808, 810, 823, 827, 831, 846, 851, 854, 858, 863, 868, 880, 889,
    900, 908, 914, 917, 919, 928, 936, 949, 959, 966, 973, 976, 978, 988
]

# np.random.seed(5)  # 0.9362849948224653
# np.random.seed(6)  # 0.9362815350584652
np.random.seed(8)

print("loading data...")
child_wishlist = pd.read_csv("data/child_wishlist_v2.csv", header=None).drop(0, axis=1).values
gift_goodkids = pd.read_csv("data/gift_goodkids_v2.csv", header=None).drop(0, axis=1).values

print("computing child happiness matrix...")
child_happiness_mat = np.full((1000000, 1000), -10, dtype=np.int16)
val = np.arange(100, 0, -1) * 20 + 10
for cid in tqdm(range(1000000)):
    child_happiness_mat[cid, child_wishlist[cid]] += val

# df = pd.read_csv("results/mcf_v2_3.csv")
df = pd.read_csv("min_cost_flow.csv")
gift_ids = np.full(1000000, -1, dtype=int)
n_unhappy_triplets = 0
n_unhappy_twins = 0
n_unhappy_singles = 0
for i, row in tqdm(df.iterrows()):
    cid, gid, flow, happiness_old = row["ChildId"], row["GiftId"], row["Flow"], row["Happiness"]
    # Triplets
    if cid <= 5000:
        if flow == 3:
            happiness = np.sum(child_happiness_mat[cid:(cid+3), gid])
            if happiness <= 2000:
                n_unhappy_triplets += 1
            else:
                gift_ids[cid] = gift_ids[cid + 1] = gift_ids[cid + 2] = gid
    elif cid <= 45000:
        if flow == 2:
            happiness = np.sum(child_happiness_mat[cid:(cid+2), gid])
            if happiness <= 1600:
                n_unhappy_twins += 1
            else:
                gift_ids[cid] = gift_ids[cid + 1] = gid
    else:
        happiness = child_happiness_mat[cid, gid]
        if happiness <= 800:
            n_unhappy_singles += 1
        elif gid in gl:
            # if np.random.rand() <= 0.0013:
            if np.random.rand() <= 0.:
                n_unhappy_singles += 1
            else:
                gift_ids[cid] = gid
        else:
            gift_ids[cid] = gid

print("Unhappy Triplets: {}".format(n_unhappy_triplets))
print("Unhappy Twins: {}".format(n_unhappy_twins))
print("Unhappy Singles: {}".format(n_unhappy_singles))

gift_counter = collections.Counter(gift_ids)
leftover_gift_ids = []
n_gifts = []
for gid in range(1000):
    if gift_counter[gid] < 1000:
        leftover_gift_ids.append(gid)
        n_gifts.append(1000 - gift_counter[gid])

# import IPython; IPython.embed()

is_triplet = []
is_twin = []
child_ids = []
for cid in tqdm(range(1000000)):
    if gift_ids[cid] != -1:
        continue
    # Triplet
    if cid <= 5000 and cid % 3 == 0:
        child_ids.append(cid)
        is_triplet.append(True)
        is_twin.append(False)
    elif 5001 <= cid <= 45000 and cid % 2 == 1:
        child_ids.append(cid)
        is_triplet.append(False)
        is_twin.append(True)
    elif 45001 <= cid:
        child_ids.append(cid)
        is_triplet.append(False)
        is_twin.append(False)

solver = pywraplp.Solver("SolveAssignmentProblemMIP", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

x = {}

# child_ids = [0, 3, 5]
# leftover_gift_ids = [0, 10, 20]
# n_gifts = [3, 2, 1]
# is_triplet = [True, False, False]
# is_twin = [False, True, False]

# cost_mat = [
#     [10, 11, 12],
#     [10, 11, 12],
#     [10, 11, 12]
# ]
# cost_mat = np.array(cost_mat)
child_happiness_mat += 10

for cid in child_ids:
    for gid in leftover_gift_ids:
        x[cid, gid] = solver.BoolVar("x_{}_{}".format(cid, gid))

# Objective
objective = 0
for i, cid in enumerate(child_ids):
    for j, gid in enumerate(leftover_gift_ids):
        # objective += x[cid, gid] * cost_mat[i, j]
        # TODO: Bug Fix for Triplet and Twins
        # objective += x[cid, gid] * child_happiness_mat[cid, gid]
        if cid <= 5000:
            happiness = (child_happiness_mat[cid, gid]
                         + child_happiness_mat[cid+1, gid]
                         + child_happiness_mat[cid+2, gid])# * 2
            if happiness != 0:
                objective += x[cid, gid] * happiness
        elif 5001 <= cid <= 45000:
            happiness = (child_happiness_mat[cid, gid]
                         + child_happiness_mat[cid+1, gid])# * 3
            if happiness != 0:
                objective += x[cid, gid] * happiness
        else:
            happiness = child_happiness_mat[cid, gid]# * 6
            if happiness != 0:
                objective += x[cid, gid] * happiness

solver.Maximize(objective)

# Constraints
# The numbers of the gifts are limited
for i, gid in enumerate(leftover_gift_ids):
    gift_constraint = 0
    for j, cid in enumerate(child_ids):
        if is_triplet[j]:
            gift_constraint += 3 * x[cid, gid]
        elif is_twin[j]:
            gift_constraint += 2 * x[cid, gid]
        else:
            gift_constraint += x[cid, gid]
    solver.Add(gift_constraint == n_gifts[i])

# Each child can get exactly one gift
for i, cid in enumerate(child_ids):
    child_constraint = 0
    for j, gid in enumerate(leftover_gift_ids):
        child_constraint += x[cid, gid]
    solver.Add(child_constraint == 1)

print("Children: {}".format(len(child_ids)))
print("Gifts: {}".format(len(leftover_gift_ids)))
print("Solve MIP...")
result = solver.Solve()
print("Result: {}".format(result))
print('Total cost = ', solver.Objective().Value())
for cid in child_ids:
    for gid in leftover_gift_ids:
        if x[cid, gid].solution_value() > 0:
            # print("Child {} get gift {}".format(cid, gid))
            if cid <= 5000:
                assert cid % 3 == 0
                assert gift_ids[cid] == gift_ids[cid+1] == gift_ids[cid+2] == -1
                gift_ids[cid] = gift_ids[cid + 1] = gift_ids[cid + 2] = gid
            elif 5001 <= cid <= 45000:
                assert cid % 2 == 1
                assert gift_ids[cid] == gift_ids[cid+1] == -1
                gift_ids[cid] = gift_ids[cid+1] = gid
            else:
                assert gift_ids[cid] == -1
                gift_ids[cid] = gid

print("Time = ", solver.WallTime(), " milliseconds")
df_result = pd.DataFrame()
df_result["ChildId"] = np.arange(1000000)
df_result["GiftId"] = gift_ids
# df_result.to_csv("ultimate_final3.csv", index=False)
df_result.to_csv("submission.csv", index=False)

# 2000, 1600, 800, 0.015
