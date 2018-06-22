import numpy as np
import pandas as pd
from ortools.graph import pywrapgraph
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

print("computing cost matrix")
cost_mat = -6 * child_happiness_mat / 10
cost_mat = cost_mat.astype(np.int16)
max_cost = int(np.max(cost_mat))

print("Maximum cost: {}".format(max_cost))

# Instantiate a SimpleMinCostFlow solver
mcf = pywrapgraph.SimpleMinCostFlow()

n_child = 1000000
n_gift = 1000

print("Add arcs for triplets")
for cid in tqdm(range(5001)):
    if cid % 3 == 0:
        gift_ids = np.where(
            (cost_mat[cid] != max_cost) + (cost_mat[cid+1] != max_cost) + (cost_mat[cid+2] != max_cost)
        )[0]
        for gid in gift_ids:
            cost = (cost_mat[cid, gid] + cost_mat[cid+1, gid] + cost_mat[cid+2, gid]) / 3
            mcf.AddArcWithCapacityAndUnitCost(int(n_child+gid), cid, 3, int(cost))

print("Add arcs for twins")
for cid in tqdm(range(5001, 45001)):
    if cid % 2 == 1:
        gift_ids = np.where(
            (cost_mat[cid] != max_cost) + (cost_mat[cid+1] != max_cost)
        )[0]
        for gid in gift_ids:
            cost = int(cost_mat[cid, gid] + cost_mat[cid+1, gid]) / 2
            mcf.AddArcWithCapacityAndUnitCost(int(n_child+gid), cid, 2, int(cost))

print("Add arcs for singles")
for cid in tqdm(range(45001, n_child)):
    gift_ids = np.where(cost_mat[cid] != max_cost)[0]
    for gid in gift_ids:
        cost = int(cost_mat[cid, gid])
        mcf.AddArcWithCapacityAndUnitCost(int(n_child+gid), cid, 1, cost)

print("Add node supply for triplets")
for cid in range(5001):
    if cid % 3 == 0:
        mcf.SetNodeSupply(cid, -3)

print("Add node supply for twins")
for cid in range(5001, 45001):
    if cid % 2 == 1:
        mcf.SetNodeSupply(cid, -2)

print("Add node supply for singles")
for cid in tqdm(range(45001, n_child)):
    mcf.SetNodeSupply(cid, -1)

print("Add node supply for gifts")
for gid in tqdm(range(n_gift)):
    mcf.SetNodeSupply(n_child+gid, 1000)

print("Add arcs from gifts to generic gift")
for gid in tqdm(range(n_gift)):
    mcf.AddArcWithCapacityAndUnitCost(n_child+gid, n_child+n_gift, 1000, 0)

print("Add arcs from generic gift to triplets")
for cid in tqdm(range(5001)):
    if cid % 3 == 0:
        mcf.AddArcWithCapacityAndUnitCost(n_child+n_gift, cid, 3, max_cost)

print("Add arcs from generic gift to twins")
for cid in tqdm(range(5001, 45001)):
    if cid % 2 == 1:
        mcf.AddArcWithCapacityAndUnitCost(n_child+n_gift, cid, 2, max_cost)

print("Add arcs from generic gift to singles")
for cid in tqdm(range(45001, n_child)):
    mcf.AddArcWithCapacityAndUnitCost(n_child+n_gift, cid, 1, max_cost)

print("\nThe number of nodes: {}".format(mcf.NumNodes()))
print("The number of arcs: {}".format(mcf.NumArcs()))

del cost_mat, child_wishlist, gift_goodkids

print("Solve minimum cost flow...")
mcf_result = mcf.Solve()

n_infeasible_triplets = 0
n_infeasible_twins = 0
if mcf_result == mcf.OPTIMAL:
    print("\nGet optimal solution!")
    # result = np.full((1000000, 5), -10, dtype=int)
    # result[:, 0] = np.arange(1000000)
    result = []
    for i in tqdm(range(mcf.NumArcs())):
        # Arcs from gift nodes to generic gift node
        if mcf.Head(i) == 1001000:
            continue
        # Arcs from generic gift node to child nodes
        elif mcf.Tail(i) == 1001000:
            if mcf.Flow(i) > 0:
                cid = mcf.Head(i)
                happiness = -mcf.Flow(i) * mcf.UnitCost(i)
                result.append([cid, -1, mcf.Flow(i), happiness])
                # result[cid, 1] = -1
                # result[cid, 2] = mcf.Flow(i)
        elif mcf.Flow(i) > 0:
            gid = mcf.Tail(i) - n_child
            cid = mcf.Head(i)
            flow = mcf.Flow(i)
            happiness = -mcf.Flow(i) * mcf.UnitCost(i)
            result.append([cid, gid, flow, happiness])
            # result[cid, 1] = gid
            # result[cid, 2] = flow
            # result[cid, 3] = child_happiness_mat[cid, gid]
            # result[cid, 4] = gift_happiness_mat[cid, gid]

            if cid <= 5000 and cid % 3 == 0 and flow != 3:
                n_infeasible_triplets += 1
            if 5001 <= cid <= 45000 and cid % 2 == 1 and flow != 2:
                n_infeasible_twins += 1

    result = np.array(result)
    df = pd.DataFrame()
    df["ChildId"] = result[:, 0]
    df["GiftId"] = result[:, 1]
    df["Flow"] = result[:, 2]
    df["Happiness"] = result[:, 3]
    # df["GiftHappiness"] = result[:, 4]
    df.to_csv("min_cost_flow.csv", index=False)
