# Solution for Santa Gift Matching Challenge 

This is a silver medal (38th) solution for [Santa Gift Matching Challenge](https://www.kaggle.com/c/santa-gift-matching) in Kaggle.

## Algorithm

- Solve minimum cost flow problem and obtain a relaxed solution
- Unfix some variables by heuristic rules
- Solve mixed integer problem

## How to Run

### Download Data

```bash
kaggle competitions download -c santa-gift-matching -p data
unzip data/child_wishlist_v2.csv.zip -d data
unzip data/gift_goodkids_v2.csv.zip -d data
```

### Obtain Relaxed Solution

```bash
python min_cost_flow.py
```

### Solve MIP

```bash
python mip.py
```

### Score

```bash
python score.py
```

You will get score 0.9362270126.
That is slightly smaller than my best score 0.9362916416 because the heuristic condition to fix variables in `mip.py` is weaken for computational cost.
But, this result can reach silver medal!
