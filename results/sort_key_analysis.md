# Why PCA-sort does not beat value-sort

Scalar DC replaces a block of BS=16 rows with one number `mu_B`. The squared error
decomposes exactly:
```
sum_{r in B} sum_d (w[r,d] - mu_B)^2
  = sum_r sum_d (w[r,d] - mu_r)^2    <- WITHIN-ROW scatter: independent of ordering
  + D * sum_r (mu_r - mu_B)^2        <- BETWEEN-ROW scatter: the only orderable term
```
The within-row term is each row's spread around its own mean; no permutation changes
it. Sorting can only reduce the between-row term, which depends on row means alone --
and sorting by row mean is *exactly* optimal for it. Value-sort is the optimum, not a
heuristic. PCA-sort orders by a different statistic and can only tie or lose.

## Measured (300k sampled rows per table, 4 largest tables per dataset)
```

==============================================================================
Criteo Terabyte  (D=64)
==============================================================================
             table      rows  corr(PC1,mean)   within%  between: value         pca      random
   emb_l.19.weight    300000         +0.8114    100.0%       0.0003038    0.007253     0.02033
                                                        total MSE: value 0.6754  pca 0.6824  random 0.6954
    emb_l.0.weight    300000         +0.8814     99.7%        0.002598    0.009479     0.03127
                                                        total MSE: value 0.7607  pca 0.7675  random 0.7893
   emb_l.21.weight    300000         +0.4728     99.8%        0.001451     0.01067     0.01217
                                                        total MSE: value 0.674  pca 0.6832  random 0.6847
    emb_l.9.weight    300000         +0.7996     99.9%       0.0009787    0.008352      0.0205
                                                        total MSE: value 0.7085  pca 0.7159  random 0.728

  |cos(PC1, all-ones)| on largest table: 0.5190   (1.0 => PCA-sort IS value-sort)

==============================================================================
Criteo Kaggle  (D=16)
==============================================================================
             table      rows  corr(PC1,mean)   within%  between: value         pca      random
    emb_l.2.weight    300000         +0.9851     97.2%          0.0165     0.02504      0.3002
                                                        total MSE: value 0.5835  pca 0.592  random 0.8671
   emb_l.11.weight    300000         +0.9858     99.3%        0.006203     0.01713      0.3903
                                                        total MSE: value 0.8503  pca 0.8613  random 1.234
   emb_l.20.weight    300000         -0.9255     99.4%        0.002603     0.01591     0.09467
                                                        total MSE: value 0.4377  pca 0.451  random 0.5298
   emb_l.15.weight    300000         +0.9901     99.6%        0.008395     0.03008       1.052
                                                        total MSE: value 1.903  pca 1.925  random 2.947

  |cos(PC1, all-ones)| on largest table: 0.7490   (1.0 => PCA-sort IS value-sort)
```

## Conclusions

1. **At D=64 the orderable term is <=0.3% of the total error**, so no sort key can
   matter much. This is why value/pca/freq collapse together on Terabyte
   (-0.0341 / -0.0338 / -0.0818 % AUC at 4.3% hot -- only freq separates).
2. **At D=16 it is up to 2.8%**, ~10x more leverage. More dimensions means more
   within-row spread that one scalar cannot capture, so the orderable fraction
   shrinks as D grows. That alone explains the Kaggle/Terabyte difference.
3. **PCA-sort was worse than value-sort on all 8 tables tested, Kaggle included.**
   Its between-row term is 7-24x larger than value-sort's.
4. On Kaggle, corr(PC1, row mean) is 0.93-0.99 -- the two orderings are nearly the
   same permutation. The README's claim that PCA gives 17-26% less AUC loss than
   value-sort on Kaggle therefore has no mechanism behind it: two near-identical
   orderings, both acting on a term worth <3% of the error, cannot differ by 17-26%.
   That figure came from the contextual-bandit agent; re-run before publishing.
5. Practical consequence: **use value-sort**. It is optimal for scalar DC, costs one
   O(N log N) sort, and needs no power iteration.
