# Reward

Coleman supports reward formulations inspired by TCP, ranking, and cost-aware literature:

- RNFail (binary fault signal)
- TimeRank (failure-aware order-sensitive reward)
- ReciprocalRank (inverse-rank gain)
- TopKRNFail (prefix-constrained binary reward; precision-style)
- DiscountedFailure (DCG-like logarithmic discount)
- APFDc (cost-aware reward using execution time and detected failure positions)

Literature references:

- Spieker, H.; Gotlieb, A.; Marijan, D.; Mossige, M. (2017). Reinforcement Learning for Automatic Test Case Prioritization and Selection in Continuous Integration. ISSTA.
- Jarvelin, K.; Kekalainen, J. (2002). Cumulated gain-based evaluation of IR techniques. ACM TOIS.
- Rothermel, G.; Untch, R. H.; Chu, C.; Harrold, M. J. (2001). Prioritizing test cases for regression testing. IEEE TSE.
- Elbaum, S.; Malishevsky, A. G.; Rothermel, G. (2002). Test case prioritization: a family of empirical studies. IEEE TSE.
- Manning, C. D.; Raghavan, P.; Schutze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.

::: coleman.reward
