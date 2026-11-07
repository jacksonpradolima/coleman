# Research behind the workflows

Use this reading path to connect a capability to the question it was designed to
study. A paper's findings apply to its own subjects, assumptions and experiment;
they do not validate every later policy, adapter or dataset in this repository.

## Adaptive prioritization

**Prado Lima and Vergilio. _A Multi-Armed Bandit Approach for Test Case
Prioritization in Continuous Integration Environments._ IEEE Transactions on
Software Engineering 48(2), 453–465, 2022.**
[DOI: 10.1109/TSE.2020.2992428](https://doi.org/10.1109/TSE.2020.2992428).

The original COLEMAN article motivates learning from test feedback as suites
change across CI cycles. Read it alongside [the prioritization loop](concepts.md),
[policy choices](policies.md) and [native experiments](experiments.md).
The DOI contains 2020, reflecting its publication history; the volume/issue
citation is 2022. [CITATION.cff](https://github.com/jacksonpradolima/coleman/blob/main/CITATION.cff)
records the preferred citation.

**Li, Chu, Langford and Schapire. _A Contextual-Bandit Approach to Personalized
News Article Recommendation._ WWW, 2010.**
[Author preprint](https://arxiv.org/abs/1003.0146).

This introduces LinUCB in a recommendation setting. It provides algorithmic
background for a contextual family, rather than evidence about this CI runner.
Continue with [contextual feature availability](policies.md#contextual-policies)
and [point-in-time observations](point-in-time-observations.md).

**Auer, Cesa-Bianchi and Fischer. _Finite-time Analysis of the Multiarmed Bandit
Problem._ Machine Learning 47, 235–256, 2002.**
[Publisher and DOI](https://doi.org/10.1023/A:1013689704352).

Read this for classical confidence-based exploration under its stated reward
assumptions. The [policy guide](policies.md) distinguishes that theoretical setting
from a particular CI history, reward transformation or implementation.

## Full-information expert mixtures

**Freund and Schapire. _A decision-theoretic generalization of on-line learning
and an application to boosting._ Journal of Computer and System Sciences 55(1),
119–139, 1997.** [Author paper](https://www.schapire.net/papers/FreundSc95.pdf).

This supplies the decision-theoretic mixture-loss setting: the complete bounded
expert loss vector is observed, and the mixture is compared with a retrospective
fixed expert. Follow the [expert-mixture walkthrough](expert-mixtures.md) for
COLEMAN's exact input contract and constant-rate method. A coefficient vector
does not establish an executed CI outcome or a random selection mechanism.

**Mourtada and Gaïffas. _On the optimality of the Hedge algorithm in the
stochastic regime._ Journal of Machine Learning Research 20(83), 1–28, 2019.**
[Original article and paper](https://jmlr.org/papers/v20/18-869.html).

Read this for the decreasing-rate anytime formulation and its distinctions from
constant-rate and doubling variants. COLEMAN recomputes from cumulative past
losses using the current rate. The guide states the mathematical comparator,
prior normalization, synchronous assumptions and finite-arithmetic limits;
delayed processing has a separate replay qualification.

**Herbster and Warmuth. _Tracking the Best Expert._ Machine Learning 32,
151–178, 1998.** [Author paper](https://mwarmuth.bitbucket.io/pubs/J39.pdf).

Its Fixed Share construction motivates the explicitly named excluding-origin
reference. The implemented transition redistributes to the other experts; a
share toward a declared target is a separate method. The controlled projection
and L1 movement variants have their own objectives and qualifications, rather
than inheriting the references' regret statements or promising lower domain
costs.

## Structured policies and subset feedback

**Chen, Wang and Yuan. _Combinatorial Multi-Armed Bandit: General Framework and
Applications._ ICML, PMLR 28(1), 151–159, 2013.**
[Original proceedings and paper](https://proceedings.mlr.press/v28/chen13a.html).

This provides a foundation for thinking about subset actions, component feedback
and reward structure. Read it alongside [the policy families](policies.md) and
[finite coverage selection](finite-coverage.md). A native full-ranking adaptation
or a finite solver is not automatically the paper's online feedback/oracle model;
check the actual assumptions before applying a guarantee.

## Information availability and data leakage

**Kaufman, Rosset, Perlich and Stitelman. _Leakage in data mining._
ACM Transactions on Knowledge Discovery from Data 6(4), 2012.**
[Publisher and DOI](https://doi.org/10.1145/2382577.2382579).

This explains how information unavailable at prediction time can compromise an
experiment. In CI, an eventual verdict, corrected duration or late report can be
present in a later export without being available to an earlier policy.
[Late results and corrections](decisions/001-point-in-time-observations.md)
explains that problem through an execution timeline.
[Point-in-time observations](point-in-time-observations.md) provides the complete
API example, while [protected evaluation](protected-evaluation.md) adds the temporal
and grouped comparison protocol.

## Configurations and variants

**Prado Lima, Mendonça, Vergilio and Assunção. _Learning-based prioritization of
test cases in continuous integration of highly-configurable software._ SPLC, 2020.**
[DOI: 10.1145/3382025.3414967](https://doi.org/10.1145/3382025.3414967).

**The same authors. _Cost-effective learning-based strategies for test case
prioritization in continuous integration of highly-configurable software._
Empirical Software Engineering 27, article 133, 2022.**
[Publisher and DOI](https://doi.org/10.1007/s10664-021-10093-3).

These studies motivate prioritization across changing variants. Use
[WTS and VTS explained](highly-configurable.md) before following the
[LibSSH workflow](hcs-parquet-etl.md). The current
[original CTest evidence](hcs-trace-evidence.md) adds source corroboration; it does
not reproduce all findings or qualify every historical source assumption.

## Datasets and source studies

| Publication or original source | Why read it | COLEMAN workflow |
| --- | --- | --- |
| Mattis, Rein, Dürsch and Hirschfeld, _RTPTorrent_, MSR 2020. [DOI](https://doi.org/10.1145/3379597.3387458) | Understand the source's collected class results and research context | [Source assessment](rtptorrent.md), [Parquet and replay](rtptorrent-parquet-etl.md) |
| Cheng, Wang, Jabbarvand and Marinov, _Revisiting Test-Case Prioritization on Long-Running Test Suites_, ISSTA 2024. [DOI](https://doi.org/10.1145/3650212.3680307), [authors' artifact](https://github.com/lrtsuser/LRTS) | Understand the long-running-suite source and its original processing | [Source assessment](lrts.md), [Karaf replay](lrts-parquet-etl.md), [stage views](lrts-stage-views.md) |
| Strandberg, _The Westermo test results data set_, 2022. [Author paper, v1](https://arxiv.org/abs/2211.13622v1) | Understand embedded-system observations, anonymization and source context | [Original analytical Parquet](westermo-parquet-etl.md) |
| Original highly configurable system data. [Dataverse versioned source](https://doi.org/10.7910/DVN/I1XAVR) | Inspect component terms and original CSV/JSON/log relationships | [LibSSH data path](hcs-parquet-etl.md), [CTest corroboration](hcs-trace-evidence.md) |

A source paper describes the original study. COLEMAN's adapter guides separately
state byte pins, units, exclusions and current qualification limits. Use both;
do not infer missing source clocks or test inventories from a publication's
aggregate statistics.

## Evaluation and interpretation

| Original method reference | Role in the evaluation path |
| --- | --- |
| Student, _The Probable Error of a Mean_, Biometrika, 1908. [DOI](https://doi.org/10.1093/biomet/6.1.1) | Background for the legacy normal paired-difference protocol; read its actual assumptions in [protected evaluation](protected-evaluation.md) |
| Hoeffding, _Probability Inequalities for Sums of Bounded Random Variables_, JASA, 1963. [DOI](https://doi.org/10.1080/01621459.1963.10500830) | Concentration bound underlying the supported [bounded paired method](bounded-paired-evaluation.md), with independent bounded inference units |
| Holm, _A Simple Sequentially Rejective Multiple Test Procedure_, Scandinavian Journal of Statistics, 1979. [Original article](https://www.jstor.org/stable/4615733) | Step-down multiplicity control for the full frozen comparison family; not a way to create independence |


The [protected-evaluation guide](protected-evaluation.md) declares supported
comparison assumptions, pairing, multiplicity and practical-equivalence rules.
The [bounded paired guide](bounded-paired-evaluation.md) documents its conservative
bounds and independence requirements. Read the actual implemented protocol before
choosing an analysis by name.

[Repeated observations](corpus-evaluation.md) explains why builds, seeds and
variants cannot be pooled as independent projects. [Source admission](corpus-admission.md)
separates source integrity from scientific eligibility. [Evidence status](roadmap.md)
explains the missing prerequisites for historical cross-corpus confirmation.

The [independent decision benchmarks](decision-benchmarks.md) validate bounded
mathematical routines against independent references. Such checks establish
correctness under declared assumptions, rather than empirical policy superiority.

## Cite the software and the source

Cite the original COLEMAN article when the method contributes to your work. Cite
the HCS studies when using WTS/VTS, and each dataset's original publication and
version when using its observations. Record the actual software version,
source/adapter/runtime identities, experiment specification and analysis protocol
in your reproducibility material.
