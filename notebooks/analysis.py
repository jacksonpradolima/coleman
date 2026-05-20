"""Coleman Analysis Notebook.

Marimo notebook for analyzing Coleman experiment results.
Includes performance metrics, visualizations, and statistical tests.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    """Import libraries and configure plotting."""
    from enum import Enum

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scikit_posthocs as posthocs
    import seaborn as sns
    from scipy.stats import kruskal

    from coleman.statistics import vargha_delaney

    plt.style.use("ggplot")
    sns.set_style("whitegrid")
    sns.set(palette="pastel")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)


@app.cell
def connect_db():
    """Connect to the DuckDB experiments database."""
    conn = duckdb.connect("experiments.db")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        scenario VARCHAR,
        experiment_id INTEGER,
        step INTEGER,
        policy VARCHAR,
        reward_function VARCHAR,
        sched_time FLOAT,
        sched_time_duration FLOAT,
        total_build_duration FLOAT,
        prioritization_time FLOAT,
        detected INTEGER,
        missed INTEGER,
        tests_ran INTEGER,
        tests_not_ran INTEGER,
        ttf FLOAT,
        ttf_duration FLOAT,
        time_reduction FLOAT,
        fitness FLOAT,
        cost FLOAT,
        rewards FLOAT,
        avg_precision FLOAT,
        prioritization_order VARCHAR
    );
    """)
    return (conn,)


@app.cell
def preprocess(conn):
    """Filter and preprocess experiment data."""
    conn.execute("""
    CREATE OR REPLACE TABLE filtered_data AS
    SELECT scenario, experiment_id, step, policy, reward_function,
           sched_time, sched_time_duration, total_build_duration,
           prioritization_time, ttf, ttf_duration, time_reduction,
           fitness, cost, avg_precision
    FROM experiments
    WHERE reward_function = 'Time-ranked Reward'
    """)
    conn.execute("""
    UPDATE filtered_data
    SET policy = CASE
        WHEN policy LIKE '\u03b5-greedy (\u03b5=0.5)' THEN '\u03b5-greedy'
        WHEN policy LIKE 'UCB (C=0.5)' THEN 'UCB'
        WHEN policy LIKE 'FRRMAB (C=0.5, D=1, SW=100)' THEN 'FRRMAB'
        WHEN policy LIKE 'Greedy' THEN 'Greedy'
        ELSE 'Random'
    END
    """)
    conn.execute("""
    UPDATE filtered_data
    SET reward_function = CASE
        WHEN reward_function LIKE 'Time-ranked Reward' THEN 'TimeRank'
        WHEN reward_function LIKE 'Reward Based on Failures' THEN 'RNFail'
    END
    """)
    main_data = conn.execute("SELECT DISTINCT scenario, policy, sched_time FROM filtered_data").fetchdf()
    scenarios = main_data["scenario"].unique()
    policies = main_data["policy"].unique()
    sched_times = main_data["sched_time"].unique()
    return (sched_times,)


@app.cell
def reward_enum():
    """Define the RewardFunction enum."""

    class RewardFunction(Enum):
        TimeRank = "Time-ranked Reward"
        RNFail = "Reward Based on Failures"

    return (RewardFunction,)


@app.cell
def helpers():
    """Helper function for metric labels."""

    def get_metric_ylabel(column, rw=None):
        metric = "NAPFD"
        ylabel = metric
        if "cost" in column:
            metric = "APFDc"
            ylabel = metric
        elif "ttf" in column:
            metric = "RFTC"
            ylabel = "Rank of the Failing Test Cases"
        elif "prioritization_time" in column:
            metric = "PrioritizationTime"
            ylabel = "Prioritization Time (sec.)"
        elif "rewards" in column:
            metric = rw
            ylabel = rw
        return metric, ylabel

    return (get_metric_ylabel,)


@app.cell
def accumulative_data(conn):
    """Compute accumulative fitness and cost data."""
    acc_data = conn.execute("""
    SELECT step, scenario, policy, sched_time, reward_function,
           SUM(fitness) OVER (PARTITION BY scenario, policy, sched_time, reward_function ORDER BY step) AS cum_fitness,
           SUM(cost) OVER (PARTITION BY scenario, policy, sched_time, reward_function ORDER BY step) AS cum_cost
    FROM filtered_data
    ORDER BY step, policy, sched_time
    """).fetchdf()
    return (acc_data,)


@app.cell
def plot_acc_fn(acc_data, get_metric_ylabel, sched_times):
    """Define accumulative plot function."""

    def plot_accumulative(column, scenarios=None, reward_function=None):
        df_filtered = acc_data[acc_data["scenario"].isin(scenarios)] if scenarios else acc_data
        if reward_function:
            df_filtered = df_filtered[df_filtered["reward_function"] == reward_function.name]
        fig, axes = plt.subplots(nrows=1, ncols=len(sched_times), sharey=True, figsize=(15, 5))
        if len(sched_times) == 1:
            axes = [axes]
        metric, _ = get_metric_ylabel(column)
        for ax, sched_time in zip(axes, sched_times):
            df_st = df_filtered[df_filtered["sched_time"] == sched_time]
            for name, group_df in df_st.groupby("policy"):
                group_df.plot(x="step", y=column, ax=ax, label=name, linewidth=1)
            ax.set_title(f"Time Budget: {round(sched_time * 100)}%")
            ax.set_xlabel("CI Cycle")
            ax.set_ylabel(f"Accumulative {metric}")
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return fig

    return (plot_accumulative,)


@app.cell
def show_acc_napfd(RewardFunction, plot_accumulative):
    """Plot accumulative NAPFD."""
    plot_accumulative(
        "cum_fitness",
        scenarios=["alibaba@druid"],
        reward_function=RewardFunction.TimeRank,
    )
    return


@app.cell
def show_acc_cost(RewardFunction, plot_accumulative):
    """Plot accumulative APFDc."""
    plot_accumulative(
        "cum_cost",
        scenarios=["alibaba@druid"],
        reward_function=RewardFunction.TimeRank,
    )
    return


@app.cell
def variation_data(conn):
    """Compute variation data."""
    variation_data = conn.execute("""
    SELECT scenario, step, policy, reward_function,
           AVG(fitness) AS fitness_variation,
           AVG(cost) AS cost_variation
    FROM filtered_data
    GROUP BY scenario, step, policy, reward_function
    ORDER BY scenario, step, policy, reward_function
    """).fetchdf()
    return (variation_data,)


@app.cell
def plot_var_fn(get_metric_ylabel, variation_data):
    """Define variation plot function."""

    def plot_metric_variation(column, title=None, scenarios=None, reward_function=None):
        df_filtered = variation_data[variation_data["scenario"].isin(scenarios)] if scenarios else variation_data
        if reward_function:
            df_filtered = df_filtered[df_filtered["reward_function"] == reward_function.name]
        metric, _ = get_metric_ylabel(column)
        pivot_df = df_filtered.pivot(index="step", columns="policy", values=column)
        fig, ax = plt.subplots(figsize=(15, 5))
        pivot_df.plot(ax=ax, linewidth=1)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(f"Average {metric}", fontsize=12)
        plot_title = title if title else f"Average {metric} per Step for Each Policy"
        ax.set_title(plot_title, fontsize=14)
        ax.legend(title="Policy")
        plt.tight_layout()
        return fig

    return (plot_metric_variation,)


@app.cell
def show_var_napfd(RewardFunction, plot_metric_variation):
    """Plot NAPFD variation."""
    plot_metric_variation(
        "fitness_variation",
        scenarios=["alibaba@druid"],
        reward_function=RewardFunction.TimeRank,
    )
    return


@app.cell
def show_var_cost(RewardFunction, plot_metric_variation):
    """Plot APFDc variation."""
    plot_metric_variation(
        "cost_variation",
        scenarios=["alibaba@druid"],
        reward_function=RewardFunction.TimeRank,
    )
    return


@app.cell
def ntr_data(conn):
    """Compute normalized time reduction data."""
    ntr_by_policy = conn.execute("""
    SELECT scenario, experiment_id, sched_time, policy, reward_function,
           (SUM(time_reduction) / SUM(total_build_duration)) * 100 AS n_reduction
    FROM filtered_data
    WHERE avg_precision = 123
    GROUP BY scenario, experiment_id, sched_time, policy, reward_function
    ORDER BY sched_time, policy, reward_function, scenario
    """).fetchdf()
    return (ntr_by_policy,)


@app.cell
def plot_ntr_fn(ntr_by_policy):
    """Define normalized time reduction plot function."""

    def plot_normalize_time_reduction(scenarios=None, reward_function=None):
        df_filtered = ntr_by_policy[ntr_by_policy["scenario"].isin(scenarios)] if scenarios else ntr_by_policy
        if reward_function:
            df_filtered = df_filtered[df_filtered["reward_function"] == reward_function.name]
        st = df_filtered["sched_time"].unique()
        fig, axes = plt.subplots(nrows=1, ncols=len(st), sharey=True, figsize=(15, 5))
        if len(st) == 1:
            axes = [axes]
        for ax, sched_time in zip(axes, st):
            df_st = df_filtered[df_filtered["sched_time"] == sched_time]
            sns.boxplot(x="policy", y="n_reduction", data=df_st, ax=ax)
            ax.set_title(f"Time Budget: {round(sched_time * 100)}%")
            ax.set_xlabel("Policy")
            ax.set_ylabel("Normalized Time Reduction (%)")
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return fig

    return (plot_normalize_time_reduction,)


@app.cell
def show_ntr(RewardFunction, plot_normalize_time_reduction):
    """Plot normalized time reduction."""
    plot_normalize_time_reduction(scenarios=["alibaba@druid"], reward_function=RewardFunction.TimeRank)
    return


@app.cell
def mean_dist_data(conn):
    """Compute mean distribution data."""
    mean_distribution = conn.execute("""
    SELECT scenario, experiment_id, sched_time, policy, reward_function,
           AVG(prioritization_time) AS avg_prioritization_time,
           AVG(fitness) AS avg_fitness_time,
           AVG(cost) AS avg_cost_time
    FROM filtered_data
    GROUP BY scenario, experiment_id, sched_time, policy, reward_function
    ORDER BY sched_time, policy, reward_function, scenario
    """).fetchdf()
    return (mean_distribution,)


@app.cell
def plot_dist_fn(get_metric_ylabel, mean_distribution, sched_times):
    """Define distribution plot function."""

    def plot_distribution(column, scenarios=None, reward_function=None):
        df_filtered = (
            mean_distribution[mean_distribution["scenario"].isin(scenarios)] if scenarios else mean_distribution
        )
        if reward_function:
            df_filtered = df_filtered[df_filtered["reward_function"] == reward_function.name]
        fig, axes = plt.subplots(nrows=1, ncols=len(sched_times), sharey=True, figsize=(15, 5))
        if len(sched_times) == 1:
            axes = [axes]
        metric, _ = get_metric_ylabel(column)
        for ax, sched_time in zip(axes, sched_times):
            df_st = df_filtered[df_filtered["sched_time"] == sched_time]
            sns.boxplot(x="policy", y=column, data=df_st, ax=ax)
            ax.set_title(f"Time Budget: {round(sched_time * 100)}%")
            ax.set_xlabel("Policy")
            ax.set_ylabel("Average Prioritization Time (seconds)")
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return fig

    return (plot_distribution,)


@app.cell
def show_dist_ptime(RewardFunction, plot_distribution):
    """Plot prioritization time distribution."""
    plot_distribution(
        "avg_prioritization_time",
        scenarios=["alibaba@druid"],
        reward_function=RewardFunction.TimeRank,
    )
    return


@app.cell
def show_dist_napfd(RewardFunction, plot_distribution):
    """Plot NAPFD distribution."""
    plot_distribution(
        "avg_fitness_time",
        scenarios=["alibaba@druid"],
        reward_function=RewardFunction.TimeRank,
    )
    return


@app.cell
def show_dist_cost(RewardFunction, plot_distribution):
    """Plot APFDc distribution."""
    plot_distribution(
        "avg_cost_time",
        scenarios=["alibaba@druid"],
        reward_function=RewardFunction.TimeRank,
    )
    return


@app.cell
def stat_test_fns():
    """Define statistical test functions."""

    def perform_statistical_test(dataframe, group_column, value_column, alpha=0.05):
        groups = dataframe.groupby(group_column)[value_column].apply(list).values
        stat, p_value = kruskal(*groups)
        all_equivalent = p_value >= alpha
        posthoc_result = None
        if not all_equivalent:
            posthoc_result = posthocs.posthoc_nemenyi(dataframe, val_col=value_column, group_col=group_column)
        vda_result = vargha_delaney.vd_a_df(dataframe, val_col=value_column, group_col=group_column)
        return {
            "kruskal_stat": stat,
            "kruskal_pvalue": p_value,
            "all_equivalent": all_equivalent,
            "nemenyi_posthoc": posthoc_result,
            "vda_result": vda_result,
        }

    def calculate_policy_statistics(df, column, direction="max"):
        stats = df.groupby(["policy"], as_index=False).agg({column: ["mean", "std", "max", "min"]})
        stats.columns = ["policy", "mean", "std", "max", "min"]
        stats = stats.round({"mean": 4, "std": 3, "max": 4, "min": 4})
        best_policy = stats.loc[
            stats["mean"].idxmax() if direction == "max" else stats["mean"].idxmin(),
            "policy",
        ]
        return stats, best_policy

    def determine_effect_size_symbol(policy, best_policy, effect_size_df):
        if policy == best_policy:
            return "$\\bigstar$"
        vals = effect_size_df.loc[effect_size_df.compared_with == policy, "effect_size_symbol"].values
        if vals.size > 0:
            return vals[0]
        return effect_size_df.loc[effect_size_df.base == policy, "effect_size_symbol"].values[0]

    def generate_latex_configuration(row, best_policy, posthoc_df, all_equivalent):
        current_policy = row["policy"]
        if all_equivalent:
            return f"\\cellgray{{{row['avg_std_effect']}}}"
        if current_policy == best_policy:
            return f"\\cellbold{{{row['avg_std_effect']}}}"
        is_equivalent = False
        if posthoc_df is not None:
            if (
                best_policy in posthoc_df.columns
                and current_policy in posthoc_df.index
                and not np.isnan(posthoc_df.loc[current_policy][best_policy])
            ):
                is_equivalent = posthoc_df.loc[current_policy][best_policy] >= 0.05
            elif (
                current_policy in posthoc_df.columns
                and best_policy in posthoc_df.index
                and not np.isnan(posthoc_df.loc[best_policy][current_policy])
            ):
                is_equivalent = posthoc_df.loc[best_policy][current_policy] >= 0.05
        return f"\\cellgray{{{row['avg_std_effect']}}}" if is_equivalent else row["avg_std_effect"]

    def statistical_test_procedure(dataframe, column, use_latex=False):
        sched_times_local = dataframe["sched_time"].unique()
        policies_local = dataframe["policy"].unique()
        stat_columns = np.insert(policies_local, 0, "TimeBudget")
        df_stats = pd.DataFrame(columns=stat_columns)
        for sched_time in sched_times_local:
            df_st = dataframe[dataframe["sched_time"] == sched_time]
            statistical_results = perform_statistical_test(df_st, "policy", column)
            stats, best_policy = calculate_policy_statistics(df_st, column)
            df_reduced = vargha_delaney.reduce(statistical_results["vda_result"], best_policy)
            if use_latex:
                stats["eff_symbol"] = stats.apply(
                    lambda x: determine_effect_size_symbol(x["policy"], best_policy, df_reduced),
                    axis=1,
                )
                stats["avg_std_effect"] = stats.apply(
                    lambda row: f"{row['mean']:.4f} $\\pm$ {row['std']:.4f} {row['eff_symbol']}".strip(),
                    axis=1,
                )
                stats["latex_format"] = stats.apply(
                    lambda row: generate_latex_configuration(
                        row,
                        best_policy,
                        statistical_results["nemenyi_posthoc"],
                        statistical_results["all_equivalent"],
                    ),
                    axis=1,
                )
            stats_trans = stats[["policy", "latex_format"]].copy() if use_latex else stats[["policy", "mean"]].copy()
            stats_trans.index = stats["policy"]
            stats_trans = stats_trans.transpose().drop(stats_trans.index[0])
            row_vals = np.insert(stats_trans.values[0], 0, sched_time)
            df_stats = pd.concat(
                [
                    df_stats,
                    pd.DataFrame(
                        [row_vals],
                        columns=np.append("TimeBudget", stats_trans.columns.values),
                    ),
                ],
                ignore_index=True,
            )
        df_stats["Metric"] = column
        return df_stats

    return (statistical_test_procedure,)


@app.cell
def run_stat_ptime(mean_distribution, statistical_test_procedure):
    """Statistical test: Prioritization Time."""
    df_stats_ptime = statistical_test_procedure(mean_distribution, "avg_prioritization_time")
    return


@app.cell
def run_stat_napfd(mean_distribution, statistical_test_procedure):
    """Statistical test: NAPFD."""
    df_stats_napfd = statistical_test_procedure(mean_distribution, "avg_fitness_time")
    return


@app.cell
def run_stat_cost(mean_distribution, statistical_test_procedure):
    """Statistical test: APFDc."""
    df_stats_cost = statistical_test_procedure(mean_distribution, "avg_cost_time")
    return


if __name__ == "__main__":
    app.run()
