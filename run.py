"""Single entry point: run the taxi carshare simulation and plot the results.

Two modes
---------
1. Full comparison study (no arguments, or --compare to change the scenario):
     python run.py
     python run.py --compare --passengers 300 --seeds 4
   Runs 6 taxi configurations (individual -> 2-shared -> 4-shared + multi-hop + acceptability)
   plus two sweeps (boarding time, number of passengers) and writes:
     fig_configs.png     key metrics of the 6 configurations (bars)
     fig_breakdown.png   travel-time and emissions breakdown, makespan
     fig_timeseries.png  the 6 configurations over time
     fig_boarding.png    boarding-time sweep for 3 key configurations
     fig_passengers.png  demand sweep for 3 key configurations
     summary.csv, sweep_boarding.csv, sweep_passengers.csv
   With the default 8 seeds this takes a few minutes.

2. One configuration, chosen with the policy switches:
     python run.py --no-shared                       # individual taxis (1 passenger per taxi)
     python run.py --capacity 4 --no-reroute --no-multi-hop
     python run.py --capacity 2 --reroute --multi-hop --acceptability --passengers 300
   Writes summary.png, timeseries.png, summary.csv and timeseries.csv.

Every configuration is run on the same seeds, i.e. on the SAME passengers, so the
differences between configurations come from the taxi policy only.
Outputs go to --out (default ./results). matplotlib is used for plotting only.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import TaxiModel

# Columns of the final summary table (see `final_metrics`)
SUMMARY_COLS = ["completion_pct", "avg_travel_time", "avg_waiting_time", "avg_walking_time",
                "avg_taxi_legs", "multi_hop_completed", "mean_occupancy", "shared_distance_pct",
                "total_taxi_distance", "total_taxi_emissions", "emissions_per_passenger",
                "satisfied_pct", "n_messages", "makespan"]
COLORS = ["tab:red", "tab:orange", "tab:olive", "tab:blue", "tab:green", "tab:purple"]

# The 6 configurations of the comparison study: each step adds one feature,
# following the incremental design of the project brief.
STUDY_CONFIGS = {
    "Individual":                  dict(shared=False, capacity=1, route_change=False,
                                        multi_hop=False, acceptability=False),
    "2-shared, no re-route":       dict(shared=True, capacity=2, route_change=False,
                                        multi_hop=False, acceptability=False),
    "2-shared + re-route":         dict(shared=True, capacity=2, route_change=True,
                                        multi_hop=False, acceptability=False),
    "4-shared + re-route":         dict(shared=True, capacity=4, route_change=True,
                                        multi_hop=False, acceptability=False),
    "4-shared + multi-hop":        dict(shared=True, capacity=4, route_change=True,
                                        multi_hop=True, acceptability=False),
    "4-shared + multi-hop + accept.": dict(shared=True, capacity=4, route_change=True,
                                          multi_hop=True, acceptability=True),
}
# Configurations followed in the parameter sweeps (name: color)
SWEEP_KEYS = {"Individual": "tab:red", "2-shared + re-route": "tab:blue",
              "4-shared + multi-hop": "tab:green"}
BOARDING_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0]
PASSENGER_VALUES = [50, 100, 200, 300]


# ---------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(
        description="Taxi carshare multi-agent simulation (Mesa).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("taxi policy (the switches under study)")
    g.add_argument("--shared", action=argparse.BooleanOptionalAction, default=True,
                   help="carshare taxis; --no-shared = individual taxis (capacity 1)")
    g.add_argument("--capacity", type=int, default=4, choices=[1, 2, 3, 4],
                   help="passengers allowed per taxi when shared")
    g.add_argument("--reroute", action=argparse.BooleanOptionalAction, default=True,
                   help="taxis may change route to pick up a new passenger")
    g.add_argument("--multi-hop", action=argparse.BooleanOptionalAction, default=True,
                   help="inter-city trips go through city hubs (several taxi legs); "
                        "--no-multi-hop = single direct ride")
    g.add_argument("--acceptability", action=argparse.BooleanOptionalAction, default=False,
                   help="passengers reject too-slow / too-detoured offers and too many transfers")

    g = p.add_argument_group("scenario")
    g.add_argument("--passengers", type=int, default=200, help="number of passengers (demand)")
    g.add_argument("--taxis", type=int, default=8, help="number of taxis")
    g.add_argument("--cities", type=int, default=4, help="number of cities (4x4 districts each)")
    g.add_argument("--inter-city-prob", type=float, default=0.3,
                   help="probability that a trip goes to another city")
    g.add_argument("--boarding-time", type=float, default=0.5, help="time units per boarding")
    g.add_argument("--horizon", type=float, default=250.0,
                   help="departure times are drawn uniformly in [0, horizon]")

    g = p.add_argument_group("experiment")
    g.add_argument("--seeds", type=int, default=8, help="number of runs (seeds 1..N) to average")
    g.add_argument("--max-steps", type=int, default=2000,
                   help="stop a run after this many steps if passengers are still travelling")
    g.add_argument("--compare", action="store_true",
                   help="run the full comparison study (6 configurations + boarding and "
                        "passenger sweeps), ignoring the policy switches; this is also what "
                        "happens when run.py is started without any argument")
    g.add_argument("--out", default="results", help="output directory")
    g.add_argument("--no-plot", action="store_true", help="only print and save the tables")
    args = p.parse_args()
    args.compare = args.compare or len(sys.argv) == 1
    if args.shared and args.capacity == 1:
        p.error("--shared needs --capacity >= 2 (use --no-shared for individual taxis)")
    return args


def policy_label(cfg):
    if not cfg["shared"]:
        name = "Individual"
    else:
        name = f"{cfg['capacity']}-shared"
        name += ", re-route" if cfg["route_change"] else ", no re-route"
    name += ", multi-hop" if cfg["multi_hop"] else ", single-hop"
    if cfg["acceptability"]:
        name += ", accept."
    return name


def build_configs(args):
    """Return {label: model kwargs for the taxi policy}."""
    if args.compare:
        return STUDY_CONFIGS
    own = dict(shared=args.shared, capacity=args.capacity if args.shared else 1,
               route_change=args.reroute, multi_hop=args.multi_hop,
               acceptability=args.acceptability)
    return {policy_label(own): own}


# ---------------------------------------------------------------- simulation
def run_one(seed, policy, args):
    """Run one simulation; return (per-step DataFrame, dict of final metrics)."""
    model = TaxiModel(seed=seed, n_passengers=args.passengers, n_taxis=args.taxis,
                      n_cities=args.cities, inter_city_prob=args.inter_city_prob,
                      boarding_time=args.boarding_time, demand_horizon=args.horizon,
                      **policy).run(args.max_steps)
    df = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
    df.index.name = "step"
    return df, final_metrics(model, df)


def final_metrics(model, df):
    out = df.iloc[-1].to_dict()
    done = max(model.n_completed, 1)
    out["completion_pct"] = 100 * model.n_completed / max(model.n_created, 1)
    out["mean_occupancy"] = df["taxi_occupancy"].mean()
    out["shared_distance_pct"] = 100 * out["shared_distance"] / max(out["total_taxi_distance"], 1e-9)
    out["emissions_per_passenger"] = out["total_taxi_emissions"] / done
    out["satisfied_pct"] = 100 * out["satisfied_share"]
    out["makespan"] = model.sim_time
    return out


def run_config(label, policy, args):
    """Run all seeds of one configuration. Runs stop when every passenger has arrived,
    so the shorter time series are padded with their last value."""
    series, finals = [], []
    for seed in range(1, args.seeds + 1):
        df, fin = run_one(seed, policy, args)
        series.append(df)
        finals.append(fin)
    finals = pd.DataFrame(finals)
    print(f"  {label:<32} {finals['completion_pct'].mean():5.1f}% completed, "
          f"travel time {finals['avg_travel_time'].mean():5.2f}, "
          f"emissions/passenger {finals['emissions_per_passenger'].mean():4.2f}")
    length = max(len(df) for df in series)
    series = [df.reindex(range(length)).ffill() for df in series]
    return series, finals


def run_sweep(param, values, args):
    """Re-run the SWEEP_KEYS configurations for each value of one scenario parameter.
    Returns a long table: one row per (configuration, value, seed)."""
    rows = []
    for v in values:
        print(f"Sweep {param} = {v}")
        sub = argparse.Namespace(**{**vars(args), param: v})
        for label in SWEEP_KEYS:
            _, finals = run_config(label, STUDY_CONFIGS[label], sub)
            rows.append(finals.assign(config=label, value=v))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- plots
def stack(series, col, smooth=1):
    data = np.array([s[col].to_numpy(dtype=float) for s in series])
    if smooth > 1:
        data = np.array([pd.Series(r).rolling(smooth, min_periods=1).mean().to_numpy()
                         for r in data])
    return data


def plot_timeseries(results, args, path):
    # averages are NaN until the first passenger arrives: silence the all-NaN warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    panels = [("in_system", "Passengers waiting or travelling", "passengers", 1),
              ("completed", "Passengers arrived (cumulative)", "passengers", 1),
              ("avg_travel_time", "Average travel time (arrived passengers)", "time units", 1),
              ("taxi_occupancy", "Taxi occupancy (25-step moving average)", "passengers / taxi", 25),
              ("total_taxi_emissions", "Cumulative taxi emissions", "emission units", 1),
              ("congestion_delay", "Cumulative congestion delay", "time units", 1)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for ax, (col, title, ylabel, smooth) in zip(axes.ravel(), panels):
        for (label, (series, _)), color in zip(results.items(), COLORS):
            data = stack(series, col, smooth)
            x = np.arange(data.shape[1])
            ax.plot(x, np.nanmean(data, axis=0), color=color, label=label)
            if len(series) > 1:     # band = min..max over seeds
                ax.fill_between(x, np.nanmin(data, axis=0), np.nanmax(data, axis=0),
                                color=color, alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("simulation step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Over time: mean of {args.seeds} seed(s), {args.passengers} passengers, "
                 f"{args.taxis} taxis (bands = min-max)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def short_labels(labels):
    return [l.replace(", ", "\n").replace(" + ", "\n+ ") for l in labels]


def plot_configs(results, args, path):
    """Key metrics of each configuration (mean +- std over seeds)."""
    labels = list(results)
    x = np.arange(len(labels))
    metrics = [("avg_travel_time", "Avg travel time"),
               ("avg_waiting_time", "Avg waiting time"),
               ("total_taxi_emissions", "Total taxi emissions"),
               ("emissions_per_passenger", "Emissions per passenger"),
               ("shared_distance_pct", "Distance driven shared (%)"),
               ("satisfied_pct", "Satisfied passengers (%)")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        finals = [results[l][1][col] for l in labels]
        ax.bar(x, [f.mean() for f in finals], yerr=[f.std(ddof=0) for f in finals],
               capsize=4, color=COLORS[:len(labels)])
        ax.set_xticks(x, short_labels(labels), fontsize=7)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Configurations (mean +- std over {args.seeds} seeds, {args.passengers} "
                 f"passengers, {args.taxis} taxis, boarding {args.boarding_time})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sweep(table, xlabel, path):
    """One line per key configuration: mean over seeds, band = +- std."""
    panels = [("emissions_per_passenger", "Emissions per passenger"),
              ("avg_travel_time", "Average travel time"),
              ("total_taxi_emissions", "Total taxi emissions")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for label, color in SWEEP_KEYS.items():
        stats = table[table["config"] == label].groupby("value")
        for ax, (col, title) in zip(axes, panels):
            mean, std = stats[col].mean(), stats[col].std(ddof=0)
            ax.plot(mean.index, mean, "o-", color=color, label=label)
            ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary(results, args, path):
    labels = list(results)
    short = short_labels(labels)
    colors = COLORS[:len(labels)]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def bars(ax, col, title):
        finals = [results[l][1][col] for l in labels]
        ax.bar(x, [f.mean() for f in finals], yerr=[f.std(ddof=0) for f in finals],
               capsize=4, color=colors)
        ax.set_xticks(x, short, fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")

    bars(axes[0, 0], "emissions_per_passenger", "Emissions per arrived passenger")
    bars(axes[0, 1], "shared_distance_pct", "Distance driven with >= 2 passengers (%)")
    bars(axes[0, 2], "satisfied_pct", "Satisfied passengers (%)  [acceptability]")
    bars(axes[1, 2], "makespan", "Time to serve all passengers")

    # passenger journey breakdown: wait + walk + (boarding & ride)
    ax, bottom = axes[1, 0], np.zeros(len(labels))
    wait = np.array([results[l][1]["avg_waiting_time"].mean() for l in labels])
    walk = np.array([results[l][1]["avg_walking_time"].mean() for l in labels])
    travel = np.array([results[l][1]["avg_travel_time"].mean() for l in labels])
    for vals, name in [(wait, "waiting"), (walk, "walking"), (travel - wait - walk, "boarding + riding")]:
        ax.bar(x, vals, bottom=bottom, label=name)
        bottom += vals
    ax.set_xticks(x, short, fontsize=8)
    ax.set_title("Average passenger travel time (breakdown)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # emissions by source
    ax, bottom = axes[1, 1], np.zeros(len(labels))
    for col, name in [("emissions_distance", "distance"), ("emissions_boarding", "boarding"),
                      ("emissions_congestion", "congestion")]:
        vals = np.array([results[l][1][col].mean() for l in labels])
        ax.bar(x, vals, bottom=bottom, label=name)
        bottom += vals
    ax.set_xticks(x, short, fontsize=8)
    ax.set_title("Total taxi emissions by source")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Final values (mean +- std over {args.seeds} seed(s), {args.passengers} "
                 f"passengers, {args.taxis} taxis, boarding {args.boarding_time})", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    configs = build_configs(args)
    if args.compare:
        print(f"Comparison study: {len(configs)} configurations x {args.seeds} seeds, "
              f"then boarding-time and passenger sweeps")

    results = {}
    for label, policy in configs.items():
        print(f"Running: {label}")
        results[label] = run_config(label, policy, args)

    summary = pd.DataFrame({l: fin[SUMMARY_COLS].mean() for l, (_, fin) in results.items()})
    print("\nFinal metrics (mean over seeds)")
    print(summary.round(2).to_string())
    summary.to_csv(os.path.join(args.out, "summary.csv"))

    ts = pd.concat({l: pd.concat(s).groupby(level=0).mean() for l, (s, _) in results.items()},
                   names=["config"])
    ts.to_csv(os.path.join(args.out, "timeseries.csv"))

    out = lambda name: os.path.join(args.out, name)
    if not args.compare:                       # one configuration
        if not args.no_plot:
            plot_timeseries(results, args, out("timeseries.png"))
            plot_summary(results, args, out("summary.png"))
        print(f"\nResults written to {os.path.abspath(args.out)}/")
        return

    # comparison study: configuration figures, then the two sweeps
    if not args.no_plot:
        plot_configs(results, args, out("fig_configs.png"))
        plot_summary(results, args, out("fig_breakdown.png"))
        plot_timeseries(results, args, out("fig_timeseries.png"))
    for param, values, xlabel, name in [
            ("boarding_time", BOARDING_VALUES, "Boarding time (time units)", "boarding"),
            ("passengers", PASSENGER_VALUES, "Number of passengers", "passengers")]:
        print()
        table = run_sweep(param, values, args)
        table.to_csv(out(f"sweep_{name}.csv"), index=False)
        if not args.no_plot:
            plot_sweep(table, xlabel, out(f"fig_{name}.png"))
    print(f"\nResults written to {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
