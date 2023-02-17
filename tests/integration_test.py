import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["figure.dpi"] = 72
from kmer_profile_decomposer import KmerProfileModel


def test_demo():
    df = pd.read_table("tests/data/kmer_profile.tsv", names=("depth", "count"))
    model = KmerProfileModel(df["depth"], df["count"], 12, 15, 2000)
    result = model.fit(107)
    assert 100 <= result.parameters.haploid_depth <= 120
    assert 0.01 <= result.parameters.peak_weights[0] <= 0.06
    assert 0.3 <= result.parameters.peak_weights[1] <= 0.7

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, constrained_layout=True)
    result.plot_model(axes[0, 0], scale='linear')
    result.plot_model(axes[1, 0], scale='log')
    result.plot_probablity(axes[0, 1], scale='linear')
    result.plot_probablity(axes[1, 1], scale='log')
    fig.savefig("tests/output/demo.png", dpi=300)

    prob_df = result.get_log_probablity_dataframe()
    prob_df.to_csv("tests/output/log_probablities.tsv", sep="\t")
