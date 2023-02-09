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

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), sharex=False, constrained_layout=True
    )
    result.plot_model(axes[0, 0])
    result.plot_model(axes[1, 0], scale="log")
    result.plot_probablity(axes[0, 1])
    axes[1, 1].axis("off")
    fig.savefig("tests/output/demo.png", dpi=300)

    log_prob_df = (
        pd.DataFrame(
            dict(
                depth=result.depths,
                error=result.error_log_probablities,
                **{
                    f"CN={x}": probablities
                    for x, probablities in result.peak_log_probablities.items()
                },
            ),
            dtype=np.float128,
        )
        .set_index("depth")
        .iloc[:2000, :]
    )
    log_prob_df.to_csv("tests/output/log_probablities.tsv", sep="\t")
