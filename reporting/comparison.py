"""Measured comparisons and exact empirical delay survival probabilities."""
from pathlib import Path
import numpy as np

ACS=("vo","vi","be","bk")


def relative_change(baseline: float, candidate: float) -> float|None:
    """Signed percent change; zero baseline has no defined percentage."""
    if not np.isfinite([baseline,candidate]).all() or baseline==0: return None
    return (candidate-baseline)/abs(baseline)*100


def empirical_ccdf(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """P(delay > x), with all ties handled at once."""
    samples=np.sort(np.asarray(samples,dtype=float))
    if not len(samples): return np.full(len(grid),np.nan)
    if not np.isfinite(samples).all() or (samples<0).any(): raise ValueError("Invalid packet delay")
    return (len(samples)-np.searchsorted(samples,grid,side="right"))/len(samples)


def load_delays(episode: Path) -> dict[str,np.ndarray]:
    packed=episode/"delay_samples.npz"
    if packed.exists():
        with np.load(packed,allow_pickle=False) as source:
            return {ac:np.asarray(source[ac],dtype=float) for ac in ACS}
    samples={ac:[] for ac in ACS}
    with (episode/"events.log").open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("D "):
                fields=line.split()
                samples[ACS[int(fields[3])]].append(float(fields[5]))
    return {ac:np.asarray(values,dtype=float) for ac,values in samples.items()}


def balanced_ccdf(replicates: dict, grid: np.ndarray) -> np.ndarray:
    """Equal eval-seed weight within each train seed, then equal train-seed weight.

    Empty successful-packet distributions are undefined and excluded explicitly by
    the plotting caller, which records their absence and sample counts.
    """
    means=[]
    for evaluations in replicates.values():
        curves=[empirical_ccdf(samples,grid) for samples in evaluations.values() if len(samples)]
        if curves: means.append(np.mean(curves,axis=0))
    return np.mean(means,axis=0) if means else np.full(len(grid),np.nan)
