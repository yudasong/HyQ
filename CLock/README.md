# HyQ in Combination Lock
This repository contains our FQI-style algorithm for the comblock environment.

## Offline Datasets
We exploit two forms of offline dataset, optimal trajectory and optimal occupancy, corresponding to the argument <code>--offline_dataset epsilon</code> and <code>--offline_dataset mixed</code> respectively:
* Optimal Trajectory: we collect full trajectories by following $\pi^\star$ with $\epsilon$-greedy exploration with $\epsilon=1/H$. We also add some noise by making the agent to perform randomly at timestep $H/2$.
* Optimal Occupancy: we collect transition tuples from the state-occupancy measure of $\pi^\star$ with random actions. Formally, we sample $h \sim \textrm{Unif}([H])$, $s \sim d_h^{\pi^\star}$, $a \sim \textrm{Unif}(\mathcal{A})$, $r \sim R(s,a)$, $s' \sim P(s,a)$.

Please refer to our paper for more details.

## Run our code

To reproduce our result in comblock with optimal trajectory offline dataset, please run:
```bash
python online_main_lock.py --seed 12345 --exp_name h100s12345 --offline_dataset epsilon
```
To reproduce our result in comblock with optimal occupancy offline dataset, please run:
```bash
python online_main_lock.py --seed 12345 --exp_name h100s12345 --offline_dataset mixed
```
## Credit 
Some code are adapted from BRIEE https://github.com/yudasong/briee.