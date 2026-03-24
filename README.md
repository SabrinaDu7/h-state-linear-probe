# PRNN Hidden State NOR Classifcation

Linear probe (binary) on PRNN hidden states: novel object in view or not.

## Data Format
Data is saved in save_path as a .pt file and a parquet file with the following format:
```python
    result: EvalTrajectories = {
        "obs": all_obs,
        "obs_pred": all_obs_pred,
        "obs_next": all_obs_next,
        "act": all_act,
        "states": all_states, # Contains agent_pos and agent_dir
        "hidden_states": all_hidden, # Tensor[B, T, H]
        "labels": all_labels, # Tensor[B, T]`` that assigns an integer label to each (trajectory, timestep) pair
        "renders": all_renders,
        "config" : config # EvalTrajectoryConfig
    }

    save_data = {k: v for k, v in results.items() if k != "renders" and v is not None}

    # Save .pt file
    torch.save(save_data, save_path / "trajectories.pt")

    # Summary parquet (tabular state data)
    states: list[State] = data["states"]
    records = []
    for b, state in enumerate(states):
        T_plus_1 = state["agent_pos"].shape[0]
        for t in range(T_plus_1):
            records.append(
                {
                    "traj_id": b,
                    "timestep": t,
                    "pos_x": float(state["agent_pos"][t, 0]),
                    "pos_y": float(state["agent_pos"][t, 1]),
                    "direction": int(state["agent_dir"][t]),
                }
            )
    pd.DataFrame(records).to_parquet(save_path / "summary.parquet", index=False)
```

## Data Pipeline
- Data is generated in ```/RL_for_pRNN/scripts/analysis_OMT.py/```.
- Data is symlink'ed to this repo ```ln -s ../RL_for_pRNN/outputs/data_cur_lroom_step1408_goal72/ ./data/```
- Each data point is (label, hidden_state) with label == 1 if the hidden state came from a view where novel object was in-view.
label == 0 if the novel object was NOT in view. 
- K-Fold Cross-Validation (k=5) on ONE TIMESTEP and ONE goal location (stratified to avoid class imbalances)
- Fold-splitting strategy is by trajectory (NOT TIMESTEP)


## Evaluation Setup


## Structure

```
src/
  config.py   # Centralized paths and constants
  probe.py    # Linear probe training with optuna
  plot.py     # Analysis plots from metrics
  metrics.py  # All functions with eval metrics are here. 
  data.py     # Inspect the data (e.g. number of samples for each class)

outputs/
  probes/     # probe_step{step_n}_goal{goal_loc}.pt - weights
  plots/      # Generated visualizations
```

## PITFALLS
- **C for logistic regression scales with dataset size** - 
