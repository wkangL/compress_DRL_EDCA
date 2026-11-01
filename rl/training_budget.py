"""Check that configured collection and exploration budgets can reach their goals."""
import math


def training_budget(config: dict) -> dict:
    t=config["training"]
    ratio=t["episode_s"]/config["timing"]["decision_s"]
    if ratio<1 or not math.isclose(ratio,round(ratio),abs_tol=1e-9):
        raise ValueError("Training episode_s must be a positive integer multiple of decision_s")
    per_stage=round(ratio)*t["episodes_per_stage"]
    steps=per_stage*t["stages"]
    threshold=max(t["warmup"],t["batch_size"])
    if per_stage<=0 or t["stages"]<=0 or t["epsilon_steps"]<=0 or t["updates_per_stage"]<=0:
        raise ValueError("Training and exploration budgets must be positive")
    if threshold>min(steps,t["replay_capacity"]):
        raise ValueError("Training budget/replay capacity cannot reach warmup and batch size")
    if config["run"]["mode"]=="full" and t["epsilon_steps"]>steps:
        raise ValueError("Full training ends before epsilon decay completes; correct the recorded budget first")
    first=math.ceil(threshold/per_stage)
    return dict(total_steps=steps,steps_per_stage=per_stage,first_update_stage=first,
                total_updates=(t["stages"]-first+1)*t["updates_per_stage"],
                final_action_epsilon=t["epsilon_end"]+(t["epsilon_start"]-t["epsilon_end"])*max(0,1-(steps-1)/t["epsilon_steps"]))
