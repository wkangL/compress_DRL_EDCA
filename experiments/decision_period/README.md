# 决策周期

先使用同一checkpoint分别以1、5、10s运行action hold；采样保持100ms、历史窗口保持1s。记录zero-shot，不得将重新训练或fine-tune结果混入同一组。

使用 `experiments/run_ablations.py --kind decision_period --checkpoint PATH --config configs/quick.yaml`。
