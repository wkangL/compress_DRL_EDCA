# 核心对比

入口：`python run.py --config configs/quick.yaml`。正式参数：`configs/full.yaml`。

六组方法均在相同ns-3后端、业务定义、Profile表和reward下运行。除Default外各方法独立训练，不共用某一方法的checkpoint。统计同时覆盖全部STA及上下行。
