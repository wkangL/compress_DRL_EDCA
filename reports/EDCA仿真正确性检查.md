# EDCA仿真正确性检查

运行目录：`D:\compress_DRL_EDCA\results\raw\edca_validation_20261021_091205`

训练门禁：通过。

## 上游协议测试

- wifi-devices-dcf: PASS，{'passed': True, 'returncode': 0}
- wifi-devices-edca: PASS，{'passed': True, 'returncode': 0}
- wifi-channel-access-manager: PASS，{'passed': True, 'returncode': 0}
- wifi-mac-queue: PASS，{'passed': True, 'returncode': 0}
- wifi-txop-nonht: PASS，{'passed': True, 'returncode': 0}

## 场景检查

| 检查 | 理论预期 | 实测 | 状态 |
|---|---|---|---|
|单STA低负载|实际重叠接收失败接近0|0.0|PASS|
|STA数量增加|固定每STA offered load，竞争失败总体增加|[0.0029739978339013914, 0.015823167374894906, 0.07628585568127669]|PASS|
|AC优先级|VO/VI平均时延低于BE/BK|[21.3503036890557, 26.01184898949465, 1904.0737281565657, 3981.37311545]|PASS|
|CWmin增加|失败比例下降且平均等待增加|timeout 0.0364945313661386/0.02197184749701908; delay 0.5187160515365337/1.9702736399117977 ms|PASS|
|RTS/CTS帧|trace实际连续产生RTS→CTS→DATA→ACK|平均完整序列数 1338.0|PASS|
|隐藏节点恢复|隐藏节点降低吞吐，RTS/CTS部分恢复|20.234250000000003/10.9965/19.275750000000002 Mbps|PASS|
|BEB退化|相同种子和负载，custom beta=2 gamma=7近似标准BEB|20.234250000000003/20.234250000000003 Mbps|PASS|
|碰撞独立归因|预期接收端真实RxDrop且有重叠可达信号；timeout单列|碰撞 1114.5; timeout 1114.5; 低负载其他失败 0.0|PASS|

碰撞定义为预期接收端DATA/RTS实际接收失败，且同一接收时间有另一个可达发送信号。分母为DATA/RTS尝试数。ACK/CTS timeout单列。该归因依赖本场景固定矩阵链路，尚不支持任意功率捕获归因。

完整上游 wifi-txop 在 HE 子用例发生越界（见首轮记录），未宣称修复。wifi-txop-nonht 使用原测试实现的三个802.11a用例，未修改断言；仅适用于当前无聚合802.11a场景，不代表HE通过。
