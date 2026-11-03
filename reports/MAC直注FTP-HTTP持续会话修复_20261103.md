# MAC直注FTP/HTTP持续会话修复

日期：2026-11-03

## 问题

`direct_high_watermark=0`表示外生业务到达后立即交给ns-3原生`WifiMacQueue`，不运行应用侧准入轮询。原实现却把FTP文件完成后的阅读间隔和HTTP页面完成后的下一阅读间隔统一放在`AdmissionTick()`中安排。零水位模式不调用该函数，因此FTP/HTTP只生成启动阶段的首次文件或页面，后续会话停止。

这会让40秒评估窗口中的BE/BK新增负载异常偏低，并使低优先级Profile的评价失真。历史实验结果和checkpoint保留不变，但其BE/BK结论应标注这一限制。

## 修复

- 新增`ScheduleNextBulkSession()`，只负责安排FTP/HTTP下一次独立会话。
- 历史非零水位模式仍在pending队列排空后调用该函数，原有语义不变。
- 零水位即时入队模式在FTP文件或HTTP页面的全部MSDU同步提交给原生MAC队列后直接调用该函数。
- 没有恢复轮询，没有引入64包水位，也没有修改包长、文件/页面分布、阅读间隔、EDCA、Reward或网络结构。

## 影响

修复只影响后续新运行。此前平衡Profile 400步实验的VO/VI结果仍可读取，但其BE/BK负载与饥饿结论不能作为修复后持续FTP/HTTP场景的基线。使用该场景继续研究前，至少需要重新运行Default与final checkpoint配对评估；若业务负载和状态分布变化明显，正式结论还应重新训练。

## 验证

- C++后端重新编译成功。
- 1 AP + 1 STA、零水位、Yans PHY、完整Trace的HTTP 100 s冒烟测试共记录5次HTTP页面生成，时间为0.249、25.519、26.138、89.429和96.778 s；不再停在首次页面。
- 相同最小环境的FTP 600 s测试共记录3次文件生成，时间为0.016、327.778和477.644 s；指数阅读间隔继续生效。
- 回归测试：`113 passed, 7 skipped`。
- 验证原始数据：`results/validation/mac_bulk_session_reschedule_20261103/`。
