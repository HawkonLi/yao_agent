# 多智能体编排

源代码：[example_group.py](https://github.com/HawkonLi/yao_agent/blob/main/example_group.py)

完整的多智能体协作场景：用 `SessionGroup` 把四个智能体编排成：

1. **Parallel 调研** — 宏观 + 微观两个研究员并行调研同一主题
2. **综述** — 合成研究员读黑板上的发现，产出综述
3. **Loop 自我精修** — 修订员迭代改进直到输出包含 `[OK]`

所有智能体通过共享 `Notebook`（EnvironmentObject）通信。

```bash
python3 example_group.py
```
