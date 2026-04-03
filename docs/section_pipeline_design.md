# Section 解析流程改单一实现入口设计（去抽象层）

## 背景

上一版实现引入了协议抽象层（7 个 Protocol）与实现层分离。当前需求已明确：

- 不需要抽象协议层。
- 只需要一个 section 解析方法作为业务入口。
- 仍需保留核心流程能力：分类后流式拆 item、item 级异步处理、最终统一聚合。

## 澄清结果

1. 删除抽象协议层。
2. 仅保留实现层代码。
3. 统一入口使用一个方法：`parse_section(...)`。
4. 入口方法放在 `llm_core/section_pipeline_impl.py`。
5. 功能语义保持不变：
   - 先分类，再流式拆 item
   - item 一旦完整立即异步处理
   - 等全部 item 完成后再聚合

## WHAT（做什么）

1. 重构 `llm_core/section_pipeline_impl.py`
   - 去掉对 `section_pipeline_protocols.py` 的依赖
   - 去掉对 `section_pipeline_types.py` 的依赖
   - 将必要数据结构内聚到该文件
   - 提供单一入口 `parse_section(...)`

2. 删除不再需要的抽象文件
   - `llm_core/section_pipeline_protocols.py`
   - `llm_core/section_pipeline_types.py`

3. 更新对外导出
   - `llm_core/__init__.py` 移除抽象层导出
   - 仅导出实现层可用对象与入口

4. 更新 demo
   - 演示直接调用单一入口方法

## WHY（为什么）

1. 降低理解成本：业务方只需要一个入口，不需要掌握协议体系。
2. 降低维护成本：减少跨文件抽象跳转。
3. 保持效果不变：去抽象不去能力，仍保持流式 + 并发 + 最终聚合。

## HOW（怎么做）

### 1) 实现层内聚

在 `section_pipeline_impl.py` 内定义：

- `SectionContext`
- `SectionItem`
- `ItemProcessResult`
- `StructuredNode`
- `JsonlIncrementalItemParser`
- `AsyncioItemTaskScheduler`
- `DefaultItemProcessor`
- `DefaultStructuredNodeAggregator`
- `OpenAISectionItemStreamer`
- `parse_section(...)`（唯一流程入口）

### 2) 单一入口方法

`parse_section(...)` 负责完整编排：

1. 执行 section 分类
2. 启动流式 item 拆分
3. 每个完整 item 立即调度处理
4. 等待全部 item 收敛
5. 统一聚合并返回 structured node

### 3) 错误处理

- 默认 `fail_fast=False`：单 item 失败不导致 section 失败。
- `fail_fast=True`：出现失败后停止新增并尽快收敛，最终抛错。

### 4) 兼容性

- 会移除协议层文件，属于结构简化。
- 对外入口改为实现层单入口，`__init__.py` 会同步导出。

## 验证方案

1. 运行编译检查。
2. 运行 demo，确认日志体现：
   - 边流式拆分，边并发处理
   - 最终统一聚合输出
