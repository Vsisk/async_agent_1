# Section 单入口重构开发计划（去抽象层）

关联设计文档：`docs/section_pipeline_design.md`

## Phase #1 设计落盘与任务重排

- Task #1：将设计文档改为“单入口实现层”方案
  - 预期结果：文档不再包含抽象协议层目标
  - 状态：Finished

- Task #2：重写开发计划
  - 预期结果：任务围绕“删除抽象层 + 单入口实现”
  - 状态：Finished

## Phase #2 实现层重构

- Task #1：将必要领域数据结构内聚到 `section_pipeline_impl.py`
  - 预期结果：不再依赖 `section_pipeline_types.py`
  - 状态：Finished

- Task #2：移除协议层依赖并删除抽象协议文件
  - 预期结果：不再依赖 `section_pipeline_protocols.py`
  - 状态：Finished

- Task #3：实现/保留单入口方法 `parse_section(...)`
  - 预期结果：可一站式完成分类、流式拆分、并发处理、聚合
  - 状态：Finished

## Phase #3 对外接口与示例更新

- Task #1：更新 `llm_core/__init__.py` 导出
  - 预期结果：只暴露实现层可用对象
  - 状态：Finished

- Task #2：更新 demo 为单入口调用方式
  - 预期结果：示例不再展示抽象协议用法
  - 状态：Finished

## Phase #4 验证与收尾

- Task #1：执行可审计验证命令
  - 预期结果：编译通过、demo 可运行
  - 状态：Finished

- Task #2：完成分阶段 git 提交
  - 预期结果：提交记录清晰
  - 状态：Finished

- Task #3：追加 `progress.md` 进度
  - 预期结果：仅追加，不修改历史
  - 状态：Finished

- Task #4：调用 make_pr
  - 预期结果：PR 信息完整
  - 状态：Finished
