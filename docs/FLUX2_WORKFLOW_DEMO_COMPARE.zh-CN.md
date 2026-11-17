# FLUX.2 工作流与当前demo复现差异

2026-11-14，只读核对当前object-inpainting-workflow (11)new.json、demo_flux3_recolor_pipeline.py和节点实现。未运行GPU、未改用户正在编辑的demo或训练数据。历史运行时节点版本/环境变量未核实。

| 项 | 当前workflow中的随机换色节点 | 当前磁盘demo的FLUX2调用 |
| --- | --- | --- |
| 指令 | 节点长prompt，插入choose_recolor选择的color_name | 短prompt，不含目标颜色名称 |
| 步数 | 8 | 未显式传，当前本地Flux2KleinPipeline默认50 |
| CFG | 4 | 未显式传，默认4 |
| seed | -1，执行时secrets生成actual_seed | 固定43 |
| 色块 | 本次随机挑选palette的128×128 sampled_color | 从test/主颜色B读32×32再放大128 |
| 结果处理 | pad到16倍数→生成→crop→Image.composite恢复mask外原像素 | 直接保存pipeline原输出 |

当前demo已被用户改为原图A/涂灰图/主颜色B，之前交接所记28步/CFG3.5不再代表当前文件。未改回或覆盖用户编辑。当前FLUX3调用也没有显式步数/CFG，不能凭注释判断两模型采样配置相同。

workflow的autocolorregions_2从随机换色后的source估计颜色，colorregionswatches_2输出的是区域平均RGB的32×32图；与randomrecolor最初送给模型的sampled_color不同。拿生成结果的估计颜色重新输入，不等于重放原始生成条件。当前样本主颜色B为[243,243,247]，不能从这个值唯一恢复palette颜色名及actual_seed。

工作流目前保存img4连到fluxkleinreference的平铺图输出，并未直接连接fluxrandomrecolor_2.source；img5是生成结果提取色块。因此这份当前JSON本身不能完整证明00019_00的生成图B当时如何保存/改名。该PNG没有metadata，test下未找到JSON记录；不据此断言历史运行配置与当前文件完全相同。

重放某张历史图需要：当次原图及相同mask、original sampled_color及color_name、actual_seed、generation_prompt、8步/CFG4、基础模型及相关库版本，再执行相同crop/composite。默认节点模型为FLUX.2-klein-base-9B，但运行时OBJECT_INPAINTING_FLUX_MODEL_DIR可覆盖，历史值未知。现有数据不足以保证逐像素复现。保持用户“颜色仅由色块提供”的新约束时，应把它视为新实验，不能把含颜色词的旧工作流结果当成同条件基准。


后续按用户授权已将demo改为工作流对齐模式（以上表格描述修改前版本）：同节点长prompt+color_name、8步、CFG4、seed=-1解析并记录，颜色块由同节点选色函数生成，同pad/crop/composite。实际3568041805/sky blue双模型均生成天蓝衣服，结果见outputs/flux_workflow_aligned_demo/3568041805；不是历史白衣逐像素复现，也不代表仅色块模式成功。
