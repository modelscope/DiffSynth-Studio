# 可控生成、Template 与 Adapter 研究

适用于控制、编辑、个性化、LoRA组合、Image-to-LoRA与可训练Template。先检查具体模型的条件Unit/model_fn、训练和验证示例。关键来源：`diffsynth/diffusion/{template,training_module,base_pipeline,loss}.py`、`docs/zh/Diffusion_Templates/` 与 `docs/zh/Research_Tutorial/controllable_models.md`。

## 选择控制机制与对照

| 研究机制 | 已有入口举例 | 主要实验与评价 |
| --- | --- | --- |
| 空间结构、区域与局部编辑 | ControlNet、Flux/Qwen的EliGen、inpaint、布局条件 | 控制精度、区域泄漏、编辑保真、条件强度/缺失/冲突 |
| 连续属性 | FLUX.1 AttriCtrl 控制模型；FLUX.2 Brightness/SoftRGB Diffusion Template | 分别核对加载、控制接口和训练路径；扫描控制值、单调性、校准误差、内容保持、属性解耦 |
| 参考与视觉类比 | ContentRef、`Qwen-Image-Edit-2511-ICEdit.py` | 参考数量/顺序/相关性、示范变换泛化、跨主体与分布外条件 |
| 参数高效学习 | 目标模型LoRA/full、差分LoRA、Timestep-LoRA | rank/目标层/组件/时间门控、参数与训练预算、泛化及遗忘 |
| 组合与检索 | Flux LoRA-Fusion、Flux2 TreeAdapter | 静态/门控/层级组合、冲突、权重与顺序、长尾分类泛化 |
| 权重生成/元学习 | KleinBase4B/ZImage/HidreamO1的i2L示例 | 参考集合大小、风格与内容解耦、生成权重rank、跨任务迁移 |

这些机制不都使用Template。复用真实目标入口，不能仅把条件重命名就宣称跨模型兼容。研究参考图时保持目标/来源及顺序明确，训练/验证按主体、风格或原始数据来源隔离。

AttriCtrl 示例 `examples/flux/model_inference/FLUX.1-dev-AttriCtrl.py` 使用 `FluxImagePipeline`、额外控制模型配置与 `value_controller_inputs`；FLUX.2 的 `examples/flux2/model_inference/Template-KleinBase4B-Brightness.py` 使用 `TemplatePipeline` 与 `template_inputs`。前者不属于 Diffusion Template，不能套用下文的 Template 加载或训练契约。

LoRA研究核对实际命中的层、可训练参数量、初始化与缩放，不能只比较rank数字。蒸馏模型微调后少步能力可能退化；差分训练和再蒸馏是不同路线，应分别比较。查 `docs/zh/Training/Differential_LoRA.md` 和目标 `special/differential_training/`：二阶段先融合LoRA1再训练LoRA2，评价时声明底模关系和实际加载组合，不称其为DPO。

推理组合核对 `utils/lora/{merge,reset_rank}.py`、目标loader/patcher。A/B因子的拼接可表示增量相加；对不同训练结果分别平均A和B并不等于平均完整增量，rank压缩也可能有损。控制随机分解的seed，检查兼容键/shape、缩放、压缩误差和最终生成。

i2L/TreeAdapter的实际示例包括 `examples/flux2/model_inference/{KleinBase4B-i2L-v2,TreeAdapter-KleinBase4B}.py`。KleinBase4B-i2L-v2有相应 `model_training/full/` 训练配方；当前TreeAdapter同名shell只有注释，说明系统仅支持推理并指向Aesthetic模板的训练示例，不能当成TreeAdapter系统训练入口。读取模型代码及真实训练契约后再决定复用或项目实现，不由文件存在推导支持。生成LoRA的forward可以作为权重生成研究基础，不等于普通逐样本LoRA训练。

## Template 推理与组合契约

正确导入：

```python
from diffsynth.diffusion.template import TemplatePipeline
```

从真实样例构建base pipe和template的model_configs后，调用 `template(pipe=pipe, template_inputs=[...], negative_template_inputs=[...], **base_inputs)`。这不是完整脚本，交付时补齐真实加载与保存。项目入口将 `model_id` 校验为model_configs范围内的非负整数，默认0；它不是Hub模型ID。当前fetch_model直接索引列表，不能依赖框架拒绝负值或替项目校验；上层加载流程也未保证非法值有一致报错。各模板字段由model.py定义，`scale`不是所有模板共有参数。

当前 `call_single_side` 用model_id路由，调用 `process_inputs(pipe=pipe, **input_dict)` 后再 `forward(pipe=pipe, **cache)`；model_id不是自动传入forward的保证字段。`pipe`保留给框架注入，model_id保留作路由，业务输入不要占用它们。

缓存只注入base `__call__` 中明确声明的字段；即使base接收 `**kwargs` 也不保证通过该检查。负缓存必须有对应的 `negative_` 字段。继续追踪签名 → Unit → model_fn → 模型内部，验证字段未被覆盖/忽略及实际控制效果。

| 输出缓存字段 | 当前合并规则 |
| --- | --- |
| `kv_cache` | 同block的K/V按dim=1拼接，校验batch/head/dim及block名称 |
| `lora` | 使用merge_lora拼接因子；需兼容目标键/shape，核验生成权重及rank成本 |
| `text_embedding` | 按dim=-2拼接 |
| 其他同名字段 | 输出冲突提示，只保留第一项 |

Flux2实际输入是 `extra_text_embedding`，不等于合并器特判的 `text_embedding`；不能把任何embedding都当成自动拼接。Qwen-Image-2.1等还有不同命名如 `template_kv_cache`，旧模板不会自动完成字段和shape适配。组合检查顺序、各正负条件及残差/LoRA/缓存冲突，必要时在项目显式适配并做单模板/组合消融。

TemplatePipeline不实现普通VRAM管理，ModelConfig里的offload等配置会被忽略。lazy_loading按需构造模板，不是逐层offload；base资源策略另配。输出LoRA的模板先按样例对base启用hot loading，检查正负分支切换、重复调用与清理；不要混入不可卸载的冷融合权重后仍把clear_lora当作基线恢复。

## 自定义可训练 Template

`model.py` 由loader通过 `exec_module()` 实际执行，不是只读权重；外部模板先审查来源和代码，不能因已选定模型ID就跳过代码检查。`TEMPLATE_MODEL` 指向模型类；`TEMPLATE_MODEL_PATH=None`走随机初始化/无权重模块分支，预训练文件名及配置按loader实际支持核对。

`process_inputs`、`forward`都需存在，且当前格式检查要求参数名为 `kwargs` 的 `**kwargs`。预处理函数显式写 `@torch.no_grad()`；训练加载链不会统一替任意override加装饰器。forward承担需训练的计算，底模冻结仍要保留对cache的梯度，不调用带no_grad的完整推理入口替代训练路径。

当前训练加载器直接调用 `load_template_data_processor(path)()`，因此此训练链的model.py应定义可实例化的 `TEMPLATE_DATA_PROCESSOR`；即使预计算输入也提供透传processor。推理TemplatePipeline不使用训练processor，训练/推理输入可以不同，但需转换到相同模型语义。例如：

```python
class ScaleProcessor:
    def __call__(self, scale, **kwargs):
        return {"scale": float(scale)}

TEMPLATE_DATA_PROCESSOR = ScaleProcessor
```

这是需放进自定义model.py的processor片段，不是完整模板。若改用原版brightness DataAnnotator，训练输入须给image路径，它会直接Image.open，且不会自动拼dataset_base_path。预计算scale时须确实替换processor；嵌套媒体不由顶层普通image operator自动打开。

当前Flux2 metadata中的 `template_inputs` 是dict，推理才是list。训练源头 `examples/flux2/model_training/scripts/brightness/model.py` 和相邻训练/验证shell；按其真实路径/语义适配：

- `--template_model_id_or_path` 本地目录，或当前解析器要求的以冒号结尾的Hub配置。
- `--extra_inputs template_inputs`；编辑任务还需核对edit_image的加载和传递。
- `--trainable_models template_model`，部分组件可指定子模块路径。
- `--remove_prefix_in_ckpt pipe.template_model.`，明确checkpoint键空间。

声明输入、cache字段、shape和dtype；训练cache应以有梯度的路径到达loss。检查Unit消费/覆盖、shared/正负字段归属及拆分缓存保留，尤其非kv_cache输出；按需读 [扩展点](extension-points.md)。现有Flux2训练支持不能推导所有Pipeline都能直接训练Template。

## 保存与重新使用

做有限前后向并检查预期参数更新，再保存、新进程加载，对固定条件生成真实输出。验证控制值扫描和无模板/单模板/组合对照，不只检查“没有报错”。

`TEMPLATE_MODEL_PATH=None`分支忽略传入的state_dict。可在eager构造后对 `template.models[0]` 严格load_state_dict加载完整权重，或导出完整模型目录并写入实际权重路径。lazy_loading会重新构造，不可只更新临时fetch实例。完整脚本交付代码、processor、结构配置、权重引用和加载方式。

部分参数训练时，框架导出只含可训练项：从原始冻结权重得到完整state，再用训练项覆盖，核对键与shape，保存合并结果；方向不能反过来覆盖训练结果。只在验证缺失项确实是原冻结项时使用非严格加载，并报告unexpected/missing，不能吞掉任意不匹配。
