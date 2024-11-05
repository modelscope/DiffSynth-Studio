# 欢迎来到 Diffusion 的魔法世界

欢迎来到 Diffusion 的魔法世界，这里是 DiffSynth-Studio，一个开源的 Diffusion 引擎，我们希望通过这样一个开源项目，构建统一、互联、创新的 Diffusion 模型生态！

## 统一

目前的开源 Diffusion 模型结构五花八门，以文生图模型为例，有 Stable Diffusion、Kolors、FLUX 等。

|    FLUX    | Stable Diffusion 3 | Kolors | Hunyuan-DiT | Stable Diffusion | Stable Diffusion XL |
|-|-|-|-|-|-|
| <img src="https://github.com/user-attachments/assets/984561e9-553d-4952-9443-79ce144f379f" width="80" height="80"/> | <img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098" width="80" height="80"/> | <img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/53ef6f41-da11-4701-8665-9f64392607bf" width="80" height="80"/> | <img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/60b022c8-df3f-4541-95ab-bf39f2fa8bb5" width="80" height="80"/> | <img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5" width="80" height="80"/> | <img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90" width="80" height="80"/> |

<table>
  <tr>
    <th>FLUX</th>
    <th>Stable Diffusion 3</th>
    <th>Kolors</th>
    <th>Hunyuan-DiT</th>
    <th>Stable Diffusion</th>
    <th>Stable Diffusion XL</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/984561e9-553d-4952-9443-79ce144f379f" width="100"/></td>
    <td><img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098" width="100"/></td>
    <td><img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/53ef6f41-da11-4701-8665-9f64392607bf" width="100"/></td>
    <td><img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/60b022c8-df3f-4541-95ab-bf39f2fa8bb5" width="100"/></td>
    <td><img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5" width="100"/></td>
    <td><img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90" width="100"/></td>
  </tr>
</table>

   <table>
     <thead>
       <tr>
         <th style="width:15%">FLUX</th>
         <th style="width:15%">Stable Diffusion 3</th>
         <th style="width:15%">Kolors</th>
         <th style="width:15%">Hunyuan-DiT</th>
         <th style="width:15%">Stable Diffusion</th>
         <th style="width:25%">Stable Diffusion XL</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <td><img src="https://github.com/user-attachments/assets/984561e9-553d-4952-9443-79ce144f379f" alt="image_1024_cfg" /></td>
         <td><img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098" alt="image_1024" /></td>
         <td><img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/53ef6f41-da11-4701-8665-9f64392607bf" alt="image_1024" /></td>
         <td><img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/60b022c8-df3f-4541-95ab-bf39f2fa8bb5" alt="image_1024" /></td>
         <td><img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5" alt="1024" /></td>
         <td><img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90" alt="1024" /></td>
       </tr>
     </tbody>
   </table>


<style>
  table {
    width: 100%;
    table-layout: fixed; /* 表格布局设置为固定，以便可以设置列宽 */
  }

  th, td {
    width: 16.6%; /* 每列的宽度，大约为 100/6 */
    text-align: center;
  }

  /* 具体设置每一列的宽度（如果每列需要不同的宽度，可以分别设置） */
  th:nth-child(1),
  td:nth-child(1) {
    width: 15%;
  }
  
  th:nth-child(2),
  td:nth-child(2) {
    width: 15%;
  }
  
  th:nth-child(3),
  td:nth-child(3) {
    width: 15%;
  }
  
  th:nth-child(4),
  td:nth-child(4) {
    width: 15%;
  }
  
  th:nth-child(5),
  td:nth-child(5) {
    width: 20%;
  }
  
  th:nth-child(6),
  td:nth-child(6) {
    width: 20%;
  }
</style>

<table>
  <thead>
    <tr>
      <th>FLUX</th>
      <th>Stable Diffusion 3</th>
      <th>Kolors</th>
      <th>Hunyuan-DiT</th>
      <th>Stable Diffusion</th>
      <th>Stable Diffusion XL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/984561e9-553d-4952-9443-79ce144f379f" alt="image_1024_cfg"></td>
      <td><img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098" alt="image_1024"></td>
      <td><img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/53ef6f41-da11-4701-8665-9f64392607bf" alt="image_1024"></td>
      <td><img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/60b022c8-df3f-4541-95ab-bf39f2fa8bb5" alt="image_1024"></td>
      <td><img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5" alt="1024"></td>
      <td><img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90" alt="1024"></td>
    </tr>
  </tbody>
</table>



我们设计了统一的框架，实现了通用的增强模块，例如提示词分区控制技术。

<div align="center">
<video width="512" height="256" controls>
  <source src="https://github.com/user-attachments/assets/59613157-de51-4109-99b3-97cbffd88076" type="video/mp4">
您的浏览器不支持Video标签。
</video>
</div>

以及一站式的训练脚本。

||FLUX.1-dev|Kolors|Stable Diffusion 3|Hunyuan-DiT|
|-|-|-|-|-|
|Without LoRA|<img src="https://github.com/user-attachments/assets/df62cef6-d54f-4e3d-a602-5dd290079d49" width="100"  alt="image_without_lora">|<img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/9d79ed7a-e8cf-4d98-800a-f182809db318" width="100"  alt="image_without_lora">|<img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/ddb834a5-6366-412b-93dc-6d957230d66e" width="100"  alt="image_without_lora">|<img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1aa21de5-a992-4b66-b14f-caa44e08876e" width="100"  alt="image_without_lora">|
|With LoRA|<img src="https://github.com/user-attachments/assets/4fd39890-0291-4d19-8a88-d70d0ae18533" width="100"  alt="image_with_lora">|<img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/02f62323-6ee5-4788-97a1-549732dbe4f0" width="100"  alt="image_with_lora">|<img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/8e7b2888-d874-4da4-a75b-11b6b214b9bf" width="100"  alt="image_with_lora">|<img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/83a0a41a-691f-4610-8e7b-d8e17c50a282" width="100"  alt="image_with_lora">|


## 互联

与语言模型不同，Diffusion 模型存在生态模型，包括 LoRA、ControlNet、IP-Adapter 等，这些模型由不同的开发者开发、训练、开源，我们为这些模型提供了一站式的推理支持。例如基于 Stable Diffusion XL，你可以随意使用这些相关的生态模型组装出丰富的功能。

|底模生成|使用 ControlNet 保持画面结构重新生成|继续叠加 LoRA 使画面更扁平|叠加 IP-Adapter 转换为水墨风格|
|-|-|-|-|
|<img src="https://github.com/user-attachments/assets/cc094e8f-ff6a-4f9e-ba05-7a5c2e0e609f" width="100" >|<img src="https://github.com/user-attachments/assets/d50d173e-e81a-4d7e-93e3-b2787d69953e" width="100" >|<img src="https://github.com/user-attachments/assets/c599b2f8-8351-4be5-a6ae-8380889cb9d8" width="100" >|<img src="https://github.com/user-attachments/assets/e5924aef-03b0-4462-811f-a60e2523fd7f" width="100" >|


你甚至可以继续叠加 AnimateDiff 构建视频转绘方案。

<div align="center">
<video width="512" height="256" controls>
  <source src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd" type="video/mp4">
您的浏览器不支持Video标签。
</video>
</div>

## 创新

DiffSynth-Studio 集成了多个开源模型，这是属于开源社区的奇迹。我们致力于用强工程基础驱动算法上的创新，目前我们公开了多项创新性生成技术。

* ExVideo: 视频生成模型的扩展训练技术
    * 项目页面: [https://ecnu-cilab.github.io/ExVideoProjectPage/](https://ecnu-cilab.github.io/ExVideoProjectPage/)
    * 技术报告: [https://arxiv.org/abs/2406.14130](https://arxiv.org/abs/2406.14130)
    * 模型 (ExVideo-CogVideoX)
        * HuggingFace: [https://huggingface.co/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1](https://huggingface.co/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1)
        * ModelScope: [https://modelscope.cn/models/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1](https://modelscope.cn/models/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1)
    * 模型 (ExVideo-SVD)
        * HuggingFace: [https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)
        * ModelScope: [https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)
* Diffutoon: 动漫风格视频渲染方案
    * 项目页面: [https://ecnu-cilab.github.io/DiffutoonProjectPage/](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
    * 技术报告: [https://arxiv.org/abs/2401.16224](https://arxiv.org/abs/2401.16224)
    * 样例代码: [https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/Diffutoon](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/Diffutoon)
* FastBlend: 视频去闪烁算法
    * 独立仓库: [https://github.com/Artiprocher/sd-webui-fastblend](https://github.com/Artiprocher/sd-webui-fastblend)
    * 视频演示
        * [https://www.bilibili.com/video/BV1d94y1W7PE](https://www.bilibili.com/video/BV1d94y1W7PE)
        * [https://www.bilibili.com/video/BV1Lw411m71p](https://www.bilibili.com/video/BV1Lw411m71p)
        * [https://www.bilibili.com/video/BV1RB4y1Z7LF](https://www.bilibili.com/video/BV1RB4y1Z7LF)
    * 技术报告: [https://arxiv.org/abs/2311.09265](https://arxiv.org/abs/2311.09265)
* DiffSynth: DiffSynth-Studio 的前身
    * 项目页面: [https://ecnu-cilab.github.io/DiffSynth.github.io/](https://ecnu-cilab.github.io/DiffSynth.github.io/)
    * 早期代码: [https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth)
    * 技术报告: [https://arxiv.org/abs/2308.03463](https://arxiv.org/abs/2308.03463)
