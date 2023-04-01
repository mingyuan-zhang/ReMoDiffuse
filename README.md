<div align="center">

<h1>ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model</h1>

<div>
    <a href='https://mingyuan-zhang.github.io/' target='_blank'>Mingyuan Zhang</a><sup>1</sup>&emsp;
    <a href='https://gxyes.github.io/' target='_blank'>Xinying Guo</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=lSDISOcAAAAJ&hl=zh-CN' target='_blank'>Liang Pan</a><sup>1</sup>&emsp;
    <a href='https://caizhongang.github.io/' target='_blank'>Zhongang Cai</a><sup>1,2</sup>&emsp;
    <a href='https://hongfz16.github.io/' target='_blank'>Fangzhou Hong</a><sup>1</sup>&emsp;
    <a href='https://www.linkedin.com/in/huirong-li' target='_blank'>Huirong Li</a><sup>1</sup>&emsp; <br>
    <a href='https://yanglei.me/' target='_blank'>Lei Yang</a><sup>2</sup>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>1+</sup>
</div>
<div>
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp;
    <sup>2</sup>SenseTime Research&emsp;
</div>
<div>
    <sup>+</sup>corresponding author
</div>


---

<h4 align="center">
  <a href="https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html" target='_blank'>[Project Page]</a> •
  <a href="https://arxiv.org/abs/2304.xxxxx" target='_blank'>[arXiv]</a> •
  <a href="https://youtu.be/NeFezKIl7GE" target='_blank'>[Video]</a>
</h4>

</div>

<div>
The code will be available soon.
</div>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=mingyuan-zhang/ReMoDiffuse)

>**Abstract:** 3D human motion generation is crucial for creative industry. Recent advances rely on generative models with domain knowledge for text-driven motion generation, leading to substantial progress in capturing common motions. However, the performance on more diverse motions remains unsatisfactory. In this work, we propose **ReMoDiffuse**, a diffusion-model-based motion generation framework that integrates a retrieval mechanism to refine the denoising process.

<div align="center">
<tr>
    <img src="imgs/teaser.png" width="90%"/>
    <img src="imgs/pipeline.png" width="90%"/>
</tr>
</div>

>**Pipeline Overview:** ReMoDiffuse is a retrieval-augmented 3D human motion diffusion model. Benefiting from the extra knowledge from the retrieved samples, ReMoDiffuse is able to achieve high-fidelity on the given prompts. It contains three core components: a) **Hybrid Retrieval** database stores multi-modality features of each motion sequence. b) Semantics-modulated transformer incorporates several identical decoder layers, including a **Semantics-Modulated Attention (SMA)** layer and an FFN layer. The SMA layer will adaptively absorb knowledge from both retrived samples and the given prompts. c) **Condition Mxture** technique is proposed to better mix model's outputs under different combinations of conditions.