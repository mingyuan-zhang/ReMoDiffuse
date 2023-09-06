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
  <a href="https://arxiv.org/abs/2304.01116" target='_blank'>[arXiv]</a> •
  <a href="https://youtu.be/wSddrIA_2p8" target='_blank'>[Video]</a> <br> <br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=mingyuan-zhang/ReMoDiffuse" width="8%" alt="visitor badge"/>
</h4>

</div>


>**Abstract:** 3D human motion generation is crucial for creative industry. Recent advances rely on generative models with domain knowledge for text-driven motion generation, leading to substantial progress in capturing common motions. However, the performance on more diverse motions remains unsatisfactory. In this work, we propose **ReMoDiffuse**, a diffusion-model-based motion generation framework that integrates a retrieval mechanism to refine the denoising process.

<div align="center">
<tr>
    <img src="imgs/teaser.png" width="90%"/>
    <img src="imgs/pipeline.png" width="90%"/>
</tr>
</div>

>**Pipeline Overview:** ReMoDiffuse is a retrieval-augmented 3D human motion diffusion model. Benefiting from the extra knowledge from the retrieved samples, ReMoDiffuse is able to achieve high-fidelity on the given prompts. It contains three core components: a) **Hybrid Retrieval** database stores multi-modality features of each motion sequence. b) Semantics-modulated transformer incorporates several identical decoder layers, including a **Semantics-Modulated Attention (SMA)** layer and an FFN layer. The SMA layer will adaptively absorb knowledge from both retrived samples and the given prompts. c) **Condition Mxture** technique is proposed to better mix model's outputs under different combinations of conditions.

## Updates

[09/2023] Release code for [ReMoDiffuse](https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html) and [MotionDiffuse](https://mingyuan-zhang.github.io/projects/MotionDiffuse.html)


## Citation

If you find our work useful for your research, please consider citing the paper:

```
@article{zhang2023remodiffuse,
  title={ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model},
  author={Zhang, Mingyuan and Guo, Xinying and Pan, Liang and Cai, Zhongang and Hong, Fangzhou and Li, Huirong and Yang, Lei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2304.01116},
  year={2023}
}
@article{zhang2022motiondiffuse,
  title={MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model},
  author={Zhang, Mingyuan and Cai, Zhongang and Pan, Liang and Hong, Fangzhou and Guo, Xinying and Yang, Lei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2208.15001},
  year={2022}
}
```

## Installation

```shell
# Create Conda Environment
conda create -n mogen python=3.9 -y
conda activate mogen

# C++ Environment
export PATH=/mnt/lustre/share/gcc/gcc-8.5.0/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gcc-8.5.0/lib:/mnt/lustre/share/gcc/gcc-8.5.0/lib64:/mnt/lustre/share/gcc/gmp-4.3.2/lib:/mnt/lustre/share/gcc/mpc-0.8.1/lib:/mnt/lustre/share/gcc/mpfr-2.4.2/lib:$LD_LIBRARY_PATH

# Install Pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

# Install MMCV
pip install "mmcv-full>=1.4.2,<=1.9.0" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.1/index.html

# Install Pytorch3d
conda install -c bottler nvidiacub -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install pytorch3d -c pytorch3d -y

# Install other requirements
pip install -r requirements.txt
```

## Data Preparation

Download data files from google drive [link](https://drive.google.com/drive/folders/13kwahiktQ2GMVKfVH3WT-VGAQ6JHbvUv?usp=sharing) or Baidu Netdisk [link](https://pan.baidu.com/s/1604jks-9PtBUtqCpQQmeEg)(access code: vprc). Unzipped all files and arrange them in the following file structure:

```text
ReMoDiffuse
├── mogen
├── tools
├── configs
├── logs
│   ├── motiondiffuse
│   └── remodiffuse
└── data
    ├── database
    ├── datasets
    ├── evaluators
    └── glove
```

## Training

### Training with a single / multiple GPUs

```shell
PYTHONPATH=".":$PYTHONPATH python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --no-validate
```

**Note:** The provided config files are designed for training with 8 gpus. If you want to train on a single gpu, you can reduce the number of epochs to one-fourth of the original.

### Training with Slurm

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --no-validate
```

Common optional arguments include:
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--no-validate`: Whether not to evaluate the checkpoint during training.

Example: using 8 GPUs to train ReMoDiffuse on a slurm cluster.
```shell
./tools/slurm_train.sh my_partition my_job configs/remodiffuse/remodiffuse_kit.py logs/remodiffuse_kit 8 --no-validate
```

## Evaluation

### Evaluate with a single GPU / multiple GPUs

```shell
PYTHONPATH=".":$PYTHONPATH python tools/test.py ${CONFIG} --work-dir=${WORK_DIR} ${CHECKPOINT}
```

### Evaluate with slurm

```shell
./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} ${CHECKPOINT}
```
Example:
```shell
./tools/slurm_test.sh my_partition test_remodiffuse configs/remodiffuse/remodiffuse_kit.py logs/remodiffuse_kit logs/remodiffuse_kit/latest.pth
```

**Note:** Run full evaluation for HumanML3D dataset is very slow. You can change `replication_times` in [human_ml3d_bs128.py](configs/_base_/datasets/human_ml3d_bs128.py) to $1$ for a quick evaluation.

## Visualization

```shell
PYTHONPATH=".":$PYTHONPATH python tools/visualize.py ${CONFIG} ${CHECKPOINT} \
    --text ${TEXT} \
    --motion_length ${MOTION_LENGTH} \
    --out ${OUTPUT_ANIMATION_PATH} \
    --device cpu
```

Example:
```shell
PYTHONPATH=".":$PYTHONPATH python tools/visualize.py \
    configs/remodiffuse/remodiffuse_t2m.py \
    logs/remodiffuse/remodiffuse_t2m/latest.pth \
    --text "a person is running quickly" \
    --motion_length 120 \
    --out "test.gif" \
    --device cpu
```

## Acknowledgement

This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP20221-0012), NTU NAP, and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

The visualization tool is developed on top of [Generating Diverse and Natural 3D Human Motions from Text](https://github.com/EricGuo5513/text-to-motion)
