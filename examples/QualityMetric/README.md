# Image Quality Metric

The image quality assessment functionality has now been integrated into Diffsynth.

## Usage

### Step 1: Download pretrained reward models 

```
modelscope download --model 'DiffSynth-Studio/QualityMetric_reward_pretrained' 
```

The file directory is shown below.

```
DiffSynth-Studio/
└── models/
    └── QualityMetric/
        ├── HPS_v2/
        │   ├── HPS_v2_compressed.safetensors
        │   ├── HPS_v2.1_compressed.safetensors
        └── ...
```

### Step 2: Test image quality metric

Prompt: "a painting of an ocean with clouds and birds, day time, low depth field effect"

|1.webp|2.webp|3.webp|4.webp|
|-|-|-|-|
|![0](images/1.webp)|![1](images/2.webp)|![2](images/3.webp)|![3](images/4.webp)|



```
CUDA_VISIBLE_DEVICES=0 python testreward.py
```

### Output:

```
ImageReward: [0.5811904668807983, 0.2745198607444763, -1.4158903360366821, -2.032487154006958]
Aesthetic [5.900862693786621, 5.776571273803711, 5.799864292144775, 5.05204963684082]
PickScore: [0.20737126469612122, 0.20443597435951233, 0.20660750567913055, 0.19426065683364868]
CLIPScore: [0.3894640803337097, 0.3544551134109497, 0.33861416578292847, 0.32878392934799194]
HPScorev2: [0.2672519087791443, 0.25495243072509766, 0.24888549745082855, 0.24302822351455688]
HPScorev21: [0.2321144938468933, 0.20233657956123352, 0.1978294551372528, 0.19230154156684875]
MPS_score: [10.921875, 10.71875, 10.578125, 9.25]
```
