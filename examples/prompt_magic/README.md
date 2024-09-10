# Prompt Refining

Prompt refining is supported in DiffSynth-Studio.

### Example: Qwen

If you are not native English user, we provide LLM-based translation for you. Qwen is a typical example. See [`qwen_prompt_refining.py`](./qwen_prompt_refining.py).

Prompt: "鹰". This prompt will be refined:

* A majestic eagle soaring high above a vast expanse of open sky, its wings spread wide and its eyes fixed on the horizon.
* A majestic eagle soaring high above the horizon, its wingspan stretching out towards the endless sky. Its sharp beak and powerful talons stand out against the azure sky, highlighting its strength and freedom. The eagle's silhouette is silhouetted against the fading sun, casting long shadows behind it.
* A majestic eagle soaring high above a vast, open landscape, its wings spread wide and its beak pointed towards the sky. The sun casts long shadows across the ground, adding depth and texture to the image. The bird's feathers shimmer in the light, creating a sense of movement and power.
* A majestic eagle soaring high above a vast, open landscape, its sharp talons gripping a fish effortlessly in its beak. The sun casts a warm golden glow behind it, casting long shadows across the barren earth below. The eagle's wingspan stretches out towards infinity, its feathers glistening in the light. Its eyes fixate on the distant horizon, as if sensing something important about to unfold.

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![0](https://github.com/user-attachments/assets/6f509b0d-204c-4ca9-b3f6-04925fb4b40d)|![1](https://github.com/user-attachments/assets/824f2713-134a-4cae-a155-25224d4afb9a)|![2](https://github.com/user-attachments/assets/747be52a-0b46-45fc-b0e8-a0c83f7e716b)|![3](https://github.com/user-attachments/assets/490564e6-d247-40c9-8361-5db72058c337)|

### Example: OPUS + BeautifulPrompt

Our prompter can translate other language to English and refine it using "BeautifulPrompt" models. Please see [`bf_prompt_refining.py`](./bf_prompt_refining.py) for more details.

Prompt: "一个漂亮的女孩". The [translation model](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) will translate it to English.

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![0_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/ebb25ca8-7ce1-4d9e-8081-59a867c70c4d)|![1_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/a7e79853-3c1a-471a-9c58-c209ec4b76dd)|![2_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/a292b959-a121-481f-b79c-61cc3346f810)|![3_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1c19b54e-5a6f-4d48-960b-a7b2b149bb4c)|

Prompt: "一个漂亮的女孩". The [translation model](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) will translate it to English. Then the [refining model](https://huggingface.co/alibaba-pai/pai-bloom-1b1-text2prompt-sd) will refine the translated prompt for better visual quality.

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![0](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/778b1bd9-44e0-46ac-a99c-712b3fc9aaa4)|![1](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/c03479b8-2082-4c6e-8e1c-3582b98686f6)|![2](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/edb33d21-3288-4a55-96ca-a4bfe1b50b00)|![3](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/7848cfc1-cad5-4848-8373-41d24e98e584)|
