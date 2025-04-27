import torch


class Qwen25VL_7b_Embedder(torch.nn.Module):
    def __init__(self, model_path, max_length=640, dtype=torch.bfloat16, device="cuda"):
        super(Qwen25VL_7b_Embedder, self).__init__()
        self.max_length = max_length
        self.dtype = dtype
        self.device = device
        
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(torch.cuda.current_device())

        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28
        )
        
        Qwen25VL_7b_PREFIX = '''Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:'''

        self.prefix = Qwen25VL_7b_PREFIX
        
    @staticmethod
    def from_pretrained(path, torch_dtype=torch.bfloat16, device="cuda"):
        return Qwen25VL_7b_Embedder(path, dtype=torch_dtype, device=device)

    def forward(self, caption, ref_images):
        text_list = caption
        embs = torch.zeros(
            len(text_list),
            self.max_length,
            self.model.config.hidden_size,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )
        hidden_states = torch.zeros(
            len(text_list),
            self.max_length,
            self.model.config.hidden_size,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )
        masks = torch.zeros(
            len(text_list),
            self.max_length,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        input_ids_list = []
        attention_mask_list = []
        emb_list = []

        def split_string(s):
            s = s.replace("“", '"').replace("”", '"').replace("'", '''"''')  # use english quotes
            result = []
            in_quotes = False
            temp = ""

            for idx,char in enumerate(s):
                if char == '"' and idx>155:
                    temp += char
                    if not in_quotes:
                        result.append(temp)
                        temp = ""

                    in_quotes = not in_quotes
                    continue
                if in_quotes:
                    if char.isspace():
                        pass  # have space token

                    result.append("“" + char + "”")
                else:
                    temp += char

            if temp:
                result.append(temp)

            return result

        for idx, (txt, imgs) in enumerate(zip(text_list, ref_images)):

            messages = [{"role": "user", "content": []}]

            messages[0]["content"].append({"type": "text", "text": f"{self.prefix}"})

            messages[0]["content"].append({"type": "image", "image": imgs})

            # 再添加 text
            messages[0]["content"].append({"type": "text", "text": f"{txt}"})

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )

            image_inputs = [imgs]

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            old_inputs_ids = inputs.input_ids
            text_split_list = split_string(text)

            token_list = []
            for text_each in text_split_list:
                txt_inputs = self.processor(
                    text=text_each,
                    images=None,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                token_each = txt_inputs.input_ids
                if token_each[0][0] == 2073 and token_each[0][-1] == 854:
                    token_each = token_each[:, 1:-1]
                    token_list.append(token_each)
                else:
                    token_list.append(token_each)

            new_txt_ids = torch.cat(token_list, dim=1).to("cuda")

            new_txt_ids = new_txt_ids.to(old_inputs_ids.device)

            idx1 = (old_inputs_ids == 151653).nonzero(as_tuple=True)[1][0]
            idx2 = (new_txt_ids == 151653).nonzero(as_tuple=True)[1][0]
            inputs.input_ids = (
                torch.cat([old_inputs_ids[0, :idx1], new_txt_ids[0, idx2:]], dim=0)
                .unsqueeze(0)
                .to("cuda")
            )
            inputs.attention_mask = (inputs.input_ids > 0).long().to("cuda")
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values.to("cuda"),
                image_grid_thw=inputs.image_grid_thw.to("cuda"),
                output_hidden_states=True,
            )

            emb = outputs["hidden_states"][-1]

            embs[idx, : min(self.max_length, emb.shape[1] - 217)] = emb[0, 217:][
                : self.max_length
            ]

            masks[idx, : min(self.max_length, emb.shape[1] - 217)] = torch.ones(
                (min(self.max_length, emb.shape[1] - 217)),
                dtype=torch.long,
                device=torch.cuda.current_device(),
            )

        return embs, masks