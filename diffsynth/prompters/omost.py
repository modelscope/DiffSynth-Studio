from transformers import AutoTokenizer, TextIteratorStreamer
import difflib
import torch
import numpy as np
import re
from ..models.model_manager import ModelManager
from PIL import Image

valid_colors = {  # r, g, b
    'aliceblue': (240, 248, 255), 'antiquewhite': (250, 235, 215), 'aqua': (0, 255, 255),
    'aquamarine': (127, 255, 212), 'azure': (240, 255, 255), 'beige': (245, 245, 220),
    'bisque': (255, 228, 196), 'black': (0, 0, 0), 'blanchedalmond': (255, 235, 205), 'blue': (0, 0, 255),
    'blueviolet': (138, 43, 226), 'brown': (165, 42, 42), 'burlywood': (222, 184, 135),
    'cadetblue': (95, 158, 160), 'chartreuse': (127, 255, 0), 'chocolate': (210, 105, 30),
    'coral': (255, 127, 80), 'cornflowerblue': (100, 149, 237), 'cornsilk': (255, 248, 220),
    'crimson': (220, 20, 60), 'cyan': (0, 255, 255), 'darkblue': (0, 0, 139), 'darkcyan': (0, 139, 139),
    'darkgoldenrod': (184, 134, 11), 'darkgray': (169, 169, 169), 'darkgrey': (169, 169, 169),
    'darkgreen': (0, 100, 0), 'darkkhaki': (189, 183, 107), 'darkmagenta': (139, 0, 139),
    'darkolivegreen': (85, 107, 47), 'darkorange': (255, 140, 0), 'darkorchid': (153, 50, 204),
    'darkred': (139, 0, 0), 'darksalmon': (233, 150, 122), 'darkseagreen': (143, 188, 143),
    'darkslateblue': (72, 61, 139), 'darkslategray': (47, 79, 79), 'darkslategrey': (47, 79, 79),
    'darkturquoise': (0, 206, 209), 'darkviolet': (148, 0, 211), 'deeppink': (255, 20, 147),
    'deepskyblue': (0, 191, 255), 'dimgray': (105, 105, 105), 'dimgrey': (105, 105, 105),
    'dodgerblue': (30, 144, 255), 'firebrick': (178, 34, 34), 'floralwhite': (255, 250, 240),
    'forestgreen': (34, 139, 34), 'fuchsia': (255, 0, 255), 'gainsboro': (220, 220, 220),
    'ghostwhite': (248, 248, 255), 'gold': (255, 215, 0), 'goldenrod': (218, 165, 32),
    'gray': (128, 128, 128), 'grey': (128, 128, 128), 'green': (0, 128, 0), 'greenyellow': (173, 255, 47),
    'honeydew': (240, 255, 240), 'hotpink': (255, 105, 180), 'indianred': (205, 92, 92),
    'indigo': (75, 0, 130), 'ivory': (255, 255, 240), 'khaki': (240, 230, 140), 'lavender': (230, 230, 250),
    'lavenderblush': (255, 240, 245), 'lawngreen': (124, 252, 0), 'lemonchiffon': (255, 250, 205),
    'lightblue': (173, 216, 230), 'lightcoral': (240, 128, 128), 'lightcyan': (224, 255, 255),
    'lightgoldenrodyellow': (250, 250, 210), 'lightgray': (211, 211, 211), 'lightgrey': (211, 211, 211),
    'lightgreen': (144, 238, 144), 'lightpink': (255, 182, 193), 'lightsalmon': (255, 160, 122),
    'lightseagreen': (32, 178, 170), 'lightskyblue': (135, 206, 250), 'lightslategray': (119, 136, 153),
    'lightslategrey': (119, 136, 153), 'lightsteelblue': (176, 196, 222), 'lightyellow': (255, 255, 224),
    'lime': (0, 255, 0), 'limegreen': (50, 205, 50), 'linen': (250, 240, 230), 'magenta': (255, 0, 255),
    'maroon': (128, 0, 0), 'mediumaquamarine': (102, 205, 170), 'mediumblue': (0, 0, 205),
    'mediumorchid': (186, 85, 211), 'mediumpurple': (147, 112, 219), 'mediumseagreen': (60, 179, 113),
    'mediumslateblue': (123, 104, 238), 'mediumspringgreen': (0, 250, 154),
    'mediumturquoise': (72, 209, 204), 'mediumvioletred': (199, 21, 133), 'midnightblue': (25, 25, 112),
    'mintcream': (245, 255, 250), 'mistyrose': (255, 228, 225), 'moccasin': (255, 228, 181),
    'navajowhite': (255, 222, 173), 'navy': (0, 0, 128), 'navyblue': (0, 0, 128),
    'oldlace': (253, 245, 230), 'olive': (128, 128, 0), 'olivedrab': (107, 142, 35),
    'orange': (255, 165, 0), 'orangered': (255, 69, 0), 'orchid': (218, 112, 214),
    'palegoldenrod': (238, 232, 170), 'palegreen': (152, 251, 152), 'paleturquoise': (175, 238, 238),
    'palevioletred': (219, 112, 147), 'papayawhip': (255, 239, 213), 'peachpuff': (255, 218, 185),
    'peru': (205, 133, 63), 'pink': (255, 192, 203), 'plum': (221, 160, 221), 'powderblue': (176, 224, 230),
    'purple': (128, 0, 128), 'rebeccapurple': (102, 51, 153), 'red': (255, 0, 0),
    'rosybrown': (188, 143, 143), 'royalblue': (65, 105, 225), 'saddlebrown': (139, 69, 19),
    'salmon': (250, 128, 114), 'sandybrown': (244, 164, 96), 'seagreen': (46, 139, 87),
    'seashell': (255, 245, 238), 'sienna': (160, 82, 45), 'silver': (192, 192, 192),
    'skyblue': (135, 206, 235), 'slateblue': (106, 90, 205), 'slategray': (112, 128, 144),
    'slategrey': (112, 128, 144), 'snow': (255, 250, 250), 'springgreen': (0, 255, 127),
    'steelblue': (70, 130, 180), 'tan': (210, 180, 140), 'teal': (0, 128, 128), 'thistle': (216, 191, 216),
    'tomato': (255, 99, 71), 'turquoise': (64, 224, 208), 'violet': (238, 130, 238),
    'wheat': (245, 222, 179), 'white': (255, 255, 255), 'whitesmoke': (245, 245, 245),
    'yellow': (255, 255, 0), 'yellowgreen': (154, 205, 50)
}

valid_locations = {  # x, y in 90*90
    'in the center': (45, 45),
    'on the left': (15, 45),
    'on the right': (75, 45),
    'on the top': (45, 15),
    'on the bottom': (45, 75),
    'on the top-left': (15, 15),
    'on the top-right': (75, 15),
    'on the bottom-left': (15, 75),
    'on the bottom-right': (75, 75)
}

valid_offsets = {  # x, y in 90*90
    'no offset': (0, 0),
    'slightly to the left': (-10, 0),
    'slightly to the right': (10, 0),
    'slightly to the upper': (0, -10),
    'slightly to the lower': (0, 10),
    'slightly to the upper-left': (-10, -10),
    'slightly to the upper-right': (10, -10),
    'slightly to the lower-left': (-10, 10),
    'slightly to the lower-right': (10, 10)}

valid_areas = {  # w, h in 90*90
    "a small square area": (50, 50),
    "a small vertical area": (40, 60),
    "a small horizontal area": (60, 40),
    "a medium-sized square area": (60, 60),
    "a medium-sized vertical area": (50, 80),
    "a medium-sized horizontal area": (80, 50),
    "a large square area": (70, 70),
    "a large vertical area": (60, 90),
    "a large horizontal area": (90, 60)
}

def safe_str(x):
    return x.strip(',. ') + '.'

def closest_name(input_str, options):
    input_str = input_str.lower()

    closest_match = difflib.get_close_matches(input_str, list(options.keys()), n=1, cutoff=0.5)
    assert isinstance(closest_match, list) and len(closest_match) > 0, f'The value [{input_str}] is not valid!'
    result = closest_match[0]

    if result != input_str:
        print(f'Automatically corrected [{input_str}] -> [{result}].')

    return result

class Canvas:
    @staticmethod
    def from_bot_response(response: str):
        
        matched = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        assert matched, 'Response does not contain codes!'
        code_content = matched.group(1)
        assert 'canvas = Canvas()' in code_content, 'Code block must include valid canvas var!'
        local_vars = {'Canvas': Canvas}
        exec(code_content, {}, local_vars)
        canvas = local_vars.get('canvas', None)
        assert isinstance(canvas, Canvas), 'Code block must produce valid canvas var!'
        return canvas

    def __init__(self):
        self.components = []
        self.color = None
        self.record_tags = True
        self.prefixes = []
        self.suffixes = []
        return

    def set_global_description(self, description: str, detailed_descriptions: list, tags: str,
                               HTML_web_color_name: str):
        assert isinstance(description, str), 'Global description is not valid!'
        assert isinstance(detailed_descriptions, list) and all(isinstance(item, str) for item in detailed_descriptions), \
            'Global detailed_descriptions is not valid!'
        assert isinstance(tags, str), 'Global tags is not valid!'

        HTML_web_color_name = closest_name(HTML_web_color_name, valid_colors)
        self.color = np.array([[valid_colors[HTML_web_color_name]]], dtype=np.uint8)

        self.prefixes = [description]
        self.suffixes = detailed_descriptions

        if self.record_tags:
            self.suffixes = self.suffixes + [tags]

        self.prefixes = [safe_str(x) for x in self.prefixes]
        self.suffixes = [safe_str(x) for x in self.suffixes]

        return

    def add_local_description(self, location: str, offset: str, area: str, distance_to_viewer: float, description: str,
                              detailed_descriptions: list, tags: str, atmosphere: str, style: str,
                              quality_meta: str, HTML_web_color_name: str):
        assert isinstance(description, str), 'Local description is wrong!'
        assert isinstance(distance_to_viewer, (int, float)) and distance_to_viewer > 0, \
            f'The distance_to_viewer for [{description}] is not positive float number!'
        assert isinstance(detailed_descriptions, list) and all(isinstance(item, str) for item in detailed_descriptions), \
            f'The detailed_descriptions for [{description}] is not valid!'
        assert isinstance(tags, str), f'The tags for [{description}] is not valid!'
        assert isinstance(atmosphere, str), f'The atmosphere for [{description}] is not valid!'
        assert isinstance(style, str), f'The style for [{description}] is not valid!'
        assert isinstance(quality_meta, str), f'The quality_meta for [{description}] is not valid!'

        location = closest_name(location, valid_locations)
        offset = closest_name(offset, valid_offsets)
        area = closest_name(area, valid_areas)
        HTML_web_color_name = closest_name(HTML_web_color_name, valid_colors)

        xb, yb = valid_locations[location]
        xo, yo = valid_offsets[offset]
        w, h = valid_areas[area]
        rect = (yb + yo - h // 2, yb + yo + h // 2, xb + xo - w // 2, xb + xo + w // 2)
        rect = [max(0, min(90, i)) for i in rect]
        color = np.array([[valid_colors[HTML_web_color_name]]], dtype=np.uint8)

        prefixes = self.prefixes + [description]
        suffixes = detailed_descriptions

        if self.record_tags:
            suffixes = suffixes + [tags, atmosphere, style, quality_meta]

        prefixes = [safe_str(x) for x in prefixes]
        suffixes = [safe_str(x) for x in suffixes]

        self.components.append(dict(
            rect=rect,
            distance_to_viewer=distance_to_viewer,
            color=color,
            prefixes=prefixes,
            suffixes=suffixes,
            location=location,
        ))

        return

    def process(self):
        # sort components
        self.components = sorted(self.components, key=lambda x: x['distance_to_viewer'], reverse=True)

        # compute initial latent
        # print(self.color)
        initial_latent = np.zeros(shape=(90, 90, 3), dtype=np.float32) + self.color

        for component in self.components:
            a, b, c, d = component['rect']
            initial_latent[a:b, c:d] = 0.7 * component['color'] + 0.3 * initial_latent[a:b, c:d]

        initial_latent = initial_latent.clip(0, 255).astype(np.uint8)

        # compute conditions

        bag_of_conditions = [
            dict(mask=np.ones(shape=(90, 90), dtype=np.float32), prefixes=self.prefixes, suffixes=self.suffixes,location= "full")
        ]

        for i, component in enumerate(self.components):
            a, b, c, d = component['rect']
            m = np.zeros(shape=(90, 90), dtype=np.float32)
            m[a:b, c:d] = 1.0
            bag_of_conditions.append(dict(
                mask = m,
                prefixes = component['prefixes'],
                suffixes = component['suffixes'],
                location = component['location'],
            ))
            
        return dict(
            initial_latent = initial_latent,
            bag_of_conditions = bag_of_conditions,
        )


class OmostPromter(torch.nn.Module):
    
    def __init__(self,model = None,tokenizer = None, template = "",device="cpu"):
        super().__init__()
        self.model=model
        self.tokenizer = tokenizer
        self.device = device
        if template == "":
            template = r'''You are a helpful AI assistant to compose images using the below python class `Canvas`:
            ```python
            class Canvas:
                def set_global_description(self, description: str, detailed_descriptions: list[str], tags: str, HTML_web_color_name: str):
                    pass

                def add_local_description(self, location: str, offset: str, area: str, distance_to_viewer: float, description: str, detailed_descriptions: list[str], tags: str, atmosphere: str, style: str, quality_meta: str, HTML_web_color_name: str):
                    assert location in ["in the center", "on the left", "on the right", "on the top", "on the bottom", "on the top-left", "on the top-right", "on the bottom-left", "on the bottom-right"]
                    assert offset in ["no offset", "slightly to the left", "slightly to the right", "slightly to the upper", "slightly to the lower", "slightly to the upper-left", "slightly to the upper-right", "slightly to the lower-left", "slightly to the lower-right"]
                    assert area in ["a small square area", "a small vertical area", "a small horizontal area", "a medium-sized square area", "a medium-sized vertical area", "a medium-sized horizontal area", "a large square area", "a large vertical area", "a large horizontal area"]
                    assert distance_to_viewer > 0
                    pass
            ```'''
        self.template = template
        
    @staticmethod
    def from_model_manager(model_manager: ModelManager):
        model, model_path = model_manager.fetch_model("omost_prompt", require_model_path=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        omost = OmostPromter(
            model=  model,
            tokenizer = tokenizer,
            device = model_manager.device
        )
        return omost
    
    
    def __call__(self,prompt_dict:dict):
        raw_prompt=prompt_dict["prompt"]
        conversation = [{"role": "system", "content": self.template}]
        conversation.append({"role": "user", "content": raw_prompt})
        
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.bfloat16, device=self.device)
        
        generate_kwargs = dict(
            input_ids = input_ids,
            streamer = streamer,
            # stopping_criteria=stopping_criteria,
            # max_new_tokens=max_new_tokens,
            do_sample = True,
            attention_mask = attention_mask,
            pad_token_id = self.tokenizer.eos_token_id,
            # temperature=temperature,
            # top_p=top_p,
        )
        self.model.generate(**generate_kwargs)
        outputs = []
        for text in streamer:
            outputs.append(text)
        llm_outputs = "".join(outputs)
        
        canvas = Canvas.from_bot_response(llm_outputs)
        canvas_output = canvas.process()
        
        prompts = [" ".join(_["prefixes"]+_["suffixes"][:2]) for _ in canvas_output["bag_of_conditions"]]
        canvas_output["prompt"] = prompts[0]
        canvas_output["prompts"] = prompts[1:]
        
        raw_masks = [_["mask"] for _ in canvas_output["bag_of_conditions"]]
        masks=[]
        for mask in raw_masks:
            mask[mask>0.5]=255
            mask = np.stack([mask] * 3, axis=-1).astype("uint8")
            masks.append(Image.fromarray(mask))
            
        canvas_output["masks"] = masks
        prompt_dict.update(canvas_output)
        print(f"Your prompt is extended by Omost:\n")
        cnt = 0
        for component,pmt in zip(canvas_output["bag_of_conditions"],prompts):
            loc = component["location"]
            cnt += 1
            print(f"Component {cnt} - Location : {loc}\nPrompt:{pmt}\n")
        
        return prompt_dict
    
    
        
    