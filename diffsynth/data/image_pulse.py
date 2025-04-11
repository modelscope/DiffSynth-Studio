import torch, os, json, torchvision
from PIL import Image
from torchvision.transforms import v2



class SingleTaskDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, keys=(("image_1", "image_2", "editing_instruction"), ("image_2", "image_1", "reverse_editing_instruction")), height=1024, width=1024, random=True, steps_per_epoch=1000, metadata_path=None):
        self.base_path = base_path
        self.keys = keys
        self.metadata = []
        self.bad_data = []
        self.height = height
        self.width = width
        self.random = random
        self.steps_per_epoch = steps_per_epoch
        self.image_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        if metadata_path is None:
            self.search_for_data("", report_data_log=True)
            self.report_data_log()
        else:
            with open(metadata_path, "r", encoding="utf-8-sig") as f:
                self.metadata = json.load(f)


    def report_data_log(self):
        print(f"{len(self.metadata)} valid data, {len(self.bad_data)} invalid data.")


    def dump_metadata(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)
        
        
    def parse_json_file(self, absolute_path, relative_path):
        data_list = []
        with open(absolute_path, "r") as f:
            metadata = json.load(f)
            for image_1, image_2, instruction in self.keys:
                image_1 = os.path.join(relative_path, metadata[image_1])
                image_2 = os.path.join(relative_path, metadata[image_2])
                instruction = metadata[instruction]
                data_list.append((image_1, image_2, instruction))
        return data_list
    
        
    def search_for_data(self, path, report_data_log=False):
        now_path = os.path.join(self.base_path, path)
        if os.path.isfile(now_path) and path.endswith(".json"):
            try:
                data_list = self.parse_json_file(now_path, os.path.dirname(path))
                self.metadata.extend(data_list)
            except:
                self.bad_data.append(now_path)
        elif os.path.isdir(now_path):
            for sub_path in os.listdir(now_path):
                self.search_for_data(os.path.join(path, sub_path))
                if report_data_log and os.path.isdir(os.path.join(self.base_path, path, sub_path)):
                    self.report_data_log()
                
                
    def load_image(self, image_path):
        image_path = os.path.join(self.base_path, image_path)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = self.image_process(image)
        return image
                
                
    def load_data(self, data_id):
        image_1, image_2, instruction = self.metadata[data_id]
        image_1 = self.load_image(image_1)
        image_2 = self.load_image(image_2)
        return {"image_1": image_1, "image_2": image_2, "instruction": instruction}
        
        
    def __getitem__(self, data_id):
        if self.random:
            while True:
                try:
                    data_id = (torch.randint(0, len(self.metadata), (1,))[0] + data_id) % len(self.metadata)
                    data = self.load_data(data_id)
                    return data
                except:
                    continue
        else:
            return self.load_data(data_id)


    def __len__(self):
        return self.steps_per_epoch if self.random else len(self.metadata)
    
    
    
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list, dataset_weight, steps_per_epoch=1000):
        self.dataset_list = dataset_list
        self.dataset_weight = torch.tensor(dataset_weight, dtype=torch.float)
        self.steps_per_epoch = steps_per_epoch

        
    def __getitem__(self, data_id):
        while True:
            try:
                dataset_id = torch.multinomial(self.dataset_weight, 1).tolist()[0]
                data_id = torch.randint(0, len(self.dataset_list[dataset_id]), (1,))[0]
                data = self.dataset_list[dataset_id][data_id]
                return data
            except:
                continue


    def __len__(self):
        return self.steps_per_epoch
