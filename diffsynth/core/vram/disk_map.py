from safetensors import safe_open
import torch, os


class SafetensorsCompatibleTensor:
    def __init__(self, tensor):
        self.tensor = tensor
    
    def get_shape(self):
        return list(self.tensor.shape)


class SafetensorsCompatibleBinaryLoader:
    def __init__(self, path, device):
        print("Detected non-safetensors files, which may cause slower loading. It's recommended to convert it to a safetensors file.")
        self.state_dict = torch.load(path, weights_only=True, map_location=device)
        
    def keys(self):
        return self.state_dict.keys()
    
    def get_tensor(self, name):
        return self.state_dict[name]
    
    def get_slice(self, name):
        return SafetensorsCompatibleTensor(self.state_dict[name])


class DiskMap:

    def __init__(self, path, device, torch_dtype=None, state_dict_converter=None, buffer_size=10**9):
        self.path = path if isinstance(path, list) else [path]
        self.device = device
        self.torch_dtype = torch_dtype
        if os.environ.get('DIFFSYNTH_DISK_MAP_BUFFER_SIZE') is not None:
            self.buffer_size = int(os.environ.get('DIFFSYNTH_DISK_MAP_BUFFER_SIZE'))
        else:
            self.buffer_size = buffer_size
        self.files = []
        self.flush_files()
        self.name_map = {}
        for file_id, file in enumerate(self.files):
            for name in file.keys():
                self.name_map[name] = file_id
        self.rename_dict = self.fetch_rename_dict(state_dict_converter)
        
    def flush_files(self):
        if len(self.files) == 0:
            for path in self.path:
                if path.endswith(".safetensors"):
                    self.files.append(safe_open(path, framework="pt", device=str(self.device)))
                else:
                    self.files.append(SafetensorsCompatibleBinaryLoader(path, device=self.device))
        else:
            for i, path in enumerate(self.path):
                if path.endswith(".safetensors"):
                    self.files[i] = safe_open(path, framework="pt", device=str(self.device))
        self.num_params = 0

    def __getitem__(self, name):
        if self.rename_dict is not None: name = self.rename_dict[name]
        file_id = self.name_map[name]
        param = self.files[file_id].get_tensor(name)
        if self.torch_dtype is not None and isinstance(param, torch.Tensor):
            param = param.to(self.torch_dtype)
        if isinstance(param, torch.Tensor) and param.device == "cpu":
            param = param.clone()
        if isinstance(param, torch.Tensor):
            self.num_params += param.numel()
        if self.num_params > self.buffer_size:
            self.flush_files()
        return param

    def fetch_rename_dict(self, state_dict_converter):
        if state_dict_converter is None:
            return None
        state_dict = {}
        for file in self.files:
            for name in file.keys():
                state_dict[name] = name
        state_dict = state_dict_converter(state_dict)
        return state_dict
    
    def __iter__(self):
        if self.rename_dict is not None:
            return self.rename_dict.__iter__()
        else:
            return self.name_map.__iter__()
    
    def __contains__(self, x):
        if self.rename_dict is not None:
            return x in self.rename_dict
        else:
            return x in self.name_map
