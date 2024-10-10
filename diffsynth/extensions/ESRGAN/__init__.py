import torch
from einops import repeat
from PIL import Image
import numpy as np


class ResidualDenseBlock(torch.nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(torch.nn.Module):

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(torch.nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, **kwargs):
        super(RRDBNet, self).__init__()
        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = torch.torch.nn.Sequential(*[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = repeat(feat, "B C H W -> B C (H 2) (W 2)")
        feat = self.lrelu(self.conv_up1(feat))
        feat = repeat(feat, "B C H W -> B C (H 2) (W 2)")
        feat = self.lrelu(self.conv_up2(feat))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
    
    @staticmethod
    def state_dict_converter():
        return RRDBNetStateDictConverter()
    

class RRDBNetStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict, {"upcast_to_float32": True}
    
    def from_civitai(self, state_dict):
        return state_dict, {"upcast_to_float32": True}


class ESRGAN(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def from_model_manager(model_manager):
        return ESRGAN(model_manager.fetch_model("esrgan"))

    def process_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
        return image
    
    def process_images(self, images):
        images = [self.process_image(image) for image in images]
        images = torch.stack(images)
        return images
    
    def decode_images(self, images):
        images = (images.permute(0, 2, 3, 1) * 255).clip(0, 255).numpy().astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
        return images
    
    @torch.no_grad()
    def upscale(self, images, batch_size=4, progress_bar=lambda x:x):
        if not isinstance(images, list):
            images = [images]
            is_single_image = True
        else:
            is_single_image = False

        # Preprocess
        input_tensor = self.process_images(images)

        # Interpolate
        output_tensor = []
        for batch_id in progress_bar(range(0, input_tensor.shape[0], batch_size)):
            batch_id_ = min(batch_id + batch_size, input_tensor.shape[0])
            batch_input_tensor = input_tensor[batch_id: batch_id_]
            batch_input_tensor = batch_input_tensor.to(
                device=self.model.conv_first.weight.device,
                dtype=self.model.conv_first.weight.dtype)
            batch_output_tensor = self.model(batch_input_tensor)
            output_tensor.append(batch_output_tensor.cpu())
        
        # Output
        output_tensor = torch.concat(output_tensor, dim=0)

        # To images
        output_images = self.decode_images(output_tensor)
        if is_single_image:
            output_images = output_images[0]
        return output_images
