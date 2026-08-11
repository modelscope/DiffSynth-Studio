from einops import repeat, rearrange
from tqdm import tqdm
import torch


class TileWorker:
    def __init__(self):
        pass

    def build_mask(self, data, is_bound):
        H, W = data.shape[:2]
        h = repeat(torch.arange(H), "H -> H W", H=H, W=W)
        w = repeat(torch.arange(W), "W -> H W", H=H, W=W)
        border_width = (H + W) // 4
        pad = torch.ones_like(h) * border_width
        mask = torch.stack([
            pad if is_bound[0] else h + 1,
            pad if is_bound[1] else H - h,
            pad if is_bound[2] else w + 1,
            pad if is_bound[3] else W - w
        ]).min(dim=0).values
        mask = mask.clip(1, border_width)
        mask = (mask / border_width).to(dtype=data.dtype, device=data.device)
        mask = rearrange(mask, "H W -> H W 1")
        return mask

    def tiled_forward(self, forward_fn, channels, tile_size, tile_stride, tile_range, output_scale=1, device="cpu", dtype=torch.float32, border_width=None, progress_bar=tqdm):
        # Prepare
        H, W = tile_range
        border_width = int(tile_stride*0.5) if border_width is None else border_width
        weight = torch.zeros((H, W, 1), dtype=dtype, device=device)
        values = torch.zeros((H, W, channels), dtype=dtype, device=device)

        # Split tasks
        tasks = []
        for h in range(0, H, tile_stride):
            for w in range(0, W, tile_stride):
                if (h-tile_stride >= 0 and h-tile_stride+tile_size >= H) or (w-tile_stride >= 0 and w-tile_stride+tile_size >= W):
                    continue
                h_, w_ = h + tile_size, w + tile_size
                if h_ > H: h, h_ = H - tile_size, H
                if w_ > W: w, w_ = W - tile_size, W
                tasks.append((h, h_, w, w_))
        
        # Run
        for hl, hr, wl, wr in progress_bar(tasks):
            # Forward
            x = forward_fn(hl, hr, wl, wr).to(dtype=dtype, device=device)
            mask = self.build_mask(x, is_bound=(hl==0, hr>=H, wl==0, wr>=W))
            hl, hr = int(hl * output_scale), int(hr * output_scale)
            wl, wr = int(wl * output_scale), int(wr * output_scale)
            values[hl:hr, wl:wr] += x * mask
            weight[hl:hr, wl:wr] += mask
        values /= weight
        return values
