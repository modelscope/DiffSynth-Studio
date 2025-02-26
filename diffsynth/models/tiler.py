import torch
from einops import rearrange, repeat


class TileWorker:
    def __init__(self):
        pass


    def mask(self, height, width, border_width):
        # Create a mask with shape (height, width).
        # The centre area is filled with 1, and the border line is filled with values in range (0, 1].
        x = torch.arange(height).repeat(width, 1).T
        y = torch.arange(width).repeat(height, 1)
        mask = torch.stack([x + 1, height - x, y + 1, width - y]).min(dim=0).values
        mask = (mask / border_width).clip(0, 1)
        return mask


    def tile(self, model_input, tile_size, tile_stride, tile_device, tile_dtype):
        # Convert a tensor (b, c, h, w) to (b, c, tile_size, tile_size, tile_num)
        batch_size, channel, _, _ = model_input.shape
        model_input = model_input.to(device=tile_device, dtype=tile_dtype)
        unfold_operator = torch.nn.Unfold(
            kernel_size=(tile_size, tile_size),
            stride=(tile_stride, tile_stride)
        )
        model_input = unfold_operator(model_input)
        model_input = model_input.view((batch_size, channel, tile_size, tile_size, -1))

        return model_input


    def tiled_inference(self, forward_fn, model_input, tile_batch_size, inference_device, inference_dtype, tile_device, tile_dtype):
        # Call y=forward_fn(x) for each tile
        tile_num = model_input.shape[-1]
        model_output_stack = []

        for tile_id in range(0, tile_num, tile_batch_size):

            # process input
            tile_id_ = min(tile_id + tile_batch_size, tile_num)
            x = model_input[:, :, :, :, tile_id: tile_id_]
            x = x.to(device=inference_device, dtype=inference_dtype)
            x = rearrange(x, "b c h w n -> (n b) c h w")

            # process output
            y = forward_fn(x)
            y = rearrange(y, "(n b) c h w -> b c h w n", n=tile_id_-tile_id)
            y = y.to(device=tile_device, dtype=tile_dtype)
            model_output_stack.append(y)

        model_output = torch.concat(model_output_stack, dim=-1)
        return model_output


    def io_scale(self, model_output, tile_size):
        # Determine the size modification happened in forward_fn
        # We only consider the same scale on height and width.
        io_scale = model_output.shape[2] / tile_size
        return io_scale
    

    def untile(self, model_output, height, width, tile_size, tile_stride, border_width, tile_device, tile_dtype):
        # The reversed function of tile
        mask = self.mask(tile_size, tile_size, border_width)
        mask = mask.to(device=tile_device, dtype=tile_dtype)
        mask = rearrange(mask, "h w -> 1 1 h w 1")
        model_output = model_output * mask

        fold_operator = torch.nn.Fold(
            output_size=(height, width),
            kernel_size=(tile_size, tile_size),
            stride=(tile_stride, tile_stride)
        )
        mask = repeat(mask[0, 0, :, :, 0], "h w -> 1 (h w) n", n=model_output.shape[-1])
        model_output = rearrange(model_output, "b c h w n -> b (c h w) n")
        model_output = fold_operator(model_output) / fold_operator(mask)

        return model_output


    def tiled_forward(self, forward_fn, model_input, tile_size, tile_stride, tile_batch_size=1, tile_device="cpu", tile_dtype=torch.float32, border_width=None):
        # Prepare
        inference_device, inference_dtype = model_input.device, model_input.dtype
        height, width = model_input.shape[2], model_input.shape[3]
        border_width = int(tile_stride*0.5) if border_width is None else border_width

        # tile
        model_input = self.tile(model_input, tile_size, tile_stride, tile_device, tile_dtype)

        # inference
        model_output = self.tiled_inference(forward_fn, model_input, tile_batch_size, inference_device, inference_dtype, tile_device, tile_dtype)

        # resize
        io_scale = self.io_scale(model_output, tile_size)
        height, width = int(height*io_scale), int(width*io_scale)
        tile_size, tile_stride = int(tile_size*io_scale), int(tile_stride*io_scale)
        border_width = int(border_width*io_scale)

        # untile
        model_output = self.untile(model_output, height, width, tile_size, tile_stride, border_width, tile_device, tile_dtype)
        
        # Done!
        model_output = model_output.to(device=inference_device, dtype=inference_dtype)
        return model_output
    


class FastTileWorker:
    def __init__(self):
        pass


    def build_mask(self, data, is_bound):
        _, _, H, W = data.shape
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
        mask = rearrange(mask, "H W -> 1 H W")
        return mask


    def tiled_forward(self, forward_fn, model_input, tile_size, tile_stride, tile_device="cpu", tile_dtype=torch.float32, border_width=None):
        # Prepare
        B, C, H, W = model_input.shape
        border_width = int(tile_stride*0.5) if border_width is None else border_width
        weight = torch.zeros((1, 1, H, W), dtype=tile_dtype, device=tile_device)
        values = torch.zeros((B, C, H, W), dtype=tile_dtype, device=tile_device)

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
        for hl, hr, wl, wr in tasks:
            # Forward
            hidden_states_batch = forward_fn(hl, hr, wl, wr).to(dtype=tile_dtype, device=tile_device)

            mask = self.build_mask(hidden_states_batch, is_bound=(hl==0, hr>=H, wl==0, wr>=W))
            values[:, :, hl:hr, wl:wr] += hidden_states_batch * mask
            weight[:, :, hl:hr, wl:wr] += mask
        values /= weight
        return values



class TileWorker2Dto3D:
    """
    Process 3D tensors, but only enable TileWorker on 2D.
    """
    def __init__(self):
        pass


    def build_mask(self, T, H, W, dtype, device, is_bound, border_width):
        t = repeat(torch.arange(T), "T -> T H W", T=T, H=H, W=W)
        h = repeat(torch.arange(H), "H -> T H W", T=T, H=H, W=W)
        w = repeat(torch.arange(W), "W -> T H W", T=T, H=H, W=W)
        border_width = (H + W) // 4 if border_width is None else border_width
        pad = torch.ones_like(h) * border_width
        mask = torch.stack([
            pad if is_bound[0] else t + 1,
            pad if is_bound[1] else T - t,
            pad if is_bound[2] else h + 1,
            pad if is_bound[3] else H - h,
            pad if is_bound[4] else w + 1,
            pad if is_bound[5] else W - w
        ]).min(dim=0).values
        mask = mask.clip(1, border_width)
        mask = (mask / border_width).to(dtype=dtype, device=device)
        mask = rearrange(mask, "T H W -> 1 1 T H W")
        return mask


    def tiled_forward(
        self,
        forward_fn,
        model_input,
        tile_size, tile_stride,
        tile_device="cpu", tile_dtype=torch.float32,
        computation_device="cuda", computation_dtype=torch.float32,
        border_width=None, scales=[1, 1, 1, 1],
        progress_bar=lambda x:x
    ):
        B, C, T, H, W = model_input.shape
        scale_C, scale_T, scale_H, scale_W = scales
        tile_size_H, tile_size_W = tile_size
        tile_stride_H, tile_stride_W = tile_stride

        value = torch.zeros((B, int(C*scale_C), int(T*scale_T), int(H*scale_H), int(W*scale_W)), dtype=tile_dtype, device=tile_device)
        weight = torch.zeros((1, 1, int(T*scale_T), int(H*scale_H), int(W*scale_W)), dtype=tile_dtype, device=tile_device)

        # Split tasks
        tasks = []
        for h in range(0, H, tile_stride_H):
            for w in range(0, W, tile_stride_W):
                if (h-tile_stride_H >= 0 and h-tile_stride_H+tile_size_H >= H) or (w-tile_stride_W >= 0 and w-tile_stride_W+tile_size_W >= W):
                    continue
                h_, w_ = h + tile_size_H, w + tile_size_W
                if h_ > H: h, h_ = max(H - tile_size_H, 0), H
                if w_ > W: w, w_ = max(W - tile_size_W, 0), W
                tasks.append((h, h_, w, w_))

        # Run
        for hl, hr, wl, wr in progress_bar(tasks):
            mask = self.build_mask(
                int(T*scale_T), int((hr-hl)*scale_H), int((wr-wl)*scale_W),
                tile_dtype, tile_device,
                is_bound=(True, True, hl==0, hr>=H, wl==0, wr>=W),
                border_width=border_width
            )
            grid_input = model_input[:, :, :, hl:hr, wl:wr].to(dtype=computation_dtype, device=computation_device)
            grid_output = forward_fn(grid_input).to(dtype=tile_dtype, device=tile_device)
            value[:, :, :, int(hl*scale_H):int(hr*scale_H), int(wl*scale_W):int(wr*scale_W)] += grid_output * mask
            weight[:, :, :, int(hl*scale_H):int(hr*scale_H), int(wl*scale_W):int(wr*scale_W)] += mask
        value = value / weight
        return value