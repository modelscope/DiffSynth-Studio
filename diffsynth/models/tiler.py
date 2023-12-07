import torch


class Tiler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def mask(self, height, width, line_width):
        x = torch.arange(height).repeat(width, 1).T
        y = torch.arange(width).repeat(height, 1)
        mask = torch.stack([x + 1, height - x, y + 1, width - y]).min(dim=0).values
        mask = (mask / line_width).clip(0, 1)
        return mask

    def forward(self, forward_fn, x, tile_size, tile_stride, batch_size=1, inter_device="cpu", inter_dtype=torch.float32):
        # Prepare
        device = x.device
        torch_dtype = x.dtype

        # tile
        b, c_in, h_in, w_in = x.shape
        x = x.to(device=inter_device, dtype=inter_dtype)
        fold_params = {
            "kernel_size": (tile_size, tile_size),
            "stride": (tile_stride, tile_stride)
        }
        unfold_operator = torch.nn.Unfold(**fold_params)
        x = unfold_operator(x)
        x = x.view((b, c_in, tile_size, tile_size, -1))

        # inference
        x_out_stack = []
        for tile_id in range(0, x.shape[-1], batch_size):

            # process input
            next_tile_id = min(tile_id + batch_size, x.shape[-1])
            x_in = x[:, :, :, :, tile_id: next_tile_id]
            x_in = x_in.to(device=device, dtype=torch_dtype)
            x_in = x_in.permute(4, 0, 1, 2, 3)
            x_in = x_in.view((x_in.shape[0]*x_in.shape[1], x_in.shape[2], x_in.shape[3], x_in.shape[4]))

            # process output
            x_out = forward_fn(x_in)
            x_out = x_out.view((next_tile_id - tile_id, b, x_out.shape[1], x_out.shape[2], x_out.shape[3]))
            x_out = x_out.permute(1, 2, 3, 4, 0)
            x_out = x_out.to(device=inter_device, dtype=inter_dtype)
            x_out_stack.append(x_out)

        x = torch.concat(x_out_stack, dim=-1)

        # untile
        in2out_scale = x.shape[2] / tile_size
        h_out, w_out = int(h_in * in2out_scale), int(w_in * in2out_scale)

        mask = self.mask(int(tile_size * in2out_scale), int(tile_size * in2out_scale), int(tile_stride * in2out_scale * 0.5))
        mask = mask.to(device=inter_device, dtype=inter_dtype)
        mask = mask.reshape((1, 1, mask.shape[0], mask.shape[1], 1))
        x = x * mask

        fold_params = {
            "kernel_size": (int(tile_size * in2out_scale), int(tile_size * in2out_scale)),
            "stride": (int(tile_stride * in2out_scale), int(tile_stride * in2out_scale))
        }
        fold_operator = torch.nn.Fold(output_size=(h_out, w_out), **fold_params)
        divisor = fold_operator(mask.repeat(1, 1, 1, 1, x.shape[-1]).view(b, -1, x.shape[-1]))

        x = x.view((b, -1, x.shape[-1]))
        x = fold_operator(x) / divisor
        x = x.to(device=device, dtype=torch_dtype)

        return x

        

    