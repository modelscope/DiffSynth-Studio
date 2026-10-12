"""Exercise H3 VAE spatial dispatch with small, real production neural layers."""

import unittest

import torch
from torchvision.transforms import Normalize

from diffsynth.models.minimax_h3_video_vae import (
    EncoderFCN3D,
    MiniMaxH3VideoVAE,
    VAEProcessor,
    ViT3DDecoder,
)


class SmallH3VideoVAE(MiniMaxH3VideoVAE):
    """Keep H3 geometry and inherited methods, reducing only neural layer widths."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.vae_ratio = 16
        self.vae_ratio_t = 4
        self.clip_length = 17
        self.token_drop = 3
        self.frame_pre_padding = 3
        self.tokens_chunk_size = 5
        self.token_overlap = 2
        self.frame_overlap = 5
        self.tile_size = 256
        self.tile_overlap_min = 64
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.processor = VAEProcessor(
            Normalize(mean, std),
            Normalize(tuple(-m / s for m, s in zip(mean, std)), tuple(1 / s for s in std)),
        )
        self.encoder = EncoderFCN3D(
            ch=32, ch_mult=(1, 1, 1, 1), space_down=(2, 2, 2, 2), time_down=(1, 2, 2, 1),
            num_res_blocks=1, in_channels=3, z_channels=24, double_z=True,
            padding_mode="reflect", causal=True, use_t_isolated_gn=True,
        )
        self.quant_conv = torch.nn.Conv3d(48, 48, 1)
        self.post_quant_conv = torch.nn.Conv3d(24, 24, 1)
        self.decoder = ViT3DDecoder(
            patch_size=16, patch_size_t=4, in_channels=24, out_channels=3,
            heads=2, dim_head=8, num_layers=1,
        )
        # Nonzero residual gates exercise attention across the spatial canvas;
        # the default zero initialization would hide decoder tiling differences.
        with torch.no_grad():
            for block in self.decoder.transformer_blocks:
                block.scale1.weight.fill_(0.5)
                block.scale2.weight.fill_(0.5)


class H3VideoVAETilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(311)
            cls.vae = SmallH3VideoVAE().eval()
            cls.images = torch.rand(1, 3, 64, 96)
            cls.videos = {frames: torch.rand(1, 3, frames, 64, 96) for frames in (22, 39)}
            cls.latents = {tokens: torch.rand(1, 24, tokens, 4, 6) for tokens in (1, 7, 12)}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def record_calls(self, layer, method, tensor, **kwargs):
        shapes = []
        handle = layer.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
        try:
            result = method(tensor, dtype=torch.float32, **kwargs)
        finally:
            handle.remove()
        return result, shapes

    def test_untiled_encode_keeps_full_canvas_and_temporal_clips(self):
        for frames, input_tensor, tokens, clips in (
            (1, self.images, 1, 1),
            (22, self.videos[22], 7, 2),
            (39, self.videos[39], 12, 3),
        ):
            with self.subTest(frames=frames):
                image = frames == 1
                # A single spatial tile is an independent full-canvas oracle,
                # retaining the existing temporal padding/drop and normalization.
                expected = self.vae.encode_video(input_tensor, process_image=image, tiled=True, tile_size=256)
                actual, shapes = self.record_calls(
                    self.vae.encoder, self.vae.encode_video, input_tensor,
                    process_image=image, tiled=False, tile_size=64, tile_overlap=32,
                )
                self.assertEqual(shapes, [(1, 3, 1 if image else 17, 64, 96)] * clips)
                self.assertEqual(actual.shape, (1, 24, tokens, 4, 6))
                self.assertEqual(actual.dtype, input_tensor.dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                alternate = self.vae.encode_video(
                    input_tensor, process_image=image, tiled=False, tile_size=48, tile_overlap=16,
                )
                torch.testing.assert_close(actual, alternate, rtol=0, atol=0)

    def test_untiled_decode_keeps_full_canvas_and_temporal_overlap(self):
        for tokens, frames, clips in ((1, 4, 1), (7, 22, 1), (12, 39, 2)):
            with self.subTest(tokens=tokens):
                image = tokens == 1
                latents = self.latents[tokens]
                expected = self.vae.decode_video(latents, process_image=image, tiled=True, tile_size=256)
                actual, shapes = self.record_calls(
                    self.vae.decoder, self.vae.decode_video, latents,
                    process_image=image, tiled=False, tile_size=64, tile_overlap=32,
                )
                self.assertEqual(shapes, [(1, 24, 1 if image else 7, 4, 6)] * clips)
                # Preserve the existing process_image result (four decoder frames).
                self.assertEqual(actual.shape, (1, 3, frames, 64, 96))
                self.assertEqual(actual.dtype, latents.dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                alternate = self.vae.decode_video(
                    latents, process_image=image, tiled=False, tile_size=48, tile_overlap=16,
                )
                torch.testing.assert_close(actual, alternate, rtol=0, atol=0)

    def test_default_encode_stays_tiled_after_untiled_call(self):
        for image, input_tensor, clips in ((True, self.images, 1), (False, self.videos[22], 2)):
            with self.subTest(image=image):
                kwargs = {"process_image": image, "tile_size": 64, "tile_overlap": 32}
                expected, shapes = self.record_calls(
                    self.vae.encoder, self.vae.encode_video, input_tensor, tiled=True, **kwargs,
                )
                self.assertEqual(shapes, [(1, 3, 1 if image else 17, 64, 64)] * (clips * 2))
                self.vae.encode_video(input_tensor, tiled=False, **kwargs)
                default = self.vae.encode_video(input_tensor, **kwargs)
                explicit = self.vae.encode_video(input_tensor, tiled=True, **kwargs)
                torch.testing.assert_close(default, expected, rtol=0, atol=0)
                torch.testing.assert_close(explicit, expected, rtol=0, atol=0)

    def test_default_decode_stays_tiled_after_untiled_call(self):
        for image, tokens, clips in ((True, 1, 1), (False, 12, 2)):
            with self.subTest(image=image):
                kwargs = {"process_image": image, "tile_size": 64, "tile_overlap": 32}
                expected, shapes = self.record_calls(
                    self.vae.decoder, self.vae.decode_video, self.latents[tokens], tiled=True, **kwargs,
                )
                self.assertEqual(shapes, [(1, 24, 1 if image else 7, 4, 4)] * (clips * 2))
                self.vae.decode_video(self.latents[tokens], tiled=False, **kwargs)
                default = self.vae.decode_video(self.latents[tokens], **kwargs)
                explicit = self.vae.decode_video(self.latents[tokens], tiled=True, **kwargs)
                torch.testing.assert_close(default, expected, rtol=0, atol=0)
                torch.testing.assert_close(explicit, expected, rtol=0, atol=0)

    def test_compute_dtype_preserves_input_output_dtype(self):
        for image, input_tensor in ((True, self.images), (False, self.videos[22])):
            with self.subTest(image=image):
                latents = self.vae.encode_video(
                    input_tensor.double(), dtype=torch.float32, process_image=image, tiled=False,
                )
                recon = self.vae.decode_video(latents, dtype=torch.float32, process_image=image, tiled=False)
                self.assertEqual(latents.dtype, torch.float64)
                self.assertEqual(recon.dtype, torch.float64)
                self.assertTrue(torch.isfinite(latents).all())
                self.assertTrue(torch.isfinite(recon).all())


if __name__ == "__main__":
    unittest.main()
