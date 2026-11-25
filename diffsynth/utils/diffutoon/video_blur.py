"""Gaussian-blur track for a whole video: one mp4 in, one blurred mp4 out.

The counterpart of video_lineart: that one draws outlines, this one degrades the
frames so a restoration ControlNet has something to restore. Resolution, frame
count and fps are unchanged, so input and output stay pixel-aligned.

Blurring runs in-process through PyAV (`av`) on libavfilter's `gblur`, the very
filter the ffmpeg CLI would run: for the same sigma/steps the filtered frames
are bit-identical to `ffmpeg -vf gblur=...`. Audio is remuxed as packets, never
decoded or re-encoded -- what `-c:a copy` did -- so the soundtrack the training
loader reads from this file survives unchanged. No ffmpeg executable is run.

gblur is a recursive IIR approximation (Alvarez-Mazorra), not an FIR
convolution, so `steps` is the number of passes rather than a kernel size.
Measured sigma is accurate (5.0 nominal -> 5.026) and the shape is within ~1%
of a true Gaussian.

Call it from Python as `video_to_blur(input_path, output_path, sigma=15)`; it
returns the path of the written video.
"""
import os

DEFAULT_SIGMA = 5.0
STEPS = 3     # within 0.7% of a true Gaussian already; more passes buy little
CRF = 16


def _copy_stream(container, stream):
    """Output stream that reuses `stream`'s codec, so packets can be muxed as-is.

    PyAV renamed this between releases: newer versions expose
    `add_stream_from_template`, older ones only take `add_stream(template=...)`.
    """
    add_from_template = getattr(container, "add_stream_from_template", None)
    if add_from_template is not None:
        return add_from_template(stream)
    return container.add_stream(template=stream)


def video_to_blur(input_path, output_path, sigma=DEFAULT_SIGMA, steps=STEPS, crf=CRF):
    """Blur every frame with sigma, remuxing the audio track untouched.

    `av` is the optional `[audio]` extra, so it is imported here rather than at
    module level; a missing install raises the plain ImportError from the import.
    """
    import av

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with (
        av.open(input_path) as input_container,
        # +faststart puts the moov atom first, as the old ffmpeg command did.
        av.open(output_path, mode="w", options={"movflags": "+faststart"}) as output_container,
    ):
        in_video = input_container.streams.video[0]
        in_audio = input_container.streams.audio[0] if input_container.streams.audio else None
        rate = in_video.average_rate or in_video.guessed_rate
        if rate is None:
            raise ValueError(f"cannot determine the frame rate of {input_path}")

        out_video = output_container.add_stream(
            "libx264", rate=rate, options={"crf": str(crf), "preset": "medium"},
        )
        out_video.width = in_video.codec_context.width
        out_video.height = in_video.codec_context.height
        out_video.pix_fmt = "yuv420p"
        if in_video.time_base is not None:
            out_video.time_base = in_video.time_base          # frames keep their own pts
        out_audio = _copy_stream(output_container, in_audio) if in_audio is not None else None

        graph = av.filter.Graph()
        source = graph.add_buffer(template=in_video)
        blur = graph.add("gblur", f"sigma={sigma}:steps={steps}")
        sink = graph.add("buffersink")
        source.link_to(blur)
        blur.link_to(sink)
        graph.configure()

        def encode_blurred():
            """Mux every frame the graph has produced so far."""
            while True:
                try:
                    frame = graph.pull()
                except (av.error.BlockingIOError, av.error.EOFError):
                    return      # pull() reports "nothing ready" and "finished" this way
                if frame.format.name != "yuv420p":
                    frame = frame.reformat(format="yuv420p")
                for packet in out_video.encode(frame):
                    output_container.mux(packet)

        # One demux pass feeds both streams: video packets go through the filter
        # graph, audio packets are remuxed where they are. Demuxing the audio
        # afterwards cannot work, because decoding the video drains the container.
        for packet in input_container.demux():
            if packet.dts is None:            # flush marker, not media
                continue
            if packet.stream is in_video:
                for frame in in_video.codec_context.decode(packet):
                    graph.push(frame)
                    encode_blurred()
            elif out_audio is not None and packet.stream is in_audio:
                packet.stream = out_audio
                output_container.mux(packet)
        for frame in in_video.codec_context.decode(None):     # flush the decoder
            graph.push(frame)
            encode_blurred()
        graph.push(None)                                      # flush the filter graph
        encode_blurred()
        for packet in out_video.encode():                     # flush the encoder
            output_container.mux(packet)
    return output_path
