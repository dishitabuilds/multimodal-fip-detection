"""Build the bundled fictional video from its checked-in synthetic narration.

Requires the existing media dependencies (PyAV, Pillow, numpy). No downloads,
external footage, trained classifiers or prepared results are used.
"""
from fractions import Fraction
from pathlib import Path
import math
import wave

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "test_samples/media"
SCENES = [
    (0, "A fictional promotion", ["Classroom demonstration", "Synthetic narration"]),
    (5, "Guaranteed monthly returns", ["Double your money"]),
    (9, "Limited-time offer", ["Invest today"]),
    (13, "Review the evidence", ["Check the original source", "Verify claims independently"]),
]


def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def card(title, lines):
    image = Image.new("RGB", (960, 540), "#f5f2eb")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 960, 9), fill="#2d6855")
    draw.text((48, 42), "PROMOTION LENS  /  FICTIONAL DEMO", font=font(22), fill="#2d6855")
    draw.text((48, 180), title, font=font(44), fill="#203f36")
    for index, line in enumerate(lines):
        draw.text((48, 263 + index * 52), line, font=font(30), fill="#40564e")
    draw.line((48, 450, 912, 450), fill="#cad3cb", width=2)
    draw.text((48, 473), "Illustrative claims for analysis. No real investment is offered.",
              font=font(21), fill="#40564e")
    return image


def main():
    with wave.open(str(ASSETS / "demo_narration.wav"), "rb") as source:
        rate = source.getframerate()
        assert source.getnchannels() == 1 and source.getsampwidth() == 2
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    duration = len(samples) / rate + 1
    images = [(start, card(title, lines)) for start, title, lines in SCENES]
    target = ASSETS / "promotion_demo.mp4"
    with av.open(str(target), "w", options={"movflags": "+faststart"}) as output:
        video = output.add_stream("libx264", rate=10)
        video.width, video.height, video.pix_fmt = 960, 540, "yuv420p"
        video.options = {"crf": "24", "preset": "medium"}
        audio = output.add_stream("aac", rate=rate)
        audio.layout = "mono"
        audio.bit_rate = 64000
        # Interleave by presentation time for browser playback and bounded decode.
        audio_cursor = 0
        for index in range(math.ceil(duration * 10)):
            image = next(image for start, image in reversed(images) if start <= index / 10)
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in video.encode(frame):
                output.mux(packet)
            while audio_cursor < min(len(samples), math.ceil((index + 1) * rate / 10)):
                values = samples[audio_cursor:audio_cursor + 1024].astype(np.float32) / 32768
                frame = av.AudioFrame.from_ndarray(values.reshape(1, -1), format="fltp", layout="mono")
                frame.sample_rate = rate
                frame.pts, frame.time_base = audio_cursor, Fraction(1, rate)
                for packet in audio.encode(frame):
                    output.mux(packet)
                audio_cursor += len(values)
        for stream in (video, audio):
            for packet in stream.encode(None):
                output.mux(packet)
    print(f"Built {target.name}: {duration:.1f}s, {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
