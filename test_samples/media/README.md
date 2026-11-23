# Fictional media demonstration

`promotion_demo.mp4` is a project-authored classroom example, with text cards and
synthetic English narration generated using the installed Microsoft Zira Desktop
voice. It contains no real offer, third-party footage or personal recording.
`demo_narration.wav` is its reusable audio source.

Narration:

> This is a fictional promotion for a classroom demonstration. Guaranteed monthly
> returns. Double your money. This is a limited-time offer. Invest today. Review
> the claims and check the original source.

The page runs the normal decoder, speech recognizer, OCR and wording audit when
Analyze is clicked. No precomputed transcript or predictions are substituted.
Expected wording cues are return promises and urgency; recognition can vary.
This synthetic example demonstrates integration, not detection accuracy.

Rebuild the video with `python scripts/build_media_demo.py` in the media environment.
