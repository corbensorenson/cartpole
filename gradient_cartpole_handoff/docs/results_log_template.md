# Results Log Template

Copy this section for every meaningful run.

## Run name

- Date:
- Machine:
- macOS:
- Python:
- MuJoCo:
- MLX:
- Git commit / packet version:
- Config path:
- Config changes:

## Training

- Started:
- Finished:
- Wall-clock:
- Total env steps:
- Best checkpoint:
- Best update:
- Best eval return:
- Best eval success rate:

## Final evaluation

Command:

```bash
python scripts/evaluate.py ...
```

Metrics:

```json
{}
```

## Video

Command:

```bash
python scripts/render_video.py ...
```

Path:

```text
runs/.../video.mp4
```

## Notes

- Did the video reset?
- Did cart hit rail?
- Were upper links wobbling?
- Was force saturating?
- What is the next config change?

