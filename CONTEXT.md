# music-remover

Strip background music from a local media file or URL, keeping speech and sound effects, using HTDemucs trained on Divide and Remaster.

## Language

**Stem**:
One of the three separated sources the DnR model produces: speech, sfx, or music.
_Avoid_: track, channel, drums, bass, other, vocals

**StemSet**:
The three **Stems** from one separation, named and held together. Owns the mapping from **Keep** to the stereo that leaves the pipeline.
_Avoid_: stems Vec, sources array

**Keep**:
Which part of a **StemSet** to retain for output — soundtrack (speech + sfx; drop the score), speech (dialogue only), or music (score only).
_Avoid_: mode (that word means output shape: video vs audio), vocals, instrumental

**Stereo**:
A pair of left/right sample buffers at the pipeline sample rate. What decode produces, what a **Stem** holds, and what **Keep** leaves behind.
_Avoid_: channels array, `[Vec; 2]` spoken as the concept

**Job**:
One music-removal request end-to-end: resolve the input, separate, apply **Keep**, write the output. Distinct from **Plan**, which only decides video-vs-audio and the output path.
_Avoid_: pipeline, run (as the type name), main script

**Input**:
What a **Job** operates on — a local media path, or a URL to fetch first.
_Avoid_: source (when you mean the Job’s input), argument string

**Mode**:
Whether the **Job** should produce video, audio, or decide from the source (`auto`). Distinct from **Keep**.
_Avoid_: format (that’s the audio container/codec), keep

**Plan**:
The output decision only: whether to produce video, and where to write the file. Does not select stems. Produced inside a **Job**, not a substitute for one.
_Avoid_: treating Plan as the full execution

## Example

> Dev: So the Job takes an Input, builds a Plan, runs the model, then… picks stems how?
> Expert: The model returns a StemSet. Keep says soundtrack, speech, or music — that's StemSet.keep, not Plan.
> Dev: And Plan is just “mp4 next to the file”?
> Expert: Right — Mode plus path. Peak limiting is on the Stereo that Keep leaves behind. Soundtrack is the default: speech plus sound effects, suffix `nomusic`.
