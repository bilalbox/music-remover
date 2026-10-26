// Adapters for the external media toolchain this pipeline shells out to: yt-dlp (fetch),
// ffprobe (stream inspection), and ffmpeg (decode/encode/remux). Every ffmpeg/yt-dlp
// invocation in the project lives here.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};

use crate::plan::{AudioCodec, Container};
use crate::stems::{Stereo, SAMPLE_RATE};

/// Run a command to completion (stdio inherited), bailing with `fail_msg` on non-zero exit.
fn run_checked(cmd: &mut Command, program: &str, fail_msg: impl std::fmt::Display) -> Result<()> {
    let status = cmd.status().with_context(|| format!("spawning {program}"))?;
    if !status.success() {
        bail!("{fail_msg}");
    }
    Ok(())
}

/// Download the source and return its title, in a single yt-dlp invocation. The title is
/// captured via `--print-to-file` (a side-channel file, written after the download
/// completes) rather than a separate `--print --skip-download` metadata-only request, so
/// we only resolve/download the source once. `-o`'s live progress output still goes
/// straight to our inherited stdio.
pub fn fetch(url: &str, want_video: bool, dir: &Path) -> Result<(String, PathBuf)> {
    let template = dir.join("source.%(ext)s");
    let title_file = dir.join("title.txt");
    let mut cmd = Command::new("yt-dlp");
    cmd.args(["--no-warnings", "--no-playlist"]);
    if want_video {
        // ≤720p, WebM first so video+audio merge as a stream copy (VP9/AV1 + Opus).
        // Falls back to any ≤720p video plus best audio when WebM is unavailable.
        // Equivalent of:
        //   yt-dlp -f "bv*[height<=720][ext=webm]+ba[ext=webm]/bv*[height<=720]+ba"
        // Final remux may be mp4 or webm; merging to mp4 here can transcode.
        cmd.args([
            "-f",
            "bv*[height<=720][ext=webm]+ba[ext=webm]/bv*[height<=720]+ba",
        ]);
    } else {
        cmd.args(["-f", "bestaudio/best"]);
    }
    cmd.arg("-o")
        .arg(&template)
        .arg("--print-to-file")
        .arg("after_move:%(title)s")
        .arg(&title_file)
        .arg(url);
    run_checked(&mut cmd, "yt-dlp", "yt-dlp exited with failure")?;

    let title = std::fs::read_to_string(&title_file)
        .unwrap_or_default()
        .trim()
        .to_string();
    let title = if title.is_empty() {
        "output".to_string()
    } else {
        title
    };

    // Find the produced source.* file.
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_stem().map(|s| s == "source").unwrap_or(false))
        .collect();
    match found.len() {
        0 => bail!("yt-dlp produced no output file"),
        1 => Ok((title, found.pop().unwrap())),
        n => bail!(
            "yt-dlp produced {n} candidate output files (expected exactly 1): {:?}",
            found
        ),
    }
}

pub fn has_video_stream(path: &Path) -> Result<bool> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .context("spawning ffprobe")?;
    Ok(String::from_utf8_lossy(&out.stdout).contains("video"))
}

/// Decode any input to pipeline-rate stereo.
pub fn decode(input: &Path) -> Result<Stereo> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(input)
        .args([
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            "2",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("spawning ffmpeg (is it on PATH?)")?;
    if !out.status.success() {
        bail!(
            "ffmpeg decode failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let bytes = out.stdout;
    let frames = bytes.len() / 8; // 2ch * 4 bytes
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for ch in bytes.chunks_exact(8) {
        left.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
        right.push(f32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]));
    }
    Ok(Stereo { left, right })
}

/// Write a 16-bit PCM stereo WAV file directly — no ffmpeg round-trip needed for a format
/// this simple (44-byte header + samples). Used for `--keep-temp` stem dumps and wav output.
pub fn write_wav(path: &Path, audio: &Stereo) -> Result<()> {
    debug_assert_eq!(audio.left.len(), audio.right.len());
    let n = audio.len();
    const CHANNELS: u32 = 2;
    const BITS_PER_SAMPLE: u32 = 16;
    let block_align = CHANNELS * BITS_PER_SAMPLE / 8;
    let byte_rate = SAMPLE_RATE * block_align;
    let data_len = n as u32 * block_align;

    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size (PCM)
    buf.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat = PCM
    buf.extend_from_slice(&(CHANNELS as u16).to_le_bytes());
    buf.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&(block_align as u16).to_le_bytes());
    buf.extend_from_slice(&(BITS_PER_SAMPLE as u16).to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());

    for i in 0..n {
        let l = (audio.left[i].clamp(-1.0, 1.0) * 32767.0).round() as i16;
        let r = (audio.right[i].clamp(-1.0, 1.0) * 32767.0).round() as i16;
        buf.extend_from_slice(&l.to_le_bytes());
        buf.extend_from_slice(&r.to_le_bytes());
    }
    std::fs::write(path, &buf).with_context(|| format!("writing wav to {}", path.display()))
}

/// Interleaved little-endian f32 stereo — what ffmpeg's `f32le` demuxer expects on stdin.
fn write_f32le_stereo(audio: &Stereo, mut w: impl Write) -> Result<()> {
    debug_assert_eq!(audio.left.len(), audio.right.len());
    for i in 0..audio.len() {
        w.write_all(&audio.left[i].to_le_bytes())?;
        w.write_all(&audio.right[i].to_le_bytes())?;
    }
    Ok(())
}

fn wants_bitrate(format: &str) -> bool {
    matches!(
        format.to_ascii_lowercase().as_str(),
        "mp3" | "m4a" | "aac" | "opus" | "ogg" | "webm"
    )
}

/// Encode cleaned float stereo to `out`. Wav is written as 16-bit PCM; lossy formats are
/// fed f32le on stdin so ffmpeg never sees a 16-bit intermediate.
pub fn encode(audio: &Stereo, out: &Path, format: &str, bitrate: &str) -> Result<()> {
    if format.eq_ignore_ascii_case("wav") {
        return write_wav(out, audio);
    }

    let rate = SAMPLE_RATE.to_string();
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-v",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        &rate,
        "-ac",
        "2",
        "-i",
        "pipe:0",
    ]);
    if wants_bitrate(format) {
        cmd.args(["-b:a", bitrate]);
    }
    cmd.arg(out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().context("spawning ffmpeg")?;
    {
        let mut stdin = child
            .stdin
            .take()
            .context("ffmpeg stdin unavailable")?;
        write_f32le_stereo(audio, &mut stdin).context("writing float PCM to ffmpeg")?;
    }
    let output = child
        .wait_with_output()
        .context("waiting for ffmpeg encode")?;
    if !output.status.success() {
        bail!(
            "ffmpeg audio encode failed for {}: {}",
            out.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

/// Remux options for replacing a source video's audio with cleaned float PCM.
pub struct RemuxOpts<'a> {
    pub container: Container,
    pub codec: AudioCodec,
    pub bitrate: &'a str,
}

/// Replace the source's audio with the cleaned track (float PCM on stdin), copying video.
pub fn remux(source: &Path, audio: &Stereo, out: &Path, opts: &RemuxOpts<'_>) -> Result<()> {
    let rate = SAMPLE_RATE.to_string();
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-y", "-i"])
        .arg(source)
        .args([
            "-f",
            "f32le",
            "-ar",
            &rate,
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            opts.codec.ffmpeg_name(),
            "-b:a",
            opts.bitrate,
            "-shortest",
        ]);
    if opts.container == Container::Mp4 {
        cmd.args(["-movflags", "+faststart"]);
    }
    cmd.arg(out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().context("spawning ffmpeg")?;
    {
        let mut stdin = child
            .stdin
            .take()
            .context("ffmpeg stdin unavailable")?;
        write_f32le_stereo(audio, &mut stdin).context("writing float PCM to ffmpeg")?;
    }
    let output = child
        .wait_with_output()
        .context("waiting for ffmpeg remux")?;
    if !output.status.success() {
        bail!(
            "ffmpeg remux failed for {}: {}",
            out.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}
