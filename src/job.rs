// One music-removal Job end-to-end: resolve input, separate, Keep, write output.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};

use crate::demucs::{self, Demucs};
use crate::media::{self, RemuxOpts};
use crate::plan::{AudioCodec, Container, Mode, Plan};
use crate::stems::{Keep, StemLevels, SAMPLE_RATE};

/// What a Job operates on — parsed at the CLI edge.
pub enum Input {
    Url(String),
    Path(PathBuf),
}

impl Input {
    pub fn parse(s: &str) -> Self {
        if s.starts_with("http://") || s.starts_with("https://") {
            Input::Url(s.to_string())
        } else {
            Input::Path(PathBuf::from(s))
        }
    }
}

pub struct JobSpec {
    pub input: Input,
    pub mode: Mode,
    pub keep: Keep,
    pub audio_format: String,
    /// Video mux container (`mp4` / `webm`). Ignored for audio-only output.
    pub container: Container,
    /// Lossy codec for video remux. `None` picks the container default (AAC for mp4, Opus for webm).
    pub audio_codec: Option<AudioCodec>,
    /// Target bitrate for lossy audio (`192k`, `128k`, …).
    pub audio_bitrate: String,
    pub output: Option<PathBuf>,
    pub device: String,
    pub shifts: usize,
    pub model: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub keep_temp: bool,
}

pub enum Progress {
    Fetching {
        url: String,
    },
    ModelReady {
        device: String,
        elapsed_secs: f32,
        devices: Vec<String>,
        cache: String,
    },
    Decoded {
        audio_secs: f32,
        elapsed_secs: f32,
    },
    Separating(f32),
    Separated {
        elapsed_secs: f32,
        realtime: f64,
    },
    StemLevels(StemLevels),
    KeepingTemp(PathBuf),
}

pub fn run(spec: JobSpec, on_progress: &mut dyn FnMut(Progress)) -> Result<PathBuf> {
    let model_dir = spec.model.clone().unwrap_or_else(default_model_dir);
    if !model_dir.join(demucs::MODEL_XML).exists() {
        bail!(
            "model not found at {} (run ./fetch-model.sh, or pass --model)",
            model_dir.display()
        );
    }

    let mut work = TempDir::new()?;

    let (media_path, title, is_url) = match &spec.input {
        Input::Url(url) => {
            on_progress(Progress::Fetching { url: url.clone() });
            let (title, path) = media::fetch(url, spec.mode.wants_video_fetch(), work.path())
                .context("yt-dlp download failed")?;
            (path, title, true)
        }
        Input::Path(p) => {
            if !p.exists() {
                bail!("input file not found: {}", p.display());
            }
            let stem = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "output".into());
            (p.clone(), stem, false)
        }
    };

    let source_has_video = media::has_video_stream(&media_path)?;
    let default_dir = if is_url {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    } else {
        media_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    };
    let plan = Plan::resolve(
        spec.mode,
        spec.keep,
        &spec.audio_format,
        spec.container,
        spec.output.clone(),
        source_has_video,
        &title,
        &default_dir,
    )?;
    let audio_codec = spec
        .audio_codec
        .unwrap_or_else(|| spec.container.default_codec());

    // Decode and load the model concurrently — independent costs.
    let t_decode = Instant::now();
    let decode_handle = std::thread::spawn({
        let media_path = media_path.clone();
        move || media::decode(&media_path)
    });

    let cache_str = spec
        .cache_dir
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());

    let t_model = Instant::now();
    let mut model = Demucs::new(
        &model_dir.to_string_lossy(),
        &spec.device,
        cache_str.as_deref(),
    )
    .context("initializing OpenVINO model")?;
    on_progress(Progress::ModelReady {
        device: spec.device.clone(),
        elapsed_secs: t_model.elapsed().as_secs_f32(),
        devices: model.available_devices()?,
        cache: cache_str
            .as_deref()
            .unwrap_or("off")
            .to_string(),
    });

    let audio = decode_handle
        .join()
        .map_err(|_| anyhow::anyhow!("audio-decode thread panicked"))?
        .context("decoding audio with ffmpeg")?;
    let audio_secs = audio.len() as f32 / SAMPLE_RATE as f32;
    on_progress(Progress::Decoded {
        audio_secs,
        elapsed_secs: t_decode.elapsed().as_secs_f32(),
    });

    let t = Instant::now();
    let stems = model.separate(&audio, spec.shifts, &mut |frac| {
        on_progress(Progress::Separating(frac));
    })?;
    let elapsed = t.elapsed().as_secs_f32();
    on_progress(Progress::Separated {
        elapsed_secs: elapsed,
        realtime: audio_secs as f64 / elapsed as f64,
    });

    let levels = stems.levels();
    on_progress(Progress::StemLevels(levels));

    if spec.keep_temp {
        media::write_wav(&work.path().join("music.wav"), &stems.music)
            .context("writing music stem")?;
        media::write_wav(&work.path().join("sfx.wav"), &stems.sfx)
            .context("writing sfx stem")?;
        media::write_wav(&work.path().join("speech.wav"), &stems.speech)
            .context("writing speech stem")?;
    }

    let kept = stems.keep(spec.keep).limit_peak();

    if plan.out_video {
        media::remux(
            &media_path,
            &kept,
            &plan.out_path,
            &RemuxOpts {
                container: spec.container,
                codec: audio_codec,
                bitrate: &spec.audio_bitrate,
            },
        )
        .context("remuxing video")?;
    } else {
        media::encode(&kept, &plan.out_path, &spec.audio_format, &spec.audio_bitrate)
            .context("encoding audio")?;
    }

    if spec.keep_temp {
        on_progress(Progress::KeepingTemp(work.path().to_path_buf()));
        work.keep();
    }

    Ok(plan.out_path)
}

/// Look for the model next to the executable first (portable package), then fall back
/// to the build-time project dir (dev runs via `cargo run`).
fn default_model_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("models").join("htdemucs_dnr");
            if p.join(demucs::MODEL_XML).exists() {
                return p;
            }
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("htdemucs_dnr")
}

struct TempDir {
    path: PathBuf,
    keep: bool,
}

impl TempDir {
    fn new() -> Result<Self> {
        let p = std::env::temp_dir().join(format!("music-remover-{}", std::process::id()));
        std::fs::create_dir_all(&p)?;
        Ok(Self {
            path: p,
            keep: false,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn keep(&mut self) {
        self.keep = true;
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
