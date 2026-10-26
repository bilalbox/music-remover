// music-remover: strip background music from a local media file or a URL (YouTube, etc.)
// using HTDemucs trained on Divide and Remaster, on the Intel Arc GPU via OpenVINO.
//
// Examples:
//   music-remover song.mp4                      # -> song.nomusic.mp4 (speech + effects)
//   music-remover lecture.m4a                    # -> lecture.nomusic.wav (audio only)
//   music-remover https://youtu.be/<id>          # download, clean, -> <title>.nomusic.mp4
//   music-remover clip.mp4 --keep speech         # dialogue only
//   music-remover clip.mp4 --keep music          # score only
//   music-remover clip.mp4 --mode audio --audio-format mp3
//   music-remover clip.mp4 --audio-bitrate 128k
//   music-remover clip.webm --container webm --audio-codec opus

use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

mod demucs;
mod job;
mod media;
mod plan;
mod stems;
mod stft;

use job::{Input, JobSpec, Progress};
use plan::{AudioCodec, Container, Mode};
use stems::Keep;

#[derive(Parser)]
#[command(
    name = "music-remover",
    about = "Remove background music, keeping speech and sound effects (HTDemucs-DnR on the Intel Arc GPU)"
)]
struct Cli {
    /// Local media file, or a URL (YouTube, etc.)
    input: String,

    /// Output file path (default: derived from the input/title next to it)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// What to produce.
    #[arg(long, value_enum, default_value_t = Mode::Auto)]
    mode: Mode,

    /// Which part to keep: soundtrack (speech + effects), speech, or music.
    #[arg(long, value_enum, default_value_t = Keep::Soundtrack)]
    keep: Keep,

    /// Audio codec/container for audio output (e.g. wav, mp3, m4a, flac, opus).
    #[arg(long, default_value = "wav")]
    audio_format: String,

    /// Video output container. `webm` pairs well with Opus and yt-dlp WebM sources.
    #[arg(long, value_enum, default_value_t = Container::Mp4)]
    container: Container,

    /// Lossy codec when remuxing onto video. Defaults to AAC for mp4, Opus for webm.
    #[arg(long, value_enum)]
    audio_codec: Option<AudioCodec>,

    /// Target bitrate for lossy audio encode/remux (e.g. 192k, 128k).
    #[arg(long, default_value = "192k")]
    audio_bitrate: String,

    /// Inference device.
    #[arg(long, default_value = "GPU")]
    device: String,

    /// Demucs shifts — higher is slightly better but proportionally slower.
    #[arg(long, default_value_t = 1)]
    shifts: usize,

    /// Path to the htdemucs_dnr model directory.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Enable OpenVINO's compiled-model cache in this directory. Off by default — the Intel
    /// GPU driver already caches compiled kernels, so this mainly helps guarantee a fast
    /// first launch on machines where that driver cache is cold or gets cleared.
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Keep intermediate temp files (downloaded media, raw stems).
    #[arg(long)]
    keep_temp: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let spec = JobSpec {
        input: Input::parse(&cli.input),
        mode: cli.mode,
        keep: cli.keep,
        audio_format: cli.audio_format,
        container: cli.container,
        audio_codec: cli.audio_codec,
        audio_bitrate: cli.audio_bitrate,
        output: cli.output,
        device: cli.device,
        shifts: cli.shifts,
        model: cli.model,
        cache_dir: cli.cache_dir,
        keep_temp: cli.keep_temp,
    };

    let keep = spec.keep;
    let mut last_pct = -1i32;
    let out = job::run(spec, &mut |p| match p {
        Progress::Fetching { url } => println!(">> fetching {url}"),
        Progress::ModelReady {
            device,
            elapsed_secs,
            devices,
            cache,
        } => {
            println!(
                ">> htdemucs-dnr ready on {device} in {elapsed_secs:.1}s (devices: {}; cache: {cache})",
                devices.join(", ")
            );
        }
        Progress::Decoded {
            audio_secs,
            elapsed_secs,
        } => {
            println!(">> decoded {audio_secs:.1}s of audio in {elapsed_secs:.1}s");
        }
        Progress::Separating(frac) => {
            let pct = (frac * 100.0) as i32;
            if pct != last_pct {
                last_pct = pct;
                print!("\r>> separating... {pct:3}%");
                let _ = std::io::stdout().flush();
            }
        }
        Progress::Separated {
            elapsed_secs,
            realtime,
        } => {
            println!("\r>> separated in {elapsed_secs:.1}s ({realtime:.1}x realtime)      ");
        }
        Progress::StemLevels(levels) => {
            println!(
                ">> stems  music {:+.1} dB  sfx {:+.1} dB  speech {:+.1} dB",
                levels.music_db, levels.sfx_db, levels.speech_db
            );
            if keep == Keep::Soundtrack && levels.music_is_silent() {
                println!(
                    ">> note: music holds {:.1}% of energy, so the output matches the input. A second pass will not remove more score — the bed is already inside the speech or effects stem.",
                    levels.music_frac * 100.0
                );
            }
        }
        Progress::KeepingTemp(path) => {
            println!(">> keeping temp dir: {}", path.display());
        }
    })?;

    println!(">> done: {}", out.display());
    Ok(())
}
