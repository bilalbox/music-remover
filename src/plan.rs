// Decides what to produce — video vs audio, where to write — from the parsed CLI flags
// plus facts learned while resolving the input. Pure: no filesystem or process I/O.
// Stem selection lives on StemSet, not here.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::ValueEnum;

use crate::stems::Keep;

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Video out if the source has a video stream, else audio.
    Auto,
    /// Always produce a remuxed video (fails if there's no video stream).
    Video,
    /// Always produce an audio file.
    Audio,
}

impl Mode {
    /// Whether a source fetch (e.g. yt-dlp) should ask for a video stream at all. `Auto`
    /// still wants one — whether we end up *producing* video depends on whether the source
    /// actually has one, decided later in `Plan::resolve`.
    pub fn wants_video_fetch(self) -> bool {
        self != Mode::Audio
    }
}

/// Mux container for remuxed video output.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Container {
    Mp4,
    Webm,
}

impl Container {
    pub fn ext(self) -> &'static str {
        match self {
            Container::Mp4 => "mp4",
            Container::Webm => "webm",
        }
    }

    /// Default lossy codec for this container when `--audio-codec` is omitted.
    pub fn default_codec(self) -> AudioCodec {
        match self {
            Container::Mp4 => AudioCodec::Aac,
            Container::Webm => AudioCodec::Opus,
        }
    }
}

/// Lossy audio codec used when remuxing cleaned audio onto a video stream.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum AudioCodec {
    Aac,
    Opus,
}

impl AudioCodec {
    /// ffmpeg `-c:a` value.
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            AudioCodec::Aac => "aac",
            AudioCodec::Opus => "libopus",
        }
    }
}

pub struct Plan {
    pub out_video: bool,
    pub out_path: PathBuf,
}

impl Plan {
    /// `dir` is where a derived output path should land — the caller resolves that (the
    /// source file's parent for local input, the current directory for a URL fetch) since
    /// it's the one impure step; everything else here is a plain decision over its inputs.
    pub fn resolve(
        mode: Mode,
        keep: Keep,
        audio_format: &str,
        container: Container,
        explicit_output: Option<PathBuf>,
        source_has_video: bool,
        title: &str,
        dir: &Path,
    ) -> Result<Self> {
        let out_video = match mode {
            Mode::Audio => false,
            Mode::Video => {
                if !source_has_video {
                    bail!("--mode video requested but the source has no video stream");
                }
                true
            }
            Mode::Auto => source_has_video,
        };

        let out_path = explicit_output.unwrap_or_else(|| {
            let suffix = match keep {
                Keep::Soundtrack => "nomusic",
                Keep::Speech => "speech",
                Keep::Music => "music",
            };
            let ext = if out_video {
                container.ext().to_string()
            } else {
                audio_format.to_string()
            };
            dir.join(format!("{}.{suffix}.{ext}", sanitize(title)))
        });

        Ok(Self { out_video, out_path })
    }
}

pub fn sanitize(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if r#"<>:"/\|?*"#.contains(c) || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    let s = s.trim().trim_matches('.').to_string();
    if s.is_empty() {
        "output".into()
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_auto_follows_source() {
        let p = Plan::resolve(
            Mode::Auto,
            Keep::Soundtrack,
            "wav",
            Container::Mp4,
            None,
            true,
            "Song",
            Path::new("/out"),
        )
        .unwrap();
        assert!(p.out_video);
        assert_eq!(p.out_path, Path::new("/out/Song.nomusic.mp4"));

        let p = Plan::resolve(
            Mode::Auto,
            Keep::Soundtrack,
            "wav",
            Container::Mp4,
            None,
            false,
            "Song",
            Path::new("/out"),
        )
        .unwrap();
        assert!(!p.out_video);
        assert_eq!(p.out_path, Path::new("/out/Song.nomusic.wav"));
    }

    #[test]
    fn resolve_webm_container_extension() {
        let p = Plan::resolve(
            Mode::Auto,
            Keep::Soundtrack,
            "wav",
            Container::Webm,
            None,
            true,
            "Song",
            Path::new("/out"),
        )
        .unwrap();
        assert_eq!(p.out_path, Path::new("/out/Song.nomusic.webm"));
    }

    #[test]
    fn resolve_video_without_source_video_errors() {
        let err = Plan::resolve(
            Mode::Video,
            Keep::Soundtrack,
            "wav",
            Container::Mp4,
            None,
            false,
            "Song",
            Path::new("/out"),
        );
        assert!(err.is_err());
    }

    #[test]
    fn resolve_keep_suffixes_and_explicit_output_wins() {
        let p = Plan::resolve(
            Mode::Audio,
            Keep::Music,
            "mp3",
            Container::Mp4,
            None,
            true,
            "Song",
            Path::new("/out"),
        )
        .unwrap();
        assert_eq!(p.out_path, Path::new("/out/Song.music.mp3"));

        let p = Plan::resolve(
            Mode::Audio,
            Keep::Speech,
            "wav",
            Container::Mp4,
            None,
            false,
            "Song",
            Path::new("/out"),
        )
        .unwrap();
        assert_eq!(p.out_path, Path::new("/out/Song.speech.wav"));

        let explicit = PathBuf::from("/custom/name.wav");
        let p = Plan::resolve(
            Mode::Audio,
            Keep::Soundtrack,
            "wav",
            Container::Mp4,
            Some(explicit.clone()),
            true,
            "Song",
            Path::new("/out"),
        )
        .unwrap();
        assert_eq!(p.out_path, explicit);
    }

    #[test]
    fn sanitize_strips_reserved_chars() {
        assert_eq!(sanitize("a/b:c*d"), "a_b_c_d");
        assert_eq!(sanitize("  ."), "output");
    }
}
