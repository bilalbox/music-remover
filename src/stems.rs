// Stem domain types: named stems from a separation, Keep policy, and stereo buffers
// at the pipeline sample rate.

use clap::ValueEnum;

/// Sample rate every decode / Demucs / WAV write path agrees on.
pub const SAMPLE_RATE: u32 = 44100;

#[derive(Clone, Debug)]
pub struct Stereo {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl Stereo {
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.left.len(), self.right.len());
        self.left.len()
    }

    /// Mean-square over both channels. Used for stem-energy comparisons.
    pub fn mean_square(&self) -> f64 {
        let n = (self.left.len() + self.right.len()) as f64;
        if n == 0.0 {
            return 0.0;
        }
        let sum: f64 = self
            .left
            .iter()
            .chain(self.right.iter())
            .map(|&v| (v as f64) * (v as f64))
            .sum();
        sum / n
    }

    /// RMS level in dBFS. Silence is clamped via a 1e-12 mean-square floor (−120 dB).
    pub fn rms_db(&self) -> f32 {
        let ms = self.mean_square().max(1e-12);
        (10.0 * ms.log10()) as f32
    }

    /// Scale down if the peak would clip on write.
    pub fn limit_peak(mut self) -> Self {
        let mut peak = 0.0f32;
        for &v in self.left.iter().chain(self.right.iter()) {
            peak = peak.max(v.abs());
        }
        if peak > 0.999 {
            let g = 0.999 / peak;
            for v in self.left.iter_mut().chain(self.right.iter_mut()) {
                *v *= g;
            }
        }
        self
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum Keep {
    /// Drop the score. Keep speech and sound effects. The default.
    Soundtrack,
    /// Dialogue only.
    Speech,
    /// Score only.
    Music,
}

/// The three stems from one HTDemucs-DnR separation, in model order.
#[derive(Clone, Debug)]
pub struct StemSet {
    pub music: Stereo,
    pub sfx: Stereo,
    pub speech: Stereo,
}

impl StemSet {
    pub fn speech(self) -> Stereo {
        self.speech
    }

    pub fn music(self) -> Stereo {
        self.music
    }

    /// Speech plus sound effects — the soundtrack with the score removed.
    pub fn soundtrack(self) -> Stereo {
        add(&self.speech, &self.sfx)
    }

    pub fn keep(self, keep: Keep) -> Stereo {
        match keep {
            Keep::Soundtrack => self.soundtrack(),
            Keep::Speech => self.speech(),
            Keep::Music => self.music(),
        }
    }

    /// Per-stem RMS levels plus the fraction of energy in the music stem.
    /// When that fraction is ~0, `Keep::Soundtrack` is a no-op (output ≈ mix).
    pub fn levels(&self) -> StemLevels {
        let music = self.music.mean_square();
        let sfx = self.sfx.mean_square();
        let speech = self.speech.mean_square();
        let total = music + sfx + speech;
        StemLevels {
            music_db: self.music.rms_db(),
            sfx_db: self.sfx.rms_db(),
            speech_db: self.speech.rms_db(),
            music_frac: if total < 1e-20 {
                0.0
            } else {
                (music / total) as f32
            },
        }
    }
}

fn add(a: &Stereo, b: &Stereo) -> Stereo {
    let n = a.len();
    let mut left = vec![0.0f32; n];
    let mut right = vec![0.0f32; n];
    for i in 0..n {
        left[i] = a.left[i] + b.left[i];
        right[i] = a.right[i] + b.right[i];
    }
    Stereo { left, right }
}

/// Snapshot of StemSet energy after a separation, for progress/logging.
#[derive(Clone, Debug)]
pub struct StemLevels {
    pub music_db: f32,
    pub sfx_db: f32,
    pub speech_db: f32,
    /// Energy in the music stem as a fraction of all three stems.
    pub music_frac: f32,
}

impl StemLevels {
    /// Below this, the music stem is effectively silent and a second soundtrack
    /// pass cannot remove more score (the bed is already inside speech or effects).
    pub const SILENT_MUSIC: f32 = 0.02;

    pub fn music_is_silent(&self) -> bool {
        self.music_frac < Self::SILENT_MUSIC
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(v: f32) -> Stereo {
        Stereo {
            left: vec![v, v],
            right: vec![v, v],
        }
    }

    fn stems(vals: [f32; 3]) -> StemSet {
        StemSet {
            music: mono(vals[0]),
            sfx: mono(vals[1]),
            speech: mono(vals[2]),
        }
    }

    #[test]
    fn keep_soundtrack_sums_speech_and_sfx() {
        let out = stems([1.0, 2.0, 4.0]).keep(Keep::Soundtrack);
        assert_eq!(out.left[0], 6.0);
    }

    #[test]
    fn keep_speech_and_music_pick_their_stems() {
        assert_eq!(stems([1.0, 2.0, 4.0]).keep(Keep::Speech).left[0], 4.0);
        assert_eq!(stems([1.0, 2.0, 4.0]).keep(Keep::Music).left[0], 1.0);
    }

    #[test]
    fn named_methods_match_keep() {
        let a = stems([1.0, 2.0, 4.0]).soundtrack();
        let b = stems([1.0, 2.0, 4.0]).keep(Keep::Soundtrack);
        assert_eq!(a.left, b.left);
        let a = stems([1.0, 2.0, 4.0]).speech();
        let b = stems([1.0, 2.0, 4.0]).keep(Keep::Speech);
        assert_eq!(a.left, b.left);
        let a = stems([1.0, 2.0, 4.0]).music();
        let b = stems([1.0, 2.0, 4.0]).keep(Keep::Music);
        assert_eq!(a.left, b.left);
    }

    #[test]
    fn limit_peak_scales_when_hot() {
        let s = Stereo {
            left: vec![2.0],
            right: vec![-1.5],
        }
        .limit_peak();
        assert!((s.left[0].abs() - 0.999).abs() < 1e-5);
        assert!(s.right[0].abs() < 0.999 + 1e-5);
    }

    #[test]
    fn limit_peak_noop_when_safe() {
        let s = Stereo {
            left: vec![0.5],
            right: vec![-0.25],
        }
        .limit_peak();
        assert_eq!(s.left[0], 0.5);
        assert_eq!(s.right[0], -0.25);
    }

    #[test]
    fn rms_db_known_level() {
        // All samples 0.1 → mean-square 0.01 → 10*log10 = -20 dB.
        let s = Stereo {
            left: vec![0.1, 0.1],
            right: vec![0.1, 0.1],
        };
        assert!((s.rms_db() - (-20.0)).abs() < 1e-4);
    }

    #[test]
    fn levels_detect_silent_music() {
        let mixed = stems([0.01, 0.2, 1.0]).levels();
        assert!(mixed.music_is_silent());
        assert!(mixed.speech_db > mixed.music_db + 20.0);

        let split = stems([1.0, 1.0, 1.0]).levels();
        assert!(!split.music_is_silent());
        assert!((split.music_frac - (1.0 / 3.0)).abs() < 1e-5);
    }
}
