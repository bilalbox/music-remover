// STFT / iSTFT matching the conventions htdemucs (torch.stft/torch.istft) uses:
//   center = true, pad_mode = "reflect", onesided = true, periodic Hann window.
//
// torch's `normalized` flag is intentionally NOT applied: htdemucs mean/std-normalizes
// the magnitude spectrogram before the model and de-normalizes after, which cancels any
// global STFT scale factor. All that matters is that `inverse` is the exact inverse of
// `forward`, so we implement a standard analysis/synthesis pair with windowed overlap-add
// and NOLA (window^2) normalization on the synthesis side.

use std::sync::Arc;

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

pub struct Stft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    // Scratch buffers for `inverse`, reused across calls (same size every time in practice)
    // to avoid a fresh alloc + zero-fill per call.
    full_scratch: Vec<f32>,
    wsum_scratch: Vec<f32>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        // torch.hann_window(n_fft, periodic=true): w[n] = 0.5 - 0.5*cos(2*pi*n / n_fft)
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos())
            .collect();
        let mut planner = FftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n_fft);
        let inv = planner.plan_fft_inverse(n_fft);
        Self { n_fft, hop, window, fwd, inv, full_scratch: Vec::new(), wsum_scratch: Vec::new() }
    }

    pub fn n_freq(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Forward STFT of a 1-D signal. Returns complex coefficients laid out as
    /// `[freq * n_frames + frame]`, plus `n_frames`.
    pub fn forward(&self, x: &[f32]) -> (Vec<Complex32>, usize) {
        let pad = self.n_fft / 2;
        let xp = reflect_pad(x, pad, pad);
        let n_frames = 1 + (xp.len() - self.n_fft) / self.hop;
        let n_freq = self.n_freq();

        let mut out = vec![Complex32::new(0.0, 0.0); n_freq * n_frames];
        let mut buf = vec![Complex32::new(0.0, 0.0); self.n_fft];
        for t in 0..n_frames {
            let base = t * self.hop;
            for j in 0..self.n_fft {
                buf[j] = Complex32::new(xp[base + j] * self.window[j], 0.0);
            }
            self.fwd.process(&mut buf);
            for f in 0..n_freq {
                out[f * n_frames + t] = buf[f];
            }
        }
        (out, n_frames)
    }

    /// Inverse STFT. `z` is `[freq * n_frames + frame]` with `n_freq = n_fft/2+1` rows.
    /// Returns exactly `length` samples (after the center trim of `n_fft/2` each side).
    pub fn inverse(&mut self, z: &[Complex32], n_frames: usize, length: usize) -> Vec<f32> {
        let n_freq = self.n_freq();
        let full_len = (n_frames - 1) * self.hop + self.n_fft;
        self.full_scratch.clear();
        self.full_scratch.resize(full_len, 0.0);
        self.wsum_scratch.clear();
        self.wsum_scratch.resize(full_len, 0.0);
        let mut buf = vec![Complex32::new(0.0, 0.0); self.n_fft];

        let inv_n = 1.0 / self.n_fft as f32;
        for t in 0..n_frames {
            // Rebuild the full Hermitian-symmetric spectrum from the one-sided half.
            for f in 0..n_freq {
                buf[f] = z[f * n_frames + t];
            }
            for f in n_freq..self.n_fft {
                buf[f] = z[(self.n_fft - f) * n_frames + t].conj();
            }
            self.inv.process(&mut buf);
            let base = t * self.hop;
            for j in 0..self.n_fft {
                let s = buf[j].re * inv_n;
                self.full_scratch[base + j] += s * self.window[j];
                self.wsum_scratch[base + j] += self.window[j] * self.window[j];
            }
        }
        for i in 0..full_len {
            if self.wsum_scratch[i] > 1e-8 {
                self.full_scratch[i] /= self.wsum_scratch[i];
            }
        }
        let start = self.n_fft / 2;
        self.full_scratch[start..start + length].to_vec()
    }
}

/// torch-style reflect padding (does not repeat the edge sample).
pub(crate) fn reflect_pad(x: &[f32], left: usize, right: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + left + right);
    for k in (1..=left).rev() {
        out.push(x[k]);
    }
    out.extend_from_slice(x);
    for k in 1..=right {
        out.push(x[n - 1 - k]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_matches_torch() {
        let x = [10.0, 20.0, 30.0, 40.0, 50.0];
        // left=2 -> x[2], x[1]; right=2 -> x[3], x[2]
        let p = reflect_pad(&x, 2, 2);
        assert_eq!(p, vec![30.0, 20.0, 10.0, 20.0, 30.0, 40.0, 50.0, 40.0, 30.0]);
    }

    #[test]
    fn stft_istft_roundtrip() {
        // A signal long enough that the COLA region is well inside the edges.
        let n_fft = 4096;
        let hop = 1024;
        let mut s = Stft::new(n_fft, hop);
        let len = 64 * hop; // multiple of hop so inverse length lines up
        let x: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / 44100.0;
                0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
            })
            .collect();

        let (z, n_frames) = s.forward(&x);
        let y = s.inverse(&z, n_frames, len);
        assert_eq!(y.len(), len);

        // Compare the interior (skip one frame of edge on each side).
        let edge = n_fft;
        let mut max_err = 0.0f32;
        for i in edge..len - edge {
            max_err = max_err.max((x[i] - y[i]).abs());
        }
        assert!(max_err < 1e-3, "max reconstruction error {max_err} too large");
    }
}
