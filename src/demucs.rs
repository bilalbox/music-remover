// Faithful Rust port of Intel's htdemucs.cpp pipeline around the STFT-stripped
// OpenVINO IR. Operates on pipeline-rate stereo. Returns a named StemSet
// (music, sfx, speech) from ZFTurbo's HTDemucs trained on Divide and Remaster.
//
// Pipeline per the reference:
//   separate -> normalize by the mono-mix mean/std
//     apply_model_1 -> random temporal shifts, averaged
//       apply_model_2 -> split into overlapping `segment`s, triangular overlap-add
//         apply_model_3 -> center-pad each chunk to one `segment`, trim back
//           run_segment -> spectral branch (STFT, magnitude, norm) + time branch,
//                          GPU infer, then de-norm + iSTFT, sum the two branches.
//
// Perf note: the STFT/iSTFT and pre/post-processing loops here and in stft.rs are
// single-threaded scalar Rust. On CPU inference (see README's Linux section) that work
// competes with OpenVINO's CPU plugin for the same cores — worth parallelizing (rayon)
// if the CPU baseline turns out too slow, but unmeasured whether it's actually the
// bottleneck vs. inference itself.

use anyhow::{Context, Result};
use num_complex::Complex32;
use openvino::{Core, DeviceType, ElementType, RwPropertyKey, Shape, Tensor};

use crate::stems::{StemSet, Stereo, SAMPLE_RATE};
use crate::stft::{reflect_pad, Stft};

const SR: usize = SAMPLE_RATE as usize;
const SEGMENT: usize = 382788; // int((217/25) * 44100), the DnR training length
const NFFT: usize = 4096;
const HOP: usize = 1024;
const FR: usize = 2048; // kept freq bins (2049 - 1)
const T: usize = 374; // kept time frames (ceil(SEGMENT / HOP))
const NS: usize = 3; // sources: music, sfx, speech
const NC: usize = 2; // channels
const C4: usize = NC * 2; // model's spectral output packs (re, im) per channel

/// Filename of the model's OpenVINO IR graph, relative to the model dir. `.bin` shares the stem.
pub const MODEL_XML: &str = "htdemucs_fwd.xml";

/// Internal channel-major stereo used throughout the inference pipeline.
type ChanStereo = [Vec<f32>; NC];
/// Internal sources, each ChanStereo, order [music, sfx, speech].
type ChanStems = Vec<ChanStereo>;

fn mean(x: &[f32]) -> f32 {
    x.iter().copied().sum::<f32>() / x.len() as f32
}

/// Unbiased (N-1) standard deviation, matching torch's default.
fn std_unbiased(x: &[f32], m: f32) -> f32 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let var = x.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / (n - 1) as f32;
    var.sqrt()
}

pub struct Demucs {
    core: Core,
    // CompiledModel + InferRequest own independent C handles; keep both alive.
    _compiled: openvino::CompiledModel,
    req: openvino::InferRequest,
    stft: Stft,
}

fn pick_device(s: &str) -> DeviceType<'static> {
    match s {
        "CPU" => DeviceType::CPU,
        "GPU" => DeviceType::GPU,
        "NPU" => DeviceType::NPU,
        other => DeviceType::Other(other.to_string().into()),
    }
}

impl Demucs {
    pub fn new(model_dir: &str, device: &str, cache_dir: Option<&str>) -> Result<Self> {
        let xml = std::path::Path::new(model_dir).join(MODEL_XML);
        let bin = xml.with_extension("bin");
        let xml = xml.to_string_lossy().into_owned();
        let bin = bin.to_string_lossy().into_owned();
        let mut core = Core::new().context("Core::new")?;

        // Persist the device-compiled graph so subsequent runs skip the JIT compile.
        if let Some(cd) = cache_dir {
            let _ = std::fs::create_dir_all(cd);
            if let Err(e) =
                core.set_property(&pick_device(device), &RwPropertyKey::CacheDir, cd)
            {
                eprintln!("warning: could not enable model cache at {cd}: {e}");
            }
        }

        let model = core
            .read_model_from_file(&xml, &bin)
            .context("read_model_from_file")?;
        let mut compiled = core
            .compile_model(&model, pick_device(device))
            .with_context(|| format!("compile_model on {device}"))?;
        let req = compiled.create_infer_request().context("create_infer_request")?;
        Ok(Self {
            core,
            _compiled: compiled,
            req,
            stft: Stft::new(NFFT, HOP),
        })
    }

    #[allow(deprecated)] // GNA variant is deprecated upstream but the match must be exhaustive
    pub fn available_devices(&self) -> Result<Vec<String>> {
        Ok(self
            .core
            .available_devices()?
            .iter()
            .map(|d| match d {
                DeviceType::CPU => "CPU".to_string(),
                DeviceType::GPU => "GPU".to_string(),
                DeviceType::NPU => "NPU".to_string(),
                DeviceType::GNA => "GNA".to_string(),
                DeviceType::Other(s) => s.to_string(),
            })
            .collect())
    }

    /// Top-level: separate a full stereo track into a named **StemSet**. `on_progress` is
    /// called with a 0..1 fraction as segments complete.
    pub fn separate(
        &mut self,
        audio: &Stereo,
        shifts: usize,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<StemSet> {
        let n = audio.len();
        let left = &audio.left;
        let right = &audio.right;

        // Normalize by the mono mix's mean/std (ref).
        let refm: Vec<f32> = (0..n).map(|i| 0.5 * (left[i] + right[i])).collect();
        let ref_mean = mean(&refm);
        let ref_std = std_unbiased(&refm, ref_mean).max(1e-8);

        let cmix: ChanStereo = [
            left.iter().map(|&v| (v - ref_mean) / ref_std).collect(),
            right.iter().map(|&v| (v - ref_mean) / ref_std).collect(),
        ];

        let shifts = shifts.max(1);
        let stride = ((1.0 - 0.25) * SEGMENT as f32) as usize;
        let total = (shifts * (n.div_ceil(stride) + 1)).max(1);
        let mut done = 0usize;
        let mut sources = {
            let mut seg_cb = || {
                done += 1;
                on_progress((done as f32 / total as f32).min(1.0));
            };
            self.apply_model_shifts(&cmix, shifts, &mut seg_cb)?
        };

        // De-normalize.
        for s in 0..NS {
            for c in 0..NC {
                for v in sources[s][c].iter_mut() {
                    *v = *v * ref_std + ref_mean;
                }
            }
        }
        Ok(stem_set_from_chans(sources))
    }

    // apply_model_1: temporal-shift augmentation, averaged.
    fn apply_model_shifts(
        &mut self,
        mix: &ChanStereo,
        shifts: usize,
        on_seg: &mut dyn FnMut(),
    ) -> Result<ChanStems> {
        let length = mix[0].len();
        let max_shift = SR / 2; // 22050

        // Zero-pad max_shift each side.
        let padded: ChanStereo = [
            pad_zeros(&mix[0], max_shift, max_shift),
            pad_zeros(&mix[1], max_shift, max_shift),
        ];

        let mut accum: ChanStems = (0..NS)
            .map(|_| [vec![0.0f32; length], vec![0.0f32; length]])
            .collect();

        for i in 0..shifts {
            // Deterministic offset (max_shift/2) for the first shift keeps single-shift
            // runs reproducible; subsequent shifts dither.
            let offset = if shifts == 1 || i == 0 {
                max_shift / 2
            } else {
                fastrand_usize(max_shift)
            };
            let shifted_len = length + max_shift - offset;
            let shifted: ChanStereo = [
                padded[0][offset..offset + shifted_len].to_vec(),
                padded[1][offset..offset + shifted_len].to_vec(),
            ];

            let so = self.apply_model_split(&shifted, on_seg)?;
            let trim = max_shift - offset;
            for s in 0..NS {
                for c in 0..NC {
                    for k in 0..length {
                        accum[s][c][k] += so[s][c][trim + k];
                    }
                }
            }
        }

        let inv = 1.0 / shifts as f32;
        for s in 0..NS {
            for c in 0..NC {
                for v in accum[s][c].iter_mut() {
                    *v *= inv;
                }
            }
        }
        Ok(accum)
    }

    // apply_model_2: overlapping segments + triangular overlap-add.
    fn apply_model_split(
        &mut self,
        mix: &ChanStereo,
        on_seg: &mut dyn FnMut(),
    ) -> Result<ChanStems> {
        let length = mix[0].len();
        let overlap = 0.25;
        let stride = ((1.0 - overlap) * SEGMENT as f32) as usize; // 287091

        let weight = triangle_weight();

        let mut out: ChanStems = (0..NS)
            .map(|_| [vec![0.0f32; length], vec![0.0f32; length]])
            .collect();
        let mut sum_weight = vec![0.0f32; length];

        let mut off = 0usize;
        while off < length {
            let chunk_length = SEGMENT.min(length - off);
            let seg = padded_segment(mix, off, chunk_length, SEGMENT);
            let src = self.run_segment(&seg)?;

            // center_trim back to chunk_length.
            let delta = SEGMENT - chunk_length;
            let s0 = delta / 2;
            for s in 0..NS {
                for c in 0..NC {
                    let dst = &mut out[s][c];
                    let srcc = &src[s][c];
                    for k in 0..chunk_length {
                        dst[off + k] += weight[k] * srcc[s0 + k];
                    }
                }
            }
            for k in 0..chunk_length {
                sum_weight[off + k] += weight[k];
            }
            off += stride;
            on_seg();
        }

        for s in 0..NS {
            for c in 0..NC {
                for i in 0..length {
                    out[s][c][i] /= sum_weight[i].max(1e-12);
                }
            }
        }
        Ok(out)
    }

    // apply_model_3 + actually_run_model: one fixed-size SEGMENT through the model.
    fn run_segment(&mut self, mix: &ChanStereo) -> Result<ChanStems> {
        debug_assert_eq!(mix[0].len(), SEGMENT);

        // ---- spectral branch input ----
        let pad_l = HOP / 2 * 3; // 1536
        let le = (SEGMENT as f32 / HOP as f32).ceil() as usize; // 374
        let pad_r = pad_l + le * HOP - SEGMENT; // 1724

        // STFT each channel, keep [0..FR) freq bins and frames [2 .. 2+T).
        let mut z_keep = [
            vec![Complex32::new(0.0, 0.0); FR * T],
            vec![Complex32::new(0.0, 0.0); FR * T],
        ];
        for c in 0..NC {
            let xp = reflect_pad(&mix[c], pad_l, pad_r);
            let (z, n_frames) = self.stft.forward(&xp);
            // z layout: [f * n_frames + t]; keep f in 0..FR, t in 2..2+T.
            for f in 0..FR {
                for t in 0..T {
                    z_keep[c][f * T + t] = z[f * n_frames + (t + 2)];
                }
            }
        }

        // magnitude pack -> [C4, FR, T], channel-major then (real, imag).
        let n_spec = C4 * FR * T; // 4 * 2048 * 374
        let mut xin = vec![0.0f32; n_spec];
        for c in 0..NC {
            for f in 0..FR {
                for t in 0..T {
                    let z = z_keep[c][f * T + t];
                    xin[(c * 2) * (FR * T) + f * T + t] = z.re;
                    xin[(c * 2 + 1) * (FR * T) + f * T + t] = z.im;
                }
            }
        }
        let xin_mean = mean(&xin);
        let xin_std = std_unbiased(&xin, xin_mean);
        let denom = 1e-5 + xin_std;
        for v in xin.iter_mut() {
            *v = (*v - xin_mean) / denom;
        }

        // ---- time branch input ----
        let mut xt = vec![0.0f32; NC * SEGMENT];
        for c in 0..NC {
            xt[c * SEGMENT..(c + 1) * SEGMENT].copy_from_slice(&mix[c]);
        }
        let xt_mean = mean(&xt);
        let xt_std = std_unbiased(&xt, xt_mean);
        let denomt = 1e-5 + xt_std;
        for v in xt.iter_mut() {
            *v = (*v - xt_mean) / denomt;
        }

        // ---- inference ----
        let mut tx =
            Tensor::new(ElementType::F32, &Shape::new(&[1, C4 as i64, FR as i64, T as i64])?)?;
        tx.get_data_mut::<f32>()?.copy_from_slice(&xin);
        let mut txt =
            Tensor::new(ElementType::F32, &Shape::new(&[1, NC as i64, SEGMENT as i64])?)?;
        txt.get_data_mut::<f32>()?.copy_from_slice(&xt);
        self.req.set_tensor("x", &tx)?;
        self.req.set_tensor("xt", &txt)?;
        self.req.infer().context("infer")?;
        let x_out = self.req.get_tensor("x_out")?;
        let xt_out = self.req.get_tensor("xt_out")?;
        let x_out = x_out.get_data::<f32>()?; // [3, 4, FR, T]
        let xt_out = xt_out.get_data::<f32>()?; // [3, 2, SEGMENT]

        // ---- spectral post: de-norm, build complex, iSTFT ----
        // x_out viewed [S, C4=4, FR, T]; de-norm with the SAME mean/std as the input.
        let mut stems: ChanStems = (0..NS)
            .map(|_| [vec![0.0f32; SEGMENT], vec![0.0f32; SEGMENT]])
            .collect();

        // padded spectrogram has freq 2049 and frames 378 (zeros around the kept region).
        let n_freq_full = NFFT / 2 + 1; // 2049
        let n_frames_full = T + 4; // 378
        let ispec_len = HOP * le + 2 * pad_l; // 386048

        // Reused across every (s, c) below: only the [0..FR) x [2..2+T) region is ever
        // written, so the border stays zero and doesn't need re-clearing between iterations.
        let mut zfull = vec![Complex32::new(0.0, 0.0); n_freq_full * n_frames_full];
        for s in 0..NS {
            for c in 0..NC {
                for f in 0..FR {
                    for t in 0..T {
                        let re = x_out[s * (C4 * FR * T) + (c * 2) * (FR * T) + f * T + t] * xin_std
                            + xin_mean;
                        let im = x_out[s * (C4 * FR * T) + (c * 2 + 1) * (FR * T) + f * T + t]
                            * xin_std
                            + xin_mean;
                        // freq f stays (top bin 2048 left zero); frames shifted by +2.
                        zfull[f * n_frames_full + (t + 2)] = Complex32::new(re, im);
                    }
                }
                let wave = self.stft.inverse(&zfull, n_frames_full, ispec_len);
                // crop [pad_l .. pad_l + SEGMENT]
                let spec_wave = &wave[pad_l..pad_l + SEGMENT];

                // time branch + sum
                let dst = &mut stems[s][c];
                for i in 0..SEGMENT {
                    let tw = xt_out[s * (NC * SEGMENT) + c * SEGMENT + i] * xt_std + xt_mean;
                    dst[i] = tw + spec_wave[i];
                }
            }
        }
        Ok(stems)
    }
}

/// Triangular overlap-add weight of length SEGMENT, peak normalized to 1 (transition_power=1).
fn triangle_weight() -> Vec<f32> {
    let half = SEGMENT / 2; // 191394
    let mut w = Vec::with_capacity(SEGMENT);
    for k in 1..=half {
        w.push(k as f32);
    }
    for k in (1..=(SEGMENT - half)).rev() {
        w.push(k as f32);
    }
    let max = half as f32;
    for v in w.iter_mut() {
        *v /= max;
    }
    w
}

fn pad_zeros(x: &[f32], left: usize, right: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; left];
    out.extend_from_slice(x);
    out.extend(std::iter::repeat(0.0f32).take(right));
    out
}

/// Center-pad a chunk starting at `off` (length `chunk_length`) within `mix` out to
/// `valid` samples, pulling real neighbouring samples where available, zero elsewhere.
fn padded_segment(mix: &ChanStereo, off: usize, chunk_length: usize, valid: usize) -> ChanStereo {
    let total = mix[0].len() as i64;
    let delta = (valid - chunk_length) as i64;
    let start = off as i64 - delta / 2;
    let end = start + valid as i64;
    let correct_start = start.max(0);
    let correct_end = end.min(total);
    let pad_left = (correct_start - start) as usize;
    let pad_right = (end - correct_end) as usize;

    let mut out: ChanStereo = [Vec::with_capacity(valid), Vec::with_capacity(valid)];
    for c in 0..NC {
        out[c].resize(pad_left, 0.0);
        out[c].extend_from_slice(&mix[c][correct_start as usize..correct_end as usize]);
        out[c].extend(std::iter::repeat(0.0f32).take(pad_right));
        debug_assert_eq!(out[c].len(), valid);
    }
    out
}

fn stem_set_from_chans(mut sources: ChanStems) -> StemSet {
    debug_assert_eq!(sources.len(), NS);
    // Model order is [music, sfx, speech] — pop from the end.
    let [speech_l, speech_r] = sources.pop().expect("3 stems");
    let [sfx_l, sfx_r] = sources.pop().expect("3 stems");
    let [music_l, music_r] = sources.pop().expect("3 stems");
    StemSet {
        music: Stereo {
            left: music_l,
            right: music_r,
        },
        sfx: Stereo {
            left: sfx_l,
            right: sfx_r,
        },
        speech: Stereo {
            left: speech_l,
            right: speech_r,
        },
    }
}

// Tiny xorshift RNG so we don't pull in the `rand` crate just for shift dithering.
use std::cell::Cell;
thread_local! {
    static RNG: Cell<u64> = const { Cell::new(0x9E3779B97F4A7C15) };
}
fn fastrand_usize(bound: usize) -> usize {
    RNG.with(|r| {
        let mut x = r.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        r.set(x);
        (x % bound as u64) as usize
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_basic() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
    }

    #[test]
    fn std_unbiased_known_value() {
        // [1,2,3]: mean 2, unbiased var = ((1)^2+0+(1)^2)/(3-1) = 1.0 -> std 1.0
        assert!((std_unbiased(&[1.0, 2.0, 3.0], 2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn std_unbiased_degenerate_n_lt_2_is_zero() {
        assert_eq!(std_unbiased(&[], 0.0), 0.0);
        assert_eq!(std_unbiased(&[5.0], 5.0), 0.0);
    }

    #[test]
    fn pad_zeros_shape_and_values() {
        let out = pad_zeros(&[1.0, 2.0, 3.0], 2, 1);
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn triangle_weight_shape() {
        let w = triangle_weight();
        assert_eq!(w.len(), SEGMENT);
        let half = SEGMENT / 2;
        // Peak sits on the two center samples (even-length triangle has no single center).
        assert!((w[half - 1] - 1.0).abs() < 1e-6);
        assert!((w[half] - 1.0).abs() < 1e-6);
        // Ramps up from near-zero at the start and back down to near-zero at the end.
        assert!(w[0] > 0.0 && w[0] < 0.01);
        assert!(w[SEGMENT - 1] > 0.0 && w[SEGMENT - 1] < 0.01);
        // Symmetric.
        assert!((w[0] - w[SEGMENT - 1]).abs() < 1e-6);
    }

    fn stereo10() -> ChanStereo {
        let ch: Vec<f32> = (0..10).map(|i| i as f32).collect();
        [ch.clone(), ch]
    }

    #[test]
    fn padded_segment_interior_no_padding() {
        let out = padded_segment(&stereo10(), 3, 4, 6);
        assert_eq!(out[0], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn padded_segment_pads_left_at_track_start() {
        let out = padded_segment(&stereo10(), 0, 4, 6);
        assert_eq!(out[0], vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn padded_segment_pads_right_at_track_end() {
        let out = padded_segment(&stereo10(), 6, 4, 6);
        assert_eq!(out[0], vec![5.0, 6.0, 7.0, 8.0, 9.0, 0.0]);
    }
}
