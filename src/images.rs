// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Image input: decoding, and the Qwen2-VL image processor (`Qwen2VLImageProcessor`) that turns
//! an image into the vision tower's `pixel_values`.
//!
//! An image is resized so both sides are multiples of `patch * merge` and its area lies within
//! the processor's `[min_pixels, max_pixels]` (HF's `smart_resize`), with PIL-style antialiased
//! bicubic resampling rounded back to 8 bits as HF's processors do, scaled to [0, 1],
//! normalised with the processor's mean / std and cut into `patch x patch` patches listed in
//! 2x2 merge-block order, each frame repeated over the temporal patch size.

use crate::config::VisionConfig;
use anyhow::{bail, Context, Result};
use base64::Engine as _;
use std::path::Path;

/// `preprocessor_config.json` of a multimodal checkpoint (defaults from the vision config).
#[derive(Clone, Debug)]
pub struct ImageProcessor {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// One image ready for the vision tower.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    /// `[t * grid_h * grid_w, C*T*P*P]`, row-major.
    pub pixels: Vec<f32>,
    pub grid: (usize, usize, usize),
    /// Hash of the decoded image's content (prefix-cache identity).
    pub hash: u64,
}

impl PreparedImage {
    /// Language-model tokens the image takes (one per merged 2x2 block of patches).
    pub fn tokens(&self, merge: usize) -> usize {
        self.grid.0 * self.grid.1 * self.grid.2 / (merge * merge)
    }
}

impl ImageProcessor {
    pub fn load(model_dir: &Path, vision: &VisionConfig) -> Self {
        let v: serde_json::Value = std::fs::read_to_string(model_dir.join("preprocessor_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let get = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let size = v.get("size");
        let edge = |k: &str| size.and_then(|s| s.get(k)).and_then(|x| x.as_u64()).map(|x| x as usize);
        let triple = |k: &str| -> [f32; 3] {
            v.get(k)
                .and_then(|x| x.as_array())
                .filter(|a| a.len() == 3)
                .map(|a| std::array::from_fn(|i| a[i].as_f64().unwrap_or(0.5) as f32))
                .unwrap_or([0.5; 3])
        };
        Self {
            min_pixels: edge("shortest_edge").or_else(|| get("min_pixels")).unwrap_or(56 * 56),
            max_pixels: edge("longest_edge").or_else(|| get("max_pixels")).unwrap_or(28 * 28 * 1280),
            patch_size: get("patch_size").unwrap_or(vision.patch_size),
            temporal_patch_size: get("temporal_patch_size").unwrap_or(vision.temporal_patch_size),
            merge_size: get("merge_size").unwrap_or(vision.spatial_merge_size),
            mean: triple("image_mean"),
            std: triple("image_std"),
        }
    }

    /// Decode and prepare one image (PNG, JPEG, WebP, GIF, BMP).
    pub fn prepare(&self, bytes: &[u8]) -> Result<PreparedImage> {
        let img = image::load_from_memory(bytes).context("decoding the image")?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        self.prepare_rgb(img.as_raw(), w, h)
    }

    /// Prepare an RGB8 image (`rgb` row-major, `3 * w * h` bytes).
    pub fn prepare_rgb(&self, rgb: &[u8], w: usize, h: usize) -> Result<PreparedImage> {
        let factor = self.patch_size * self.merge_size;
        let (rh, rw) = smart_resize(h, w, factor, self.min_pixels, self.max_pixels)?;
        let resized = resize_bicubic(rgb, w, h, rw, rh);
        let (p, m, t) = (self.patch_size, self.merge_size, self.temporal_patch_size);
        let (gh, gw) = (rh / p, rw / p);
        let dim = 3 * t * p * p;
        let mut pixels = Vec::with_capacity(gh * gw * dim);
        // [grid_h/m, grid_w/m, m, m] patches, each [C, T, P, P]
        for br in 0..gh / m {
            for bc in 0..gw / m {
                for ir in 0..m {
                    for ic in 0..m {
                        let (y0, x0) = ((br * m + ir) * p, (bc * m + ic) * p);
                        for c in 0..3 {
                            for _ in 0..t {
                                for y in 0..p {
                                    for x in 0..p {
                                        let v = resized[((y0 + y) * rw + x0 + x) * 3 + c] as f32 / 255.0;
                                        pixels.push((v - self.mean[c]) / self.std[c]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(PreparedImage { pixels, grid: (1, gh, gw), hash: fnv1a(rgb, w, h) })
    }
}

/// Placeholder ids of image tokens in a request's prompt: above every vocabulary (bit 31 set),
/// derived from the image's content hash, so prompts with different images never share
/// prefix-cache KV. [`model_token`] maps them back to `image_token_id` for the model.
pub fn pseudo_id(hash: u64, k: usize) -> u32 {
    let x = hash.wrapping_add((k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    0x8000_0000 | ((x ^ (x >> 32)) as u32 & 0x7fff_ffff)
}

/// The token the model sees: image placeholders become `image_token_id`.
pub fn model_token(t: u32, image_token_id: u32) -> u32 {
    if t & 0x8000_0000 != 0 {
        image_token_id
    } else {
        t
    }
}

/// An image's tokens in a prompt: `len` placeholders from `start`, `grid` the merged token grid
/// `(t, h, w)` (patch grid / merge size).
#[derive(Clone, Debug)]
pub struct ImageSpan {
    pub start: usize,
    pub len: usize,
    pub grid: (usize, usize, usize),
    pub image: std::sync::Arc<PreparedImage>,
}

/// Replace the k-th `image_token_id` of `tokens` with image k's placeholders (one per token
/// the tower makes of it). Fails when the counts differ.
pub fn expand_image_tokens(
    tokens: &[u32],
    image_token_id: u32,
    images: Vec<PreparedImage>,
    merge: usize,
) -> Result<(Vec<u32>, Vec<ImageSpan>)> {
    let slots = tokens.iter().filter(|&&t| t == image_token_id).count();
    if slots != images.len() {
        bail!("the prompt has {slots} image placeholders for {} images", images.len());
    }
    let mut out = Vec::with_capacity(tokens.len() + images.iter().map(|i| i.tokens(merge)).sum::<usize>());
    let mut spans = Vec::with_capacity(images.len());
    let mut it = images.into_iter();
    for &t in tokens {
        if t != image_token_id {
            out.push(t);
            continue;
        }
        let img = it.next().unwrap();
        let (gt, gh, gw) = img.grid;
        let len = img.tokens(merge);
        spans.push(ImageSpan {
            start: out.len(),
            len,
            grid: (gt, gh / merge, gw / merge),
            image: std::sync::Arc::new(img),
        });
        let hash = spans.last().unwrap().image.hash;
        out.extend((0..len).map(|k| pseudo_id(hash, k)));
    }
    Ok((out, spans))
}

/// mRoPE positions of a sequence with images (HF's `get_rope_index`): text tokens advance all
/// three axes together; an image starting at position `s` takes `(s + frame, s + row, s + col)`
/// on its merged grid, and the text after it resumes at `s + max(rows, cols)`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MropePositions {
    /// (start index, len, merged grid) per image, in order.
    spans: Vec<(usize, usize, (usize, usize, usize))>,
}

impl MropePositions {
    pub fn new(spans: &[ImageSpan]) -> Self {
        Self { spans: spans.iter().map(|s| (s.start, s.len, s.grid)).collect() }
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// (t, h, w) position of token `i`.
    pub fn at(&self, i: usize) -> [u32; 3] {
        let mut shift: i64 = 0; // position - index
        for &(start, len, (gt, gh, gw)) in &self.spans {
            if i < start {
                break;
            }
            let s = start as i64 + shift;
            if i < start + len {
                let k = i - start;
                let (f, rem) = (k / (gh * gw), k % (gh * gw));
                let _ = gt;
                return [(s + f as i64) as u32, (s + (rem / gw) as i64) as u32, (s + (rem % gw) as i64) as u32];
            }
            shift += gh.max(gw) as i64 - len as i64;
        }
        let p = (i as i64 + shift) as u32;
        [p, p, p]
    }
}

/// Python's `round` (half to even).
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// HF's `smart_resize`: sides divisible by `factor`, area within `[min_pixels, max_pixels]`,
/// aspect ratio kept as closely as possible. Returns (height, width).
pub fn smart_resize(h: usize, w: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    if h == 0 || w == 0 {
        bail!("empty image");
    }
    if h.max(w) as f64 / h.min(w) as f64 > 200.0 {
        bail!("image aspect ratio {}x{} exceeds 200", w, h);
    }
    let f = factor as f64;
    let (hf, wf) = (h as f64, w as f64);
    let mut hb = round_half_even(hf / f) * f;
    let mut wb = round_half_even(wf / f) * f;
    if hb * wb > max_pixels as f64 {
        let beta = (hf * wf / max_pixels as f64).sqrt();
        hb = f.max((hf / beta / f).floor() * f);
        wb = f.max((wf / beta / f).floor() * f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// Keys cubic kernel with a = -0.5 (PIL's and torchvision's antialiased bicubic).
fn cubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// Per output index along one axis: first source index and normalised weights (PIL's
/// `precompute_coeffs`: the kernel is widened by the scale when downscaling).
fn coeffs(n_in: usize, n_out: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = n_in as f64 / n_out as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    (0..n_out)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
            let xmax = ((center + support + 0.5).floor().min(n_in as f64)) as usize;
            let mut w: Vec<f64> = (xmin..xmax).map(|j| cubic((j as f64 - center + 0.5) / filterscale)).collect();
            let s: f64 = w.iter().sum();
            if s != 0.0 {
                w.iter_mut().for_each(|x| *x /= s);
            }
            (xmin, w)
        })
        .collect()
}

/// Antialiased bicubic resize of an RGB8 image, rounded and clamped to 8 bits per pass.
fn resize_bicubic(rgb: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let to_u8 = |v: f64| v.round().clamp(0.0, 255.0) as u8;
    // horizontal pass
    let cx = coeffs(w, ow);
    let mut tmp = vec![0u8; ow * h * 3];
    for y in 0..h {
        for (x, (x0, ws)) in cx.iter().enumerate() {
            for c in 0..3 {
                let v: f64 = ws.iter().enumerate().map(|(k, wt)| wt * rgb[(y * w + x0 + k) * 3 + c] as f64).sum();
                tmp[(y * ow + x) * 3 + c] = to_u8(v);
            }
        }
    }
    // vertical pass
    let cy = coeffs(h, oh);
    let mut out = vec![0u8; ow * oh * 3];
    for (y, (y0, ws)) in cy.iter().enumerate() {
        for x in 0..ow {
            for c in 0..3 {
                let v: f64 = ws.iter().enumerate().map(|(k, wt)| wt * tmp[((y0 + k) * ow + x) * 3 + c] as f64).sum();
                out[(y * ow + x) * 3 + c] = to_u8(v);
            }
        }
    }
    out
}

fn fnv1a(rgb: &[u8], w: usize, h: usize) -> u64 {
    let mut x: u64 = 0xcbf2_9ce4_8422_2325;
    for b in (w as u64).to_le_bytes().iter().chain(&(h as u64).to_le_bytes()).chain(rgb) {
        x ^= *b as u64;
        x = x.wrapping_mul(0x100_0000_01b3);
    }
    x
}

/// The bytes of an image reference: a `data:` URL (base64), an http(s) URL, or a local path.
pub fn fetch(url: &str) -> Result<Vec<u8>> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').context("malformed data URL")?;
        if !meta.ends_with(";base64") {
            bail!("data URLs must be base64 encoded");
        }
        return decode_base64(data);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let resp = ureq::get(url).call().with_context(|| format!("fetching {url}"))?;
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut resp.into_reader().take(64 << 20), &mut buf)?;
        return Ok(buf);
    }
    let path = url.strip_prefix("file://").unwrap_or(url);
    std::fs::read(path).with_context(|| format!("reading image {path}"))
}

/// Raw base64 image data (Anthropic `source.data`).
pub fn decode_base64(data: &str) -> Result<Vec<u8>> {
    let clean: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    base64::engine::general_purpose::STANDARD.decode(clean).context("decoding the base64 image")
}

/// Bare base64 (no `data:` prefix) that decodes to a known image format.
pub fn bare_base64_image(data: &str) -> Option<Vec<u8>> {
    let bytes = decode_base64(data).ok()?;
    image::guess_format(&bytes).is_ok().then_some(bytes)
}

use std::io::Read as _;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_hf() {
        // values from transformers' smart_resize
        assert_eq!(smart_resize(28, 40, 8, 64, 4096).unwrap(), (32, 40));
        assert_eq!(smart_resize(1080, 1920, 32, 65536, 16777216).unwrap(), (1088, 1920));
        assert_eq!(smart_resize(100, 100, 32, 65536, 16777216).unwrap(), (256, 256));
        assert_eq!(smart_resize(4000, 3000, 32, 65536, 1 << 20).unwrap(), (1152, 864));
        assert!(smart_resize(1, 300, 8, 64, 4096).is_err());
    }

    #[test]
    fn mrope_positions_follow_hf() {
        // HF get_rope_index for 12 text tokens, vision_start, a 4x5 merged image, vision_end, 9
        // text tokens (the vision reference's prompt): the image takes positions 13.., the text
        // after it resumes at 13 + 5
        let img = PreparedImage { pixels: vec![], grid: (1, 8, 10), hash: 7 };
        let mut toks: Vec<u32> = (0..12).collect();
        toks.extend([261, 259, 262]);
        toks.extend(0..9);
        let (out, spans) = expand_image_tokens(&toks, 259, vec![img], 2).unwrap();
        assert_eq!(out.len(), 12 + 1 + 20 + 1 + 9);
        assert!(out[13..33].iter().all(|&t| t & 0x8000_0000 != 0));
        assert_eq!(model_token(out[13], 259), 259);
        let m = MropePositions::new(&spans);
        assert_eq!(m.at(12), [12, 12, 12]);
        assert_eq!(m.at(13), [13, 13, 13]);
        assert_eq!(m.at(14), [13, 13, 14]);
        assert_eq!(m.at(13 + 5), [13, 14, 13]);
        assert_eq!(m.at(32), [13, 16, 17]);
        assert_eq!(m.at(33), [18, 18, 18]); // vision_end
        assert_eq!(m.at(42), [27, 27, 27]);
        assert!(expand_image_tokens(&toks, 259, vec![], 2).is_err());
    }

    #[test]
    fn half_even_rounding() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(-2.5), -2.0);
        assert_eq!(round_half_even(2.4), 2.0);
    }
}
