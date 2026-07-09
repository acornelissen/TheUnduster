//! Probability-map cache codec: u8-quantized, zstd-compressed storage
//! with detector-hash validation and strict boundary checking.
//!
//! Also: heal delta codec, which persists a heal as its mask (bitset) plus
//! the healed pixel values inside it, provenance-hashed, reconstructed
//! bit-exactly as original + patch.

use std::path::Path;

use sha2::{Digest, Sha256};

pub const PROBS_MAGIC: &[u8; 8] = b"UNDPROB1";

/// Writes width*height probabilities as u8 (round(p*255)), zstd-compressed,
/// with the producing detector's file hash in the header. Atomic.
pub fn write_probs(
    path: &Path,
    probs: &[f32],
    width: u32,
    height: u32,
    detector_hash: &[u8; 32],
) -> Result<(), String> {
    // Validate dimensions
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "dimensions overflow".to_string())?;
    if probs.len() != expected_len {
        return Err(format!(
            "probs length {} does not match width*height {}",
            probs.len(),
            expected_len
        ));
    }

    // Quantize probabilities to u8
    let quantized: Vec<u8> = probs.iter().map(|&p| (p * 255.0).round() as u8).collect();

    // Build header
    let mut header = Vec::new();
    header.extend_from_slice(PROBS_MAGIC);
    header.extend_from_slice(&1u32.to_le_bytes()); // version
    header.extend_from_slice(&width.to_le_bytes());
    header.extend_from_slice(&height.to_le_bytes());
    header.extend_from_slice(detector_hash);

    // Compress the quantized data
    let compressed =
        zstd::encode_all(&quantized[..], 3).map_err(|e| format!("zstd compression failed: {e}"))?;

    let compressed_len = compressed.len() as u64;
    header.extend_from_slice(&compressed_len.to_le_bytes());
    header.extend_from_slice(&compressed);

    // Write atomically
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("bad path: {}", path.display()))?;
    let tmp_path = path.with_file_name(format!("{}.tmp-unduster", file_name));

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }

    std::fs::write(&tmp_path, &header).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!("write tmp failed: {e}")
    })?;

    std::fs::rename(&tmp_path, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!("rename failed: {e}")
    })?;

    Ok(())
}

/// Reads a probs cache written by write_probs. Returns None (never Err) when
/// the file is absent, malformed, dimension-mismatched, or produced by a
/// different detector -- malformed files are deleted on sight.
pub fn read_probs(
    path: &Path,
    width: u32,
    height: u32,
    detector_hash: &[u8; 32],
) -> Option<Vec<f32>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return None, // absent file is not corruption
    };

    // Helper to mark as corrupt and return None
    let corrupt = || {
        let _ = std::fs::remove_file(path);
        None
    };

    // Check minimum header size: magic(8) + version(4) + width(4) + height(4) + hash(32) + len(8) = 60
    if bytes.len() < 60 {
        return corrupt();
    }

    let mut offset = 0;

    // Check magic
    if &bytes[offset..offset + 8] != PROBS_MAGIC {
        return corrupt();
    }
    offset += 8;

    // Check version
    let version = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    offset += 4;
    if version != 1 {
        return corrupt();
    }

    // Check dimensions
    let stored_width = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    offset += 4;
    let stored_height = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    offset += 4;

    if stored_width != width || stored_height != height {
        return None; // dimension mismatch is not corruption
    }

    // Check detector hash
    let stored_hash: &[u8] = &bytes[offset..offset + 32];
    offset += 32;

    if stored_hash != detector_hash {
        return None; // hash mismatch is not corruption
    }

    // Read compressed length
    let compressed_len = u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ]);
    offset += 8;

    // Validate compressed length fits in file. Use checked_add: a crafted
    // length near u64::MAX must not panic on overflow in debug builds.
    let total_len = match (offset as u64).checked_add(compressed_len) {
        Some(t) => t,
        None => return corrupt(), // length arithmetic overflow is structural corruption
    };
    if total_len != bytes.len() as u64 {
        return corrupt();
    }

    // Expected decompressed size is known before decompressing; use it to
    // bound the allocation so a crafted small file can't decompress-bomb us.
    let expected_size = match (width as usize).checked_mul(height as usize) {
        Some(s) => s,
        None => return corrupt(), // unreachable on 64-bit, but treat as corruption not a bare bailout
    };

    // Decompress, capped at expected_size. bulk::decompress's capacity is
    // exact: a frame that decompresses to more than expected_size errors,
    // which is exactly the corruption signal we want. Equal-but-wrong
    // content is caught by the length/hash checks below.
    let compressed_data = &bytes[offset..];
    let quantized = match zstd::bulk::decompress(compressed_data, expected_size) {
        Ok(q) => q,
        Err(_) => return corrupt(),
    };

    // Validate decompressed size
    if quantized.len() != expected_size {
        return corrupt();
    }

    // Dequantize
    let probs = quantized.iter().map(|&q| q as f32 / 255.0).collect();

    Some(probs)
}

pub const HEAL_MAGIC: &[u8; 8] = b"UNDHEAL1";

/// Header layout (all little-endian):
/// magic(8) | version u32 | width u32 | height u32 | channels u8 | depth u8
/// | provenance(32) | mask_comp_len u64 | zstd(bitset) | values_comp_len u64
/// | zstd(values)
const HEAL_HEADER_LEN: usize = 8 + 4 + 4 + 4 + 1 + 1 + 32 + 8;

/// Provenance of a heal: any input that changes the output contributes.
/// Strokes are canonicalized as serde_json bytes (deterministic for
/// identical f32 bit patterns, which is exactly the invariant we want).
pub fn heal_provenance(
    threshold: f32,
    dilate_radius: u32,
    strokes: &[crate::masks::Stroke],
    detector_hash: &[u8; 32],
    inpainter_hash: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(threshold.to_le_bytes());
    hasher.update(dilate_radius.to_le_bytes());
    hasher.update(serde_json::to_vec(strokes).unwrap_or_default());
    hasher.update(detector_hash);
    hasher.update(inpainter_hash);
    hasher.finalize().into()
}

/// Depth in bytes per channel value: 1 for u8, 2 for u16.
fn pixel_data_depth(data: &fd_io::PixelData) -> u8 {
    match data {
        fd_io::PixelData::U8(_) => 1,
        fd_io::PixelData::U16(_) => 2,
    }
}

/// Persists a heal as a delta: the mask (bitset, zstd) and the healed pixel
/// values inside it (native depth, zstd), plus the provenance. Atomic.
pub fn write_heal(
    path: &Path,
    original: &fd_io::ImageBuf,
    healed: &fd_io::ImageBuf,
    mask: &[bool],
    provenance: &[u8; 32],
) -> Result<(), String> {
    if healed.width != original.width
        || healed.height != original.height
        || healed.channels != original.channels
    {
        return Err("healed dimensions/channels do not match original".to_string());
    }
    let depth = pixel_data_depth(&original.data);
    if depth != pixel_data_depth(&healed.data) {
        return Err("healed depth does not match original".to_string());
    }

    let pixel_count = (original.width as usize)
        .checked_mul(original.height as usize)
        .ok_or_else(|| "dimensions overflow".to_string())?;
    if mask.len() != pixel_count {
        return Err(format!(
            "mask length {} does not match width*height {}",
            mask.len(),
            pixel_count
        ));
    }

    let channels = original.channels as usize;

    // Pack the mask into a row-major, LSB-first bitset.
    let bitset_len = pixel_count.div_ceil(8);
    let mut bitset = vec![0u8; bitset_len];
    for (i, &m) in mask.iter().enumerate() {
        if m {
            bitset[i / 8] |= 1 << (i % 8);
        }
    }

    // Collect the healed values at masked positions, row-major, native depth.
    let mut values: Vec<u8> = Vec::new();
    match &healed.data {
        fd_io::PixelData::U8(v) => {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    let base = i * channels;
                    values.extend_from_slice(&v[base..base + channels]);
                }
            }
        }
        fd_io::PixelData::U16(v) => {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    let base = i * channels;
                    for &sample in &v[base..base + channels] {
                        values.extend_from_slice(&sample.to_le_bytes());
                    }
                }
            }
        }
    }

    let mask_compressed =
        zstd::encode_all(&bitset[..], 3).map_err(|e| format!("zstd compression failed: {e}"))?;
    let values_compressed =
        zstd::encode_all(&values[..], 3).map_err(|e| format!("zstd compression failed: {e}"))?;

    let mut header = Vec::new();
    header.extend_from_slice(HEAL_MAGIC);
    header.extend_from_slice(&1u32.to_le_bytes()); // version
    header.extend_from_slice(&original.width.to_le_bytes());
    header.extend_from_slice(&original.height.to_le_bytes());
    header.push(original.channels);
    header.push(depth * 8); // stored as bits (8 or 16), per file layout
    header.extend_from_slice(provenance);
    header.extend_from_slice(&(mask_compressed.len() as u64).to_le_bytes());
    header.extend_from_slice(&mask_compressed);
    header.extend_from_slice(&(values_compressed.len() as u64).to_le_bytes());
    header.extend_from_slice(&values_compressed);

    // Write atomically.
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("bad path: {}", path.display()))?;
    let tmp_path = path.with_file_name(format!("{}.tmp-unduster", file_name));

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }

    std::fs::write(&tmp_path, &header).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!("write tmp failed: {e}")
    })?;

    std::fs::rename(&tmp_path, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!("rename failed: {e}")
    })?;

    Ok(())
}

/// Reconstructs the healed image (original + patch) IF the cache entry
/// matches the requested provenance and the original's dimensions/depth.
/// Returns the healed copy and the mask. None on any mismatch; malformed
/// files deleted on sight.
pub fn read_heal(
    path: &Path,
    original: &fd_io::ImageBuf,
    provenance: &[u8; 32],
) -> Option<(fd_io::ImageBuf, Vec<bool>)> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return None, // absent file is not corruption
    };

    let corrupt = || {
        let _ = std::fs::remove_file(path);
        None
    };

    if bytes.len() < HEAL_HEADER_LEN {
        return corrupt();
    }

    let mut offset = 0;

    if &bytes[offset..offset + 8] != HEAL_MAGIC {
        return corrupt();
    }
    offset += 8;

    let version = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    if version != 1 {
        return corrupt();
    }

    let stored_width = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let stored_height = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;

    let stored_channels = bytes[offset];
    offset += 1;
    let stored_depth = bytes[offset];
    offset += 1;

    let stored_provenance: &[u8] = &bytes[offset..offset + 32];
    offset += 32;

    // Structural bounds (magic/version/lengths) are corruption when wrong.
    // Well-formed-but-mismatched dims/depth/channels/provenance is a miss:
    // keep the file, return None.
    if stored_depth != 8 && stored_depth != 16 {
        return corrupt();
    }

    let original_depth = pixel_data_depth(&original.data) as u32 * 8;
    if stored_width != original.width
        || stored_height != original.height
        || stored_channels != original.channels
        || stored_depth as u32 != original_depth
    {
        return None;
    }

    if stored_provenance != provenance {
        return None;
    }

    // Read mask_comp_len.
    if bytes.len() < offset + 8 {
        return corrupt();
    }
    let mask_comp_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    offset += 8;

    let mask_comp_end = match (offset as u64).checked_add(mask_comp_len) {
        Some(e) if e <= bytes.len() as u64 => e as usize,
        _ => return corrupt(),
    };
    let mask_compressed = &bytes[offset..mask_comp_end];
    offset = mask_comp_end;

    let pixel_count = match (stored_width as usize).checked_mul(stored_height as usize) {
        Some(p) => p,
        None => return corrupt(),
    };
    let expected_bitset_len = pixel_count.div_ceil(8);

    let bitset = match zstd::bulk::decompress(mask_compressed, expected_bitset_len) {
        Ok(b) => b,
        Err(_) => return corrupt(),
    };
    if bitset.len() != expected_bitset_len {
        return corrupt();
    }

    // Read values_comp_len.
    if bytes.len() < offset + 8 {
        return corrupt();
    }
    let values_comp_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    offset += 8;

    let values_comp_end = match (offset as u64).checked_add(values_comp_len) {
        Some(e) if e <= bytes.len() as u64 => e as usize,
        _ => return corrupt(),
    };
    let values_compressed = &bytes[offset..values_comp_end];
    offset = values_comp_end;

    // No trailing garbage.
    if offset != bytes.len() {
        return corrupt();
    }

    // Popcount from the *decompressed, validated* bitset bounds the values
    // decompression -- this must be computed before decompressing values.
    let mask: Vec<bool> = (0..pixel_count)
        .map(|i| (bitset[i / 8] >> (i % 8)) & 1 == 1)
        .collect();
    let popcount = mask.iter().filter(|&&m| m).count();

    let channels = stored_channels as usize;
    let depth_bytes = (stored_depth as usize) / 8;
    let expected_values_len = match popcount
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(depth_bytes))
    {
        Some(v) => v,
        None => return corrupt(),
    };

    let values = match zstd::bulk::decompress(values_compressed, expected_values_len) {
        Ok(v) => v,
        Err(_) => return corrupt(),
    };
    if values.len() != expected_values_len {
        return corrupt();
    }

    // Reconstruct: clone original, scatter values into masked positions.
    let mut result_data = original.data.clone();
    match &mut result_data {
        fd_io::PixelData::U8(out) => {
            let mut vi = 0usize;
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    let base = i * channels;
                    out[base..base + channels].copy_from_slice(&values[vi..vi + channels]);
                    vi += channels;
                }
            }
        }
        fd_io::PixelData::U16(out) => {
            let mut vi = 0usize;
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    let base = i * channels;
                    for c in 0..channels {
                        let sample = u16::from_le_bytes([values[vi], values[vi + 1]]);
                        out[base + c] = sample;
                        vi += 2;
                    }
                }
            }
        }
    }

    let result = fd_io::ImageBuf {
        width: original.width,
        height: original.height,
        channels: original.channels,
        data: result_data,
        icc: original.icc.clone(),
        exif: original.exif.clone(),
    };

    Some((result, mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_probs(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 97) as f32) / 96.0).collect()
    }

    #[test]
    fn probs_round_trip_within_quantization() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let probs = synth_probs(64 * 48);
        let hash = [7u8; 32];
        write_probs(&p, &probs, 64, 48, &hash).unwrap();
        let back = read_probs(&p, 64, 48, &hash).expect("cache readable");
        assert_eq!(back.len(), probs.len());
        for (a, b) in probs.iter().zip(&back) {
            assert!((a - b).abs() <= 0.5 / 255.0 + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn probs_reject_wrong_dims_hash_and_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let probs = synth_probs(16 * 16);
        write_probs(&p, &probs, 16, 16, &[1u8; 32]).unwrap();
        assert!(read_probs(&p, 16, 17, &[1u8; 32]).is_none()); // dims
        assert!(p.exists(), "dim mismatch is not corruption; file kept");
        assert!(read_probs(&p, 16, 16, &[2u8; 32]).is_none()); // detector changed
        assert!(p.exists(), "hash mismatch is not corruption; file kept");
        let mut bytes = std::fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&p, &bytes).unwrap();
        assert!(read_probs(&p, 16, 16, &[1u8; 32]).is_none()); // corrupt payload
        assert!(!p.exists(), "corrupt file deleted on sight");
        assert!(read_probs(&p, 16, 16, &[1u8; 32]).is_none()); // absent -> None
    }

    #[test]
    fn crafted_length_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let probs = synth_probs(16 * 16);
        let hash = [1u8; 32];
        write_probs(&p, &probs, 16, 16, &hash).unwrap();

        // Overwrite the compressed_len field (offset 52..60) with u64::MAX.
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[52..60].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&p, &bytes).unwrap();

        assert!(read_probs(&p, 16, 16, &hash).is_none());
        assert!(!p.exists(), "crafted length is treated as corruption");
    }

    #[test]
    fn oversized_decompression_is_rejected_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let width: u32 = 16;
        let height: u32 = 16; // expected decompressed size = 256 bytes

        // zstd frame that decompresses to 1MB of zeros -- wildly more than
        // the 256 bytes the header claims.
        let bomb_payload = vec![0u8; 1024 * 1024];
        let compressed = zstd::encode_all(&bomb_payload[..], 3).unwrap();

        let mut header = Vec::new();
        header.extend_from_slice(PROBS_MAGIC);
        header.extend_from_slice(&1u32.to_le_bytes());
        header.extend_from_slice(&width.to_le_bytes());
        header.extend_from_slice(&height.to_le_bytes());
        header.extend_from_slice(&[3u8; 32]);
        header.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        header.extend_from_slice(&compressed);

        std::fs::write(&p, &header).unwrap();

        // Bounded decompression must reject (not allocate 1MB and then
        // reject after the fact) -- we can only assert on the outcome:
        // rejection plus deletion, since allocation size isn't observable
        // from a test.
        assert!(read_probs(&p, width, height, &[3u8; 32]).is_none());
        assert!(
            !p.exists(),
            "oversized decompression is treated as corruption"
        );
    }

    #[test]
    fn probs_write_is_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        write_probs(&p, &synth_probs(8 * 8), 8, 8, &[0u8; 32]).unwrap();
        // no tmp siblings left behind
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter(|e| e.as_ref().unwrap().file_name() != "a.probs")
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    fn noisy16(w: u32, h: u32) -> fd_io::ImageBuf {
        let n = (w * h * 3) as usize;
        let mut s = 0x2545F4914F6CDD1Du64;
        let data: Vec<u16> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 48) as u16
            })
            .collect();
        fd_io::ImageBuf {
            width: w,
            height: h,
            channels: 3,
            data: fd_io::PixelData::U16(data),
            icc: Some(vec![1, 2, 3]),
            exif: None,
        }
    }

    #[test]
    fn heal_delta_round_trips_bit_exact() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.heal");
        let original = noisy16(40, 30);
        let mut healed = original.clone();
        let mut mask = vec![false; 40 * 30];
        for y in 5..12 {
            for x in 20..35 {
                mask[y * 40 + x] = true;
            }
        }
        if let fd_io::PixelData::U16(v) = &mut healed.data {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    for c in 0..3 {
                        v[i * 3 + c] = v[i * 3 + c].wrapping_add(1000 + c as u16);
                    }
                }
            }
        }
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32]);
        write_heal(&p, &original, &healed, &mask, &prov).unwrap();
        let (back, back_mask) = read_heal(&p, &original, &prov).expect("cache hit");
        assert_eq!(back_mask, mask);
        let (fd_io::PixelData::U16(a), fd_io::PixelData::U16(b)) = (&healed.data, &back.data)
        else {
            panic!("depth changed");
        };
        assert_eq!(a, b, "reconstruction must be bit-exact");
        assert_eq!(back.icc, original.icc, "metadata rides the original");
    }

    #[test]
    fn heal_delta_rejects_provenance_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.heal");
        let original = noisy16(16, 16);
        let healed = original.clone();
        let mask = vec![false; 256];
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32]);
        write_heal(&p, &original, &healed, &mask, &prov).unwrap();
        let other = heal_provenance(0.51, 2, &[], &[3u8; 32], &[4u8; 32]);
        assert!(read_heal(&p, &original, &other).is_none());
        assert!(p.exists(), "provenance miss keeps the file");
    }

    #[test]
    fn heal_provenance_distinguishes_every_input() {
        let base = heal_provenance(0.5, 2, &[], &[0u8; 32], &[0u8; 32]);
        let stroke = crate::masks::Stroke {
            erase: false,
            radius: 5.0,
            points: vec![[1.0, 2.0]],
        };
        assert_ne!(base, heal_provenance(0.6, 2, &[], &[0u8; 32], &[0u8; 32]));
        assert_ne!(base, heal_provenance(0.5, 3, &[], &[0u8; 32], &[0u8; 32]));
        assert_ne!(
            base,
            heal_provenance(
                0.5,
                2,
                std::slice::from_ref(&stroke),
                &[0u8; 32],
                &[0u8; 32]
            )
        );
        assert_ne!(base, heal_provenance(0.5, 2, &[], &[1u8; 32], &[0u8; 32]));
        assert_ne!(base, heal_provenance(0.5, 2, &[], &[0u8; 32], &[1u8; 32]));
    }

    #[test]
    fn heal_crafted_values_length_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.heal");
        let original = noisy16(16, 16);
        let healed = original.clone();
        let mask = vec![false; 256];
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32]);
        write_heal(&p, &original, &healed, &mask, &prov).unwrap();

        // Header layout: magic(8) version(4) width(4) height(4) channels(1)
        // depth(1) provenance(32) mask_comp_len(8) <mask bytes> values_comp_len(8) ...
        let mut bytes = std::fs::read(&p).unwrap();
        let mask_len_off = 8 + 4 + 4 + 4 + 1 + 1 + 32;
        let mask_comp_len =
            u64::from_le_bytes(bytes[mask_len_off..mask_len_off + 8].try_into().unwrap());
        let values_len_off = mask_len_off + 8 + mask_comp_len as usize;
        bytes[values_len_off..values_len_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&p, &bytes).unwrap();

        assert!(read_heal(&p, &original, &prov).is_none());
        assert!(
            !p.exists(),
            "crafted values length is treated as corruption"
        );
    }

    #[test]
    fn heal_oversized_values_decompression_is_rejected_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.heal");
        let original = noisy16(16, 16);
        let healed = original.clone();
        let mut mask = vec![false; 256];
        mask[0] = true; // one masked pixel -> expects 1 * 3 * 2 = 6 bytes of values
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32]);
        write_heal(&p, &original, &healed, &mask, &prov).unwrap();

        // Replace the values section with a zstd frame that decompresses to
        // far more than the 6 bytes the mask popcount implies.
        let mask_len_off = 8 + 4 + 4 + 4 + 1 + 1 + 32;
        let mut bytes = std::fs::read(&p).unwrap();
        let mask_comp_len =
            u64::from_le_bytes(bytes[mask_len_off..mask_len_off + 8].try_into().unwrap());
        let values_len_off = mask_len_off + 8 + mask_comp_len as usize;

        let bomb_payload = vec![0u8; 1024 * 1024];
        let compressed = zstd::encode_all(&bomb_payload[..], 3).unwrap();

        let mut new_bytes = bytes[..values_len_off].to_vec();
        new_bytes.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        new_bytes.extend_from_slice(&compressed);
        bytes = new_bytes;
        std::fs::write(&p, &bytes).unwrap();

        assert!(read_heal(&p, &original, &prov).is_none());
        assert!(
            !p.exists(),
            "oversized values decompression is treated as corruption"
        );
    }
}
