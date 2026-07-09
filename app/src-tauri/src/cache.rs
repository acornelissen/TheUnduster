//! Probability-map cache codec: u8-quantized, zstd-compressed storage
//! with detector-hash validation and strict boundary checking.

use std::path::Path;

#[cfg_attr(not(test), allow(dead_code))]
pub const PROBS_MAGIC: &[u8; 8] = b"UNDPROB1";

/// Writes width*height probabilities as u8 (round(p*255)), zstd-compressed,
/// with the producing detector's file hash in the header. Atomic.
#[cfg_attr(not(test), allow(dead_code))]
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
#[cfg_attr(not(test), allow(dead_code))]
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

    // Validate compressed length fits in file
    if offset as u64 + compressed_len != bytes.len() as u64 {
        return corrupt();
    }

    // Decompress
    let compressed_data = &bytes[offset..];
    let quantized = match zstd::decode_all(compressed_data) {
        Ok(q) => q,
        Err(_) => return corrupt(),
    };

    // Validate decompressed size
    let expected_size = (width as usize).checked_mul(height as usize)?;
    if quantized.len() != expected_size {
        return corrupt();
    }

    // Dequantize
    let probs = quantized.iter().map(|&q| q as f32 / 255.0).collect();

    Some(probs)
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
}
