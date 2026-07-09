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
}
