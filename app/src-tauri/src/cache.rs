//! Probability-map cache codec: u8-quantized, zstd-compressed storage
//! with detector-hash validation and strict boundary checking.
//!
//! Also: heal delta codec, which persists a heal as its mask (bitset) plus
//! the healed pixel values inside it, provenance-hashed, reconstructed
//! bit-exactly as original + patch.
//!
//! Also: display pyramid codec, which persists each level's RGBA buffer
//! zstd-compressed behind a stamp-only (no content hash) header -- the
//! pyramid is a pure function of the source file, so the source stamp alone
//! is enough provenance.

use std::path::Path;

use fd_tiles::{downsample_2x, Level, Pyramid, TILE_SIZE};
use sha2::{Digest, Sha256};

pub const PROBS_MAGIC: &[u8; 8] = b"UNDPROB1";

/// Binds a cache entry to the identity of the source file it was produced
/// from: size plus modification time (nanoseconds since the Unix epoch,
/// truncated to u64). A rescan or in-place edit of the source changes at
/// least one of these, so a stale cache entry can never be resurrected by
/// simply re-reading the same path -- collision requires the same size AND
/// the same nanosecond mtime, which is good enough against rescans/edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceStamp {
    pub size: u64,
    pub mtime_nanos: u64,
}

/// Stats `path` and returns its `SourceStamp`. A pre-epoch mtime maps to 0
/// via the saturating duration_since -- fine, since that just means "very
/// old" collides with other very-old files, an acceptable, exceedingly rare
/// edge case for real film-scan files.
pub fn source_stamp(path: &Path) -> Result<SourceStamp, String> {
    let metadata = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let modified = metadata
        .modified()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mtime_nanos = modified
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    Ok(SourceStamp {
        size: metadata.len(),
        mtime_nanos,
    })
}

/// Writes width*height probabilities as u8 (round(p*255)), zstd-compressed,
/// with the producing detector's file hash and the source file's stamp in
/// the header. Atomic.
pub fn write_probs(
    path: &Path,
    probs: &[f32],
    width: u32,
    height: u32,
    detector_hash: &[u8; 32],
    stamp: &SourceStamp,
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
    header.extend_from_slice(&2u32.to_le_bytes()); // version
    header.extend_from_slice(&width.to_le_bytes());
    header.extend_from_slice(&height.to_le_bytes());
    header.extend_from_slice(detector_hash);
    header.extend_from_slice(&stamp.size.to_le_bytes());
    header.extend_from_slice(&stamp.mtime_nanos.to_le_bytes());

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
/// the file is absent, malformed, dimension-mismatched, produced by a
/// different detector, or stamped for a different source file -- malformed
/// files (including any file with an unknown version, e.g. a pre-stamp v1
/// file) are deleted on sight; well-formed-but-mismatched files are kept.
pub fn read_probs(
    path: &Path,
    width: u32,
    height: u32,
    detector_hash: &[u8; 32],
    stamp: &SourceStamp,
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

    // Check minimum header size:
    // magic(8) + version(4) + width(4) + height(4) + hash(32) + size(8) +
    // mtime(8) + len(8) = 76
    if bytes.len() < 76 {
        return corrupt();
    }

    let mut offset = 0;

    // Check magic
    if &bytes[offset..offset + 8] != PROBS_MAGIC {
        return corrupt();
    }
    offset += 8;

    // Check version. Any version other than the current one -- including a
    // pre-stamp v1 file -- is unreadable and self-migrates by deletion.
    let version = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    offset += 4;
    if version != 2 {
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

    // Check the source stamp (size, mtime_nanos).
    let stored_size = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    offset += 8;
    let stored_mtime_nanos = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    offset += 8;

    if stored_size != stamp.size || stored_mtime_nanos != stamp.mtime_nanos {
        return None; // stamp mismatch is not corruption
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

// Bumped from UNDHEAL1 to UNDHEAL2 because the source stamp joined
// provenance's hashed inputs: a pre-stamp heal file's provenance can never
// match a provenance hash computed with the new formula, so it would
// otherwise linger on disk unmatched forever instead of purging. Bumping the
// magic makes it fail the structural check (corrupt) and delete on sight,
// exactly like an old-version probs file.
pub const HEAL_MAGIC: &[u8; 8] = b"UNDHEAL2";

/// Header layout (all little-endian):
/// magic(8) | version u32 | width u32 | height u32 | channels u8 | depth u8
/// | provenance(32) | mask_comp_len u64 | zstd(bitset) | values_comp_len u64
/// | zstd(values)
const HEAL_HEADER_LEN: usize = 8 + 4 + 4 + 4 + 1 + 1 + 32 + 8;

/// Provenance of a heal: any input that changes the output contributes.
/// Strokes are canonicalized as serde_json bytes (deterministic for
/// identical f32 bit patterns, which is exactly the invariant we want).
/// Hashed in a fixed, documented order: threshold, dilate radius, strokes,
/// detector hash, inpainter hash, then the source stamp (size LE, then
/// mtime_nanos LE) last.
pub fn heal_provenance(
    threshold: f32,
    dilate_radius: u32,
    strokes: &[crate::masks::Stroke],
    detector_hash: &[u8; 32],
    inpainter_hash: &[u8; 32],
    source: &SourceStamp,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(threshold.to_le_bytes());
    hasher.update(dilate_radius.to_le_bytes());
    hasher.update(serde_json::to_vec(strokes).unwrap_or_default());
    hasher.update(detector_hash);
    hasher.update(inpainter_hash);
    hasher.update(source.size.to_le_bytes());
    hasher.update(source.mtime_nanos.to_le_bytes());
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

pub const PYRAMID_MAGIC: &[u8; 8] = b"UNDPYRA1";
pub const PYRAMID_VERSION: u32 = 1;

/// Top-level header length: magic(8) | version(4) | level_count(4) |
/// size(8) | mtime_nanos(8). Per-level header (width(4) | height(4) |
/// comp_len(8)) follows separately, once per level.
const PYRAMID_HEADER_LEN: usize = 8 + 4 + 4 + 8 + 8;
const PYRAMID_LEVEL_HEADER_LEN: usize = 4 + 4 + 8;

/// A pyramid always has a base level; anything claiming more than this is
/// almost certainly a corrupted length field rather than a real pyramid --
/// `fd_tiles::Pyramid::build` halves dimensions each level down to
/// `TILE_SIZE`, so even a gigapixel scan tops out well under this cap.
const MAX_PYRAMID_LEVELS: u32 = 32;

/// Upper bound on a single level's decompressed RGBA size (2 GiB).
/// `zstd::bulk::decompress` allocates its capacity argument EAGERLY
/// (`Vec::with_capacity`) before decompressing a single byte, so a
/// ~100-byte crafted file claiming a 65535x65535 level would otherwise
/// drive a ~17GB allocation attempt straight from untrusted header bytes.
/// Any level claiming more than this is the corrupt class (delete + None),
/// checked BEFORE the decompress call. Generously above any real level:
/// level 0 of a 24000x20000 scan is 1.92GB.
const MAX_PYRAMID_LEVEL_BYTES: usize = 2 << 30;

/// Writes the display pyramid: header
/// magic(8) | version(4) | level_count(4) | size(8) | mtime_nanos(8)
/// then per level: width(4) | height(4) | comp_len(8) | zstd payload.
/// zstd level 1, not the probs codec's 3: RGBA film grain barely
/// compresses at any level, and this write runs once per fresh build --
/// encode speed is worth more than a few percent of disk.
pub fn write_pyramid(path: &Path, pyramid: &Pyramid, stamp: &SourceStamp) -> Result<(), String> {
    let level_count: u32 = pyramid
        .levels
        .len()
        .try_into()
        .map_err(|_| "too many levels".to_string())?;

    let mut header = Vec::new();
    header.extend_from_slice(PYRAMID_MAGIC);
    header.extend_from_slice(&PYRAMID_VERSION.to_le_bytes());
    header.extend_from_slice(&level_count.to_le_bytes());
    header.extend_from_slice(&stamp.size.to_le_bytes());
    header.extend_from_slice(&stamp.mtime_nanos.to_le_bytes());

    for level in &pyramid.levels {
        let expected_len = (level.width as usize)
            .checked_mul(level.height as usize)
            .and_then(|p| p.checked_mul(4))
            .ok_or_else(|| "level dimensions overflow".to_string())?;
        if level.rgba.len() != expected_len {
            return Err(format!(
                "level rgba length {} does not match width*height*4 {}",
                level.rgba.len(),
                expected_len
            ));
        }

        let compressed = zstd::encode_all(&level.rgba[..], 1)
            .map_err(|e| format!("zstd compression failed: {e}"))?;

        header.extend_from_slice(&level.width.to_le_bytes());
        header.extend_from_slice(&level.height.to_le_bytes());
        header.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        header.extend_from_slice(&compressed);
    }

    // Write atomically: same temp+rename dance as write_probs/write_heal.
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

/// Reads a cached pyramid. None on: missing file, stamp mismatch
/// (source changed -- file kept, house mismatch-keep rule), or corrupt
/// structure (file deleted on sight). Every level's rgba length is
/// validated as width*height*4 with checked arithmetic, and each
/// decompression is bounded to that expected size.
pub fn read_pyramid(path: &Path, stamp: &SourceStamp) -> Option<Pyramid> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return None, // absent file is not corruption
    };

    let corrupt = || {
        let _ = std::fs::remove_file(path);
        None
    };

    if bytes.len() < PYRAMID_HEADER_LEN {
        return corrupt();
    }

    let mut offset = 0;

    if &bytes[offset..offset + 8] != PYRAMID_MAGIC {
        return corrupt();
    }
    offset += 8;

    let version = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    if version != PYRAMID_VERSION {
        return corrupt();
    }

    let level_count = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    // A pyramid always has a base level; zero is corrupt, not a valid
    // "empty" pyramid. The upper cap guards against a crafted/garbled
    // count driving an unbounded per-level read loop.
    if level_count == 0 || level_count > MAX_PYRAMID_LEVELS {
        return corrupt();
    }

    // Stamp check FIRST, before any structural per-level parsing: a
    // well-formed pyramid for a different source file is a miss (file
    // kept), not corruption, and that verdict must not be preempted by a
    // later structural check on data we don't even need in the mismatch
    // case.
    let stored_size = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    offset += 8;
    let stored_mtime_nanos = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    offset += 8;

    if stored_size != stamp.size || stored_mtime_nanos != stamp.mtime_nanos {
        return None; // stamp mismatch is not corruption
    }

    let mut levels = Vec::with_capacity(level_count as usize);
    for _ in 0..level_count {
        if bytes.len() < offset + PYRAMID_LEVEL_HEADER_LEN {
            return corrupt();
        }

        let width = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let height = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let comp_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let comp_end = match (offset as u64).checked_add(comp_len) {
            Some(e) if e <= bytes.len() as u64 => e as usize,
            _ => return corrupt(), // crafted/oversized length or truncated file
        };
        let compressed = &bytes[offset..comp_end];
        offset = comp_end;

        let expected_len = match (width as usize)
            .checked_mul(height as usize)
            .and_then(|p| p.checked_mul(4))
        {
            Some(l) => l,
            None => return corrupt(),
        };

        // Cap the claimed size BEFORE handing it to the decompressor: the
        // capacity below is allocated eagerly (see MAX_PYRAMID_LEVEL_BYTES),
        // so an unchecked header-claimed size is a memory-DoS lever.
        if expected_len > MAX_PYRAMID_LEVEL_BYTES {
            return corrupt();
        }

        // Bounded decompression: a frame that decompresses to more than
        // expected_len errors, catching both decompress-bombs and a
        // dimensions/payload disagreement in one shot. Equal-but-short
        // payloads are caught by the length check right after.
        let rgba = match zstd::bulk::decompress(compressed, expected_len) {
            Ok(r) => r,
            Err(_) => return corrupt(),
        };
        if rgba.len() != expected_len {
            return corrupt();
        }

        levels.push(Level {
            width,
            height,
            rgba,
        });
    }

    // No trailing garbage after the last level.
    if offset != bytes.len() {
        return corrupt();
    }

    Some(Pyramid { levels })
}

/// True when `pyramid` is a well-formed, fully coherent display pyramid for
/// an image of `image_width`x`image_height`: level 0's dims match the
/// decoded image exactly, and every subsequent level is exactly the
/// halving of the previous under `Pyramid::build`'s own rule (reusing
/// `fd_tiles::downsample_2x`'s dims so this can never drift from the real
/// rounding), terminating exactly where `Pyramid::build` would stop (the
/// last level's longest side <= `TILE_SIZE`, and no level after that).
///
/// This is a stricter check than `read_pyramid`'s per-level structural
/// validation: a file can pass every per-level check (magic, version,
/// length-matches-payload) while still being shape-incoherent -- e.g. a
/// level 1 that doesn't halve level 0, or a level0 that doesn't match the
/// image that was actually decoded. Two concrete costs of skipping this:
/// `Pyramid::tile`'s `l.width - x0` subtraction assumes a self-consistent
/// grid derived from a coherent pyramid and can underflow/panic against an
/// engineered level; and an incoherent file can claim up to
/// `MAX_PYRAMID_LEVELS` (32) levels each near `MAX_PYRAMID_LEVEL_BYTES` (2
/// GiB), an aggregate zstd-bomb far beyond any real pyramid -- full halving
/// coherence bounds the aggregate to ~4/3 of level 0 by construction (a
/// geometric series in quarters: 1 + 1/4 + 1/16 + ... < 4/3).
///
/// A cache-read pyramid MUST pass this before it ever reaches `tile()`;
/// failure is the corrupt class (delete + fall through to a fresh build),
/// not the mismatch-keep class -- unlike a stamp mismatch, a well-formed
/// per-level-but-incoherent-shape file can never become valid by retrying
/// the same read later.
pub fn pyramid_shape_is_coherent(pyramid: &Pyramid, image_width: u32, image_height: u32) -> bool {
    let Some(level0) = pyramid.levels.first() else {
        return false;
    };
    if level0.width != image_width || level0.height != image_height {
        return false;
    }

    let mut prev = level0;
    for level in &pyramid.levels[1..] {
        // Pyramid::build only ever appends another level while the
        // previous one's longest side still exceeds TILE_SIZE; a level
        // beyond that point is one Pyramid::build would never have built.
        if prev.width.max(prev.height) <= TILE_SIZE {
            return false;
        }
        let (_, expected_w, expected_h) = downsample_2x(&prev.rgba, prev.width, prev.height);
        if level.width != expected_w || level.height != expected_h {
            return false;
        }
        prev = level;
    }

    // The last level must actually be terminal -- Pyramid::build stops
    // exactly when the longest side drops to TILE_SIZE or below, never
    // earlier and never later.
    prev.width.max(prev.height) <= TILE_SIZE
}

/// Deletes the oldest-mtime `*.pyr` files under `dir` until the total size
/// of the remaining ones is at or under `budget_bytes`. `keep` (the file
/// just written by this same background task) is never deleted, even if
/// its mtime makes it look oldest -- pruning the entry that motivated the
/// prune in the first place would defeat the point of caching it. Every
/// step (dir listing, stat, delete) is best-effort: an IO error on any one
/// file is skipped rather than aborting the whole prune, mirroring the
/// write-failure discipline (debug-eprintln, never surfaced) one layer up
/// in the caller.
pub fn prune_pyramid_cache(dir: &Path, budget_bytes: u64, keep: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut files: Vec<(std::path::PathBuf, u64, std::time::SystemTime)> = Vec::new();
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("pyr") {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let Ok(mtime) = meta.modified() else {
            continue;
        };
        total += meta.len();
        files.push((path, meta.len(), mtime));
    }

    if total <= budget_bytes {
        return;
    }

    // Oldest mtime first; touch-on-read keeps recently-hit entries fresh so
    // this degrades to true LRU rather than FIFO (see read path caller).
    files.sort_by_key(|(_, _, mtime)| *mtime);

    for (path, size, _) in files {
        if total <= budget_bytes {
            break;
        }
        if path == keep {
            continue;
        }
        if std::fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_probs(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 97) as f32) / 96.0).collect()
    }

    fn stamp() -> SourceStamp {
        SourceStamp {
            size: 1234,
            mtime_nanos: 5_678_901_234,
        }
    }

    #[test]
    fn probs_round_trip_within_quantization() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let probs = synth_probs(64 * 48);
        let hash = [7u8; 32];
        let s = stamp();
        write_probs(&p, &probs, 64, 48, &hash, &s).unwrap();
        let back = read_probs(&p, 64, 48, &hash, &s).expect("cache readable");
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
        let s = stamp();
        write_probs(&p, &probs, 16, 16, &[1u8; 32], &s).unwrap();
        assert!(read_probs(&p, 16, 17, &[1u8; 32], &s).is_none()); // dims
        assert!(p.exists(), "dim mismatch is not corruption; file kept");
        assert!(read_probs(&p, 16, 16, &[2u8; 32], &s).is_none()); // detector changed
        assert!(p.exists(), "hash mismatch is not corruption; file kept");
        let mut bytes = std::fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&p, &bytes).unwrap();
        assert!(read_probs(&p, 16, 16, &[1u8; 32], &s).is_none()); // corrupt payload
        assert!(!p.exists(), "corrupt file deleted on sight");
        assert!(read_probs(&p, 16, 16, &[1u8; 32], &s).is_none()); // absent -> None
    }

    #[test]
    fn probs_reject_stamp_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let probs = synth_probs(16 * 16);
        let s = stamp();
        write_probs(&p, &probs, 16, 16, &[1u8; 32], &s).unwrap();
        let mismatched = SourceStamp {
            size: s.size + 1,
            mtime_nanos: s.mtime_nanos,
        };
        assert!(read_probs(&p, 16, 16, &[1u8; 32], &mismatched).is_none());
        assert!(p.exists(), "stamp mismatch is not corruption; file kept");
    }

    #[test]
    fn old_version_probs_purge_on_sight() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let probs = synth_probs(16 * 16);
        let s = stamp();
        write_probs(&p, &probs, 16, 16, &[1u8; 32], &s).unwrap();

        // Patch the version field (offset 8..12) from 2 down to 1 -- this
        // simulates a pre-stamp cache file surviving an upgrade; it must be
        // treated as corrupt (unknown/unsupported version) and purged, not
        // silently reused without a stamp check.
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        std::fs::write(&p, &bytes).unwrap();

        assert!(read_probs(&p, 16, 16, &[1u8; 32], &s).is_none());
        assert!(!p.exists(), "old version file is deleted on sight");
    }

    #[test]
    fn crafted_length_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let probs = synth_probs(16 * 16);
        let hash = [1u8; 32];
        let s = stamp();
        write_probs(&p, &probs, 16, 16, &hash, &s).unwrap();

        // Overwrite the compressed_len field with u64::MAX. Header layout:
        // magic(8) version(4) width(4) height(4) hash(32) size(8) mtime(8) len(8)
        let len_off = 8 + 4 + 4 + 4 + 32 + 8 + 8;
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[len_off..len_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&p, &bytes).unwrap();

        assert!(read_probs(&p, 16, 16, &hash, &s).is_none());
        assert!(!p.exists(), "crafted length is treated as corruption");
    }

    #[test]
    fn oversized_decompression_is_rejected_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        let width: u32 = 16;
        let height: u32 = 16; // expected decompressed size = 256 bytes
        let s = stamp();

        // zstd frame that decompresses to 1MB of zeros -- wildly more than
        // the 256 bytes the header claims.
        let bomb_payload = vec![0u8; 1024 * 1024];
        let compressed = zstd::encode_all(&bomb_payload[..], 3).unwrap();

        let mut header = Vec::new();
        header.extend_from_slice(PROBS_MAGIC);
        header.extend_from_slice(&2u32.to_le_bytes());
        header.extend_from_slice(&width.to_le_bytes());
        header.extend_from_slice(&height.to_le_bytes());
        header.extend_from_slice(&[3u8; 32]);
        header.extend_from_slice(&s.size.to_le_bytes());
        header.extend_from_slice(&s.mtime_nanos.to_le_bytes());
        header.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        header.extend_from_slice(&compressed);

        std::fs::write(&p, &header).unwrap();

        // Bounded decompression must reject (not allocate 1MB and then
        // reject after the fact) -- we can only assert on the outcome:
        // rejection plus deletion, since allocation size isn't observable
        // from a test.
        assert!(read_probs(&p, width, height, &[3u8; 32], &s).is_none());
        assert!(
            !p.exists(),
            "oversized decompression is treated as corruption"
        );
    }

    #[test]
    fn probs_write_is_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.probs");
        write_probs(&p, &synth_probs(8 * 8), 8, 8, &[0u8; 32], &stamp()).unwrap();
        // no tmp siblings left behind
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter(|e| e.as_ref().unwrap().file_name() != "a.probs")
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    fn noisy8(w: u32, h: u32) -> fd_io::ImageBuf {
        let n = (w * h * 3) as usize;
        let mut s = 0x9E3779B97F4A7C15u64;
        let data: Vec<u8> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 56) as u8
            })
            .collect();
        fd_io::ImageBuf {
            width: w,
            height: h,
            channels: 3,
            data: fd_io::PixelData::U8(data),
            icc: None,
            exif: None,
        }
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
    fn heal_delta_round_trips_bit_exact_u8() {
        // The reference rolls are 8-bit JPEGs, so U8 is the dominant
        // production depth; the gather/scatter branches are depth-specific
        // and each needs its own bit-exact pin.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.heal");
        let original = noisy8(40, 30);
        let mut healed = original.clone();
        let mut mask = vec![false; 40 * 30];
        for y in 5..12 {
            for x in 20..35 {
                mask[y * 40 + x] = true;
            }
        }
        if let fd_io::PixelData::U8(v) = &mut healed.data {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    for c in 0..3 {
                        v[i * 3 + c] = v[i * 3 + c].wrapping_add(10 + c as u8);
                    }
                }
            }
        }
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32], &stamp());
        write_heal(&p, &original, &healed, &mask, &prov).unwrap();
        let (back, back_mask) = read_heal(&p, &original, &prov).expect("cache hit");
        assert_eq!(back_mask, mask);
        let (fd_io::PixelData::U8(a), fd_io::PixelData::U8(b)) = (&healed.data, &back.data) else {
            panic!("depth changed");
        };
        assert_eq!(a, b, "U8 reconstruction must be bit-exact");
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
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32], &stamp());
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
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32], &stamp());
        write_heal(&p, &original, &healed, &mask, &prov).unwrap();
        let other = heal_provenance(0.51, 2, &[], &[3u8; 32], &[4u8; 32], &stamp());
        assert!(read_heal(&p, &original, &other).is_none());
        assert!(p.exists(), "provenance miss keeps the file");
    }

    #[test]
    fn heal_provenance_distinguishes_every_input() {
        let base = heal_provenance(0.5, 2, &[], &[0u8; 32], &[0u8; 32], &stamp());
        let stroke = crate::masks::Stroke {
            erase: false,
            radius: 5.0,
            points: vec![[1.0, 2.0]],
        };
        assert_ne!(
            base,
            heal_provenance(0.6, 2, &[], &[0u8; 32], &[0u8; 32], &stamp())
        );
        assert_ne!(
            base,
            heal_provenance(0.5, 3, &[], &[0u8; 32], &[0u8; 32], &stamp())
        );
        assert_ne!(
            base,
            heal_provenance(
                0.5,
                2,
                std::slice::from_ref(&stroke),
                &[0u8; 32],
                &[0u8; 32],
                &stamp()
            )
        );
        assert_ne!(
            base,
            heal_provenance(0.5, 2, &[], &[1u8; 32], &[0u8; 32], &stamp())
        );
        assert_ne!(
            base,
            heal_provenance(0.5, 2, &[], &[0u8; 32], &[1u8; 32], &stamp())
        );
        let other_stamp = SourceStamp {
            size: stamp().size + 1,
            mtime_nanos: stamp().mtime_nanos,
        };
        assert_ne!(
            base,
            heal_provenance(0.5, 2, &[], &[0u8; 32], &[0u8; 32], &other_stamp)
        );
    }

    #[test]
    fn old_magic_heal_purges() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.heal");
        let original = noisy16(16, 16);
        let healed = original.clone();
        let mask = vec![false; 256];
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32], &stamp());
        write_heal(&p, &original, &healed, &mask, &prov).unwrap();

        // Patch the magic to the pre-stamp value -- a heal file written
        // before the stamp was folded into provenance must not be reused;
        // it should purge on first read, not linger unmatched forever.
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[0..8].copy_from_slice(b"UNDHEAL1");
        std::fs::write(&p, &bytes).unwrap();

        assert!(read_heal(&p, &original, &prov).is_none());
        assert!(!p.exists(), "old magic heal file is deleted on sight");
    }

    #[test]
    fn heal_crafted_values_length_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.heal");
        let original = noisy16(16, 16);
        let healed = original.clone();
        let mask = vec![false; 256];
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32], &stamp());
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
        let prov = heal_provenance(0.5, 2, &[], &[3u8; 32], &[4u8; 32], &stamp());
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

    fn synth_level(width: u32, height: u32, seed: u8) -> fd_tiles::Level {
        let n = (width * height * 4) as usize;
        let rgba: Vec<u8> = (0..n).map(|i| seed.wrapping_add((i % 251) as u8)).collect();
        fd_tiles::Level {
            width,
            height,
            rgba,
        }
    }

    fn synth_pyramid() -> fd_tiles::Pyramid {
        fd_tiles::Pyramid {
            levels: vec![synth_level(4, 4, 11), synth_level(2, 2, 97)],
        }
    }

    #[test]
    fn pyramid_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.pyr");
        let pyramid = synth_pyramid();
        let s = stamp();
        write_pyramid(&p, &pyramid, &s).unwrap();
        let back = read_pyramid(&p, &s).expect("cache readable");
        assert_eq!(back.levels.len(), pyramid.levels.len());
        for (a, b) in pyramid.levels.iter().zip(&back.levels) {
            assert_eq!(a.width, b.width);
            assert_eq!(a.height, b.height);
            assert_eq!(a.rgba, b.rgba);
        }
    }

    #[test]
    fn pyramid_stamp_mismatch_returns_none_and_keeps_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.pyr");
        let pyramid = synth_pyramid();
        let s = stamp();
        write_pyramid(&p, &pyramid, &s).unwrap();
        let mismatched = SourceStamp {
            size: s.size + 1,
            mtime_nanos: s.mtime_nanos,
        };
        assert!(read_pyramid(&p, &mismatched).is_none());
        assert!(p.exists(), "stamp mismatch is not corruption; file kept");
    }

    #[test]
    fn corrupt_pyramid_is_deleted_on_sight() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.pyr");
        let pyramid = synth_pyramid();
        let s = stamp();
        write_pyramid(&p, &pyramid, &s).unwrap();

        let mut bytes = std::fs::read(&p).unwrap();
        bytes[0..8].copy_from_slice(b"UNDPYRA0");
        std::fs::write(&p, &bytes).unwrap();

        assert!(read_pyramid(&p, &s).is_none());
        assert!(!p.exists(), "corrupt magic file is deleted on sight");
    }

    #[test]
    fn truncated_pyramid_is_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.pyr");
        let pyramid = synth_pyramid();
        let s = stamp();
        write_pyramid(&p, &pyramid, &s).unwrap();

        let bytes = std::fs::read(&p).unwrap();
        // Truncate mid-level: keep the top-level header plus the first
        // level's own header, but cut off partway through its zstd payload.
        let truncated = &bytes[..bytes.len() - 4];
        std::fs::write(&p, truncated).unwrap();

        assert!(read_pyramid(&p, &s).is_none());
        assert!(!p.exists(), "truncated file is deleted on sight");
    }

    #[test]
    fn pyramid_level_length_lie_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.pyr");
        let s = stamp();

        // Hand-craft a one-level pyramid whose header claims 4x4 (64 bytes
        // decompressed) but whose zstd payload actually decompresses to
        // 2x2 (16 bytes) worth of data -- width*height*4 disagrees with the
        // real decompressed length.
        let lying_payload = [0u8; 16];
        let compressed = zstd::encode_all(&lying_payload[..], 1).unwrap();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(PYRAMID_MAGIC);
        bytes.extend_from_slice(&PYRAMID_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // level_count
        bytes.extend_from_slice(&s.size.to_le_bytes());
        bytes.extend_from_slice(&s.mtime_nanos.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes()); // width
        bytes.extend_from_slice(&4u32.to_le_bytes()); // height
        bytes.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&compressed);

        std::fs::write(&p, &bytes).unwrap();

        assert!(read_pyramid(&p, &s).is_none());
        assert!(!p.exists(), "level length lie is treated as corruption");
    }

    #[test]
    fn coherent_pyramid_shape_passes() {
        // level0 1000x1000 (> TILE_SIZE=512) halves to level1 500x500 (<=
        // TILE_SIZE, terminal); halving math must match Pyramid::build's
        // downsample_2x (div_ceil(2).max(1)) exactly.
        let pyramid = fd_tiles::Pyramid {
            levels: vec![synth_level(1000, 1000, 11), synth_level(500, 500, 97)],
        };
        assert!(pyramid_shape_is_coherent(&pyramid, 1000, 1000));
    }

    #[test]
    fn wrong_level0_dims_fails_shape_check() {
        let pyramid = fd_tiles::Pyramid {
            levels: vec![synth_level(1000, 1000, 11), synth_level(500, 500, 97)],
        };
        assert!(!pyramid_shape_is_coherent(&pyramid, 1001, 1000));
        assert!(!pyramid_shape_is_coherent(&pyramid, 1000, 1001));
    }

    #[test]
    fn level1_too_big_fails_shape_check() {
        // level0 1000x1000 should halve to 500x500; claiming 600x600 is
        // incoherent even though it's individually well-formed.
        let pyramid = fd_tiles::Pyramid {
            levels: vec![synth_level(1000, 1000, 11), synth_level(600, 600, 97)],
        };
        assert!(!pyramid_shape_is_coherent(&pyramid, 1000, 1000));
    }

    #[test]
    fn halving_off_by_one_fails_shape_check() {
        // Real rule for an odd dimension is div_ceil(2).max(1): 1001 -> 501,
        // not 500.
        let pyramid = fd_tiles::Pyramid {
            levels: vec![synth_level(1001, 1001, 11), synth_level(500, 500, 97)],
        };
        assert!(!pyramid_shape_is_coherent(&pyramid, 1001, 1001));
    }

    #[test]
    fn missing_final_level_fails_shape_check() {
        // A pyramid that stops before reaching <= TILE_SIZE is incoherent
        // (Pyramid::build never stops early). 2000 -> 1000 is a correct
        // halving, but 1000 > TILE_SIZE so Pyramid::build would keep going;
        // stopping here is missing the terminal level.
        let pyramid = fd_tiles::Pyramid {
            levels: vec![synth_level(4000, 4000, 11), synth_level(2000, 2000, 97)],
        };
        assert!(!pyramid_shape_is_coherent(&pyramid, 4000, 4000));
    }

    #[test]
    fn empty_pyramid_fails_shape_check() {
        let pyramid = fd_tiles::Pyramid { levels: vec![] };
        assert!(!pyramid_shape_is_coherent(&pyramid, 4, 4));
    }

    #[test]
    fn single_level_pyramid_at_or_under_tile_size_is_coherent() {
        // A source image no larger than TILE_SIZE in either dimension never
        // grows a second level under Pyramid::build.
        let pyramid = fd_tiles::Pyramid {
            levels: vec![synth_level(4, 4, 11)],
        };
        assert!(pyramid_shape_is_coherent(&pyramid, 4, 4));
    }

    #[test]
    fn extra_level_past_terminal_fails_shape_check() {
        // level0 4x4 is already <= TILE_SIZE; Pyramid::build would never
        // have appended a second level at all.
        let pyramid = fd_tiles::Pyramid {
            levels: vec![synth_level(4, 4, 11), synth_level(2, 2, 97)],
        };
        assert!(!pyramid_shape_is_coherent(&pyramid, 4, 4));
    }

    #[test]
    fn prune_pyramid_cache_forces_out_oldest_when_over_budget() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.pyr");
        let b = dir.path().join("b.pyr");
        let c = dir.path().join("c.pyr");
        std::fs::write(&a, vec![0u8; 100]).unwrap();
        std::fs::write(&b, vec![0u8; 100]).unwrap();
        std::fs::write(&c, vec![0u8; 100]).unwrap();
        let now = std::time::SystemTime::now();
        std::fs::File::open(&a)
            .unwrap()
            .set_modified(now - std::time::Duration::from_secs(30))
            .unwrap();
        std::fs::File::open(&b)
            .unwrap()
            .set_modified(now - std::time::Duration::from_secs(20))
            .unwrap();
        std::fs::File::open(&c)
            .unwrap()
            .set_modified(now - std::time::Duration::from_secs(10))
            .unwrap();

        // 300 bytes total, budget 250: exactly one file (the oldest, a) must go.
        prune_pyramid_cache(dir.path(), 250, &c);

        assert!(!a.exists(), "oldest file is pruned first");
        assert!(b.exists());
        assert!(c.exists());
    }

    #[test]
    fn prune_pyramid_cache_never_deletes_the_just_written_file_even_if_oldest() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.pyr");
        let b = dir.path().join("b.pyr");
        std::fs::write(&a, vec![0u8; 100]).unwrap();
        std::fs::write(&b, vec![0u8; 100]).unwrap();
        let now = std::time::SystemTime::now();
        // `a` is the just-written file but has an (artificially) older mtime
        // than `b` -- it must still survive.
        std::fs::File::open(&a)
            .unwrap()
            .set_modified(now - std::time::Duration::from_secs(30))
            .unwrap();
        std::fs::File::open(&b)
            .unwrap()
            .set_modified(now - std::time::Duration::from_secs(20))
            .unwrap();

        prune_pyramid_cache(dir.path(), 50, &a);

        assert!(a.exists(), "the just-written file is never pruned");
        assert!(!b.exists(), "the only other file is pruned instead");
    }

    #[test]
    fn prune_pyramid_cache_is_a_noop_under_budget() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.pyr");
        std::fs::write(&a, vec![0u8; 100]).unwrap();
        prune_pyramid_cache(dir.path(), 1_000_000, &a);
        assert!(a.exists());
    }

    #[test]
    fn pyramid_giant_claimed_level_is_rejected_without_allocating() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.pyr");
        let s = stamp();

        // A ~100-byte crafted file claiming a 65535x65535 level: expected_len
        // is ~17GB, which zstd::bulk::decompress would Vec::with_capacity
        // EAGERLY before decompressing a single byte. The cap check must
        // reject this as corrupt (delete + None) BEFORE the decompress call,
        // so this test never drives a giant allocation.
        let tiny_payload = [0u8; 16];
        let compressed = zstd::encode_all(&tiny_payload[..], 1).unwrap();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(PYRAMID_MAGIC);
        bytes.extend_from_slice(&PYRAMID_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // level_count
        bytes.extend_from_slice(&s.size.to_le_bytes());
        bytes.extend_from_slice(&s.mtime_nanos.to_le_bytes());
        bytes.extend_from_slice(&65535u32.to_le_bytes()); // width
        bytes.extend_from_slice(&65535u32.to_le_bytes()); // height
        bytes.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&compressed);

        std::fs::write(&p, &bytes).unwrap();

        assert!(read_pyramid(&p, &s).is_none());
        assert!(!p.exists(), "giant claimed level is treated as corruption");
    }
}
