# Data capture and labelling

All of this lives under training/data/ (gitignored).

## Blank film scans -> data/blank_scans/

Purpose: harvest real dust, scratches, and hairs.

- Scan unexposed-but-developed film, blank leaders, and film edges.
  Do NOT clean them first. Dusty is the point.
- At least 20 strips, colour and B&W stocks mixed.
- Cover both polarities: negatives (including C-41 with its orange
  mask) AND positives (slides, or B&W reversal). Dust presents with
  different polarity and contrast on each, and the model only learns
  what it has seen. If you shoot both, scan both.
- 3200 dpi or higher, 16-bit TIFF, no scanner dust removal (ICE OFF),
  no sharpening, flat/linear profile if the software allows it.
- Name files freely; the harvester walks the whole directory.

## Clean images -> data/clean/

Any large set of defect-free photographs (your own digital photos work).
1000+ images, mixed subjects, some with fine detail (branches, birds,
stars, fabric) precisely because those are the false-positive traps.

## Benchmark roll -> data/benchmark/

- frames/: 20+ real scanned frames WITH their real defects, colour and
  B&W mixed, negatives and positives both if you shoot both, varied
  subjects including night skies and fine detail.
  Named 0001.tif, 0002.tif, ...
- labels/: for each frame, 0001.png etc (same stem): uint8, single
  channel, pixel values 0 background, 1 dust, 2 scratch, 3 hair.
  Paint in any editor over the frame; exact edges matter less than
  covering each defect's core (detection is scored at 50% coverage).
- retouch4me/ (optional): the same frames healed by the competitor,
  same filenames. The harness recovers their mask by before/after diff.

The labelled roll is sacred: never train on it, never tune on it by eye.

## If your scanner exposes an infrared channel

Dust and scratches occlude IR on dye-based film (colour negative and
slide; NOT silver B&W or Kodachrome). If your scanning software can
emit the raw IR plane (with hardware dust removal still OFF for the
RGB), capture it alongside every scan above: IR-derived masks are
near-free ground-truth labels for the benchmark roll and can seed
training labels, cutting the hand-painting substantially. Tracked as
TheUnduster-jb2.
