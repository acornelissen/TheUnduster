import { describe, expect, it } from "vitest";
import { composeActivity, composeLeft, composeRight, formatModelProgress } from "./status";

describe("composeActivity", () => {
  const base = {
    modelStatus: "loaded" as const,
    modelProgressText: "0%",
    exporting: false,
    exportDetail: null,
    isHealing: false,
    healProgress: null,
    isDetecting: false,
    detectProgress: null,
    roll: false,
    scanDone: true,
    scannedCount: 0,
    totalCount: 0,
  };

  it("is null when nothing is happening", () => {
    expect(composeActivity(base)).toBeNull();
  });

  it("prioritizes model downloading over everything else", () => {
    expect(
      composeActivity({
        ...base,
        modelStatus: "downloading",
        modelProgressText: "42%",
        exporting: true,
        isHealing: true,
        isDetecting: true,
      }),
    ).toBe("downloading healing model 42%");
  });

  it("prioritizes exporting over healing/detecting", () => {
    expect(
      composeActivity({
        ...base,
        exporting: true,
        exportDetail: "raw0002.jpg: healing 3/9",
        isHealing: true,
        isDetecting: true,
      }),
    ).toBe("exporting · raw0002.jpg: healing 3/9");
  });

  it("renders exporting without per-frame detail when none is available", () => {
    expect(composeActivity({ ...base, exporting: true, exportDetail: null })).toBe("exporting");
  });

  it("prioritizes current-frame healing over detecting", () => {
    expect(
      composeActivity({
        ...base,
        isHealing: true,
        healProgress: { done: 34, total: 87 },
        isDetecting: true,
      }),
    ).toBe("healing 34/87");
  });

  it("renders healing without progress counts when none are available yet", () => {
    expect(composeActivity({ ...base, isHealing: true, healProgress: null })).toBe("healing");
  });

  it("falls back to detecting when nothing higher-priority is active", () => {
    expect(composeActivity({ ...base, isDetecting: true })).toBe("detecting");
  });

  it("shows tile progress while detecting once it is available", () => {
    expect(
      composeActivity({
        ...base,
        isDetecting: true,
        detectProgress: { done: 12, total: 870 },
      }),
    ).toBe("detecting (12/870 tiles)");
  });

  it("renders detecting without tile counts when none are available yet", () => {
    expect(composeActivity({ ...base, isDetecting: true, detectProgress: null })).toBe(
      "detecting",
    );
  });

  it("prioritizes current-frame healing over a detecting tile progress", () => {
    expect(
      composeActivity({
        ...base,
        isHealing: true,
        healProgress: { done: 34, total: 87 },
        isDetecting: true,
        detectProgress: { done: 12, total: 870 },
      }),
    ).toBe("healing 34/87");
  });

  it("falls back to roll scanning when the roll has not finished scanning", () => {
    expect(
      composeActivity({ ...base, roll: true, scanDone: false, scannedCount: 3, totalCount: 8 }),
    ).toBe("scanning 3/8");
  });

  it("is null once roll scanning has finished and nothing else is active", () => {
    expect(
      composeActivity({ ...base, roll: true, scanDone: true, scannedCount: 8, totalCount: 8 }),
    ).toBeNull();
  });
});

describe("composeLeft", () => {
  const base = {
    fileName: "raw0002.jpg",
    position: null as { index: number; total: number } | null,
    defectCount: null as number | null,
    threshold: 0.5,
    healed: false,
    healedCached: false,
    healStale: false,
    brushStatus: null as string | null,
  };

  it("shows not-yet-detected when the frame has no defect count", () => {
    expect(composeLeft(base)).toBe("raw0002.jpg  not yet detected");
  });

  it("includes roll position when present", () => {
    expect(composeLeft({ ...base, position: { index: 2, total: 4 } })).toBe(
      "raw0002.jpg  3/4  not yet detected",
    );
  });

  it("includes defect count and threshold when known", () => {
    expect(composeLeft({ ...base, defectCount: 11 })).toBe(
      "raw0002.jpg  11 defects at 0.50",
    );
  });

  it("uses singular defect wording for exactly one defect", () => {
    expect(composeLeft({ ...base, defectCount: 1 })).toBe("raw0002.jpg  1 defect at 0.50");
  });

  it("renders the defect count at the live threshold value, not a fixed 0.50", () => {
    expect(composeLeft({ ...base, defectCount: 7, threshold: 0.35 })).toBe(
      "raw0002.jpg  7 defects at 0.35",
    );
  });

  it("shows the healed-state indicator when the frame is healed", () => {
    expect(composeLeft({ ...base, defectCount: 3, healed: true })).toBe(
      "raw0002.jpg  3 defects at 0.50  healed (space compares)",
    );
  });

  it("shows a bare healed indicator (no space-compare hint) when only the on-disk cache is healed", () => {
    // healed=false, healedCached=true: the registry has no live healed
    // tiles to compare against (an evicted or reopened frame with a
    // matching heal cache) -- SPACE has nothing to toggle yet, so the hint
    // must not be offered.
    expect(composeLeft({ ...base, defectCount: 3, healedCached: true })).toBe(
      "raw0002.jpg  3 defects at 0.50  healed",
    );
  });

  it("prefers the live space-compare hint when both the registry and the cache are healed", () => {
    expect(
      composeLeft({ ...base, defectCount: 3, healed: true, healedCached: true }),
    ).toBe("raw0002.jpg  3 defects at 0.50  healed (space compares)");
  });

  it("shows no healed indicator when neither the registry nor the cache is healed", () => {
    expect(composeLeft({ ...base, defectCount: 3 })).toBe(
      "raw0002.jpg  3 defects at 0.50",
    );
  });

  it("appends the stale-heal hint to the left zone when the frame's heal is stale", () => {
    expect(composeLeft({ ...base, defectCount: 3, healStale: true })).toBe(
      "raw0002.jpg  3 defects at 0.50  heal stale (h re-heals)",
    );
  });

  it("appends the brush status to the left zone while the brush is active", () => {
    expect(composeLeft({ ...base, defectCount: 3, brushStatus: "brush 24px" })).toBe(
      "raw0002.jpg  3 defects at 0.50  brush 24px",
    );
  });

  it("composes every fragment together in order", () => {
    expect(
      composeLeft({
        fileName: "raw0002.jpg",
        position: { index: 2, total: 4 },
        defectCount: 11,
        threshold: 0.5,
        healed: true,
        healedCached: true,
        healStale: true,
        brushStatus: "erase 12px",
      }),
    ).toBe(
      "raw0002.jpg  3/4  11 defects at 0.50  healed (space compares)  heal stale (h re-heals)  erase 12px",
    );
  });

  it("renders an empty string when there is no frame identity at all", () => {
    expect(composeLeft({ ...base, fileName: null })).toBe("");
  });
});

describe("formatModelProgress", () => {
  const MB = 1024 * 1024;

  it("shows received of total with a percentage when the total is known", () => {
    expect(formatModelProgress(82 * MB, 207 * MB)).toBe("82 / 207 MB (39%)");
  });

  it("shows received alone when the server sent no content length", () => {
    expect(formatModelProgress(82 * MB, null)).toBe("82 MB");
  });

  it("starts at zero without dividing by anything odd", () => {
    expect(formatModelProgress(0, 207 * MB)).toBe("0 / 207 MB (0%)");
    expect(formatModelProgress(0, null)).toBe("0 MB");
  });

  it("treats a zero total as unknown rather than reporting infinity", () => {
    expect(formatModelProgress(5 * MB, 0)).toBe("5 MB");
  });
});

describe("composeRight", () => {
  const base = {
    roll: false,
    approvedCount: 0,
    totalCount: 0,
    queuedJobCount: 0,
    healingEngine: "lama" as const,
  };

  it("always shows the healing engine, even with no roll and no jobs", () => {
    // The indicator is persistent on purpose: the operator asked for an
    // always-visible answer to "what is healing my frames right now".
    expect(composeRight(base)).toBe("healing: LaMa");
  });

  it("names the placeholder engine when the inpainter is the dev fixture", () => {
    expect(composeRight({ ...base, healingEngine: "placeholder" })).toBe(
      "healing: placeholder model",
    );
  });

  it("names classical-only healing when no inpainting model is loaded", () => {
    expect(composeRight({ ...base, healingEngine: "classical" })).toBe(
      "healing: classical only",
    );
  });

  it("renders approved counts for a roll", () => {
    expect(composeRight({ ...base, roll: true, approvedCount: 2, totalCount: 4 })).toBe(
      "healing: LaMa  2/4 approved",
    );
  });

  it("appends queued job count when jobs are queued", () => {
    expect(
      composeRight({ ...base, roll: true, approvedCount: 2, totalCount: 4, queuedJobCount: 3 }),
    ).toBe("healing: LaMa  2/4 approved  3 jobs queued");
  });

  it("uses singular job wording for exactly one queued job", () => {
    expect(
      composeRight({ ...base, roll: true, approvedCount: 0, totalCount: 4, queuedJobCount: 1 }),
    ).toBe("healing: LaMa  0/4 approved  1 job queued");
  });

  it("keeps the engine indicator ahead of roll and queue info", () => {
    // First in the zone so the zone's ellipsis overflow can never truncate
    // it away -- the placeholder warning especially must stay unmissable.
    expect(
      composeRight({
        ...base,
        roll: true,
        approvedCount: 2,
        totalCount: 4,
        queuedJobCount: 1,
        healingEngine: "placeholder",
      }),
    ).toBe("healing: placeholder model  2/4 approved  1 job queued");
  });
});
