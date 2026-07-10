import { describe, expect, it } from "vitest";
import { composeActivity, composeLeft, composeRight } from "./status";

describe("composeActivity", () => {
  const base = {
    modelStatus: "loaded" as const,
    modelProgressText: "0%",
    exporting: false,
    exportDetail: null,
    isHealing: false,
    healProgress: null,
    isDetecting: false,
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
    ).toBe("exporting — raw0002.jpg: healing 3/9");
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

  it("shows the healed-state indicator when the frame is healed", () => {
    expect(composeLeft({ ...base, defectCount: 3, healed: true })).toBe(
      "raw0002.jpg  3 defects at 0.50  healed (space compares)",
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

describe("composeRight", () => {
  it("renders nothing when there is no roll and no queued jobs", () => {
    expect(composeRight({ roll: false, approvedCount: 0, totalCount: 0, queuedJobCount: 0 })).toBe(
      "",
    );
  });

  it("renders approved counts for a roll", () => {
    expect(
      composeRight({ roll: true, approvedCount: 2, totalCount: 4, queuedJobCount: 0 }),
    ).toBe("2/4 approved");
  });

  it("appends queued job count when jobs are queued", () => {
    expect(
      composeRight({ roll: true, approvedCount: 2, totalCount: 4, queuedJobCount: 3 }),
    ).toBe("2/4 approved  3 jobs queued");
  });

  it("uses singular job wording for exactly one queued job", () => {
    expect(
      composeRight({ roll: true, approvedCount: 0, totalCount: 4, queuedJobCount: 1 }),
    ).toBe("0/4 approved  1 job queued");
  });

  it("renders only the queued job count outside roll mode", () => {
    expect(
      composeRight({ roll: false, approvedCount: 0, totalCount: 0, queuedJobCount: 2 }),
    ).toBe("2 jobs queued");
  });
});
