import { describe, expect, it } from "vitest";
import { composeQueueEntries } from "./queue";

const frames = [{ file_name: "a.tif" }, { file_name: "b.tif" }, { file_name: "c.tif" }];

describe("composeQueueEntries", () => {
  it("returns an empty list when nothing is running or queued", () => {
    expect(composeQueueEntries([], [], frames)).toEqual([]);
  });

  it("lists running jobs before queued jobs", () => {
    const running = [{ index: 1, kind: "heal" as const }];
    const snapshot = [{ index: 0, kind: "detect" as const }];
    expect(composeQueueEntries(running, snapshot, frames)).toEqual([
      { key: "heal:1", kind: "heal", index: 1, label: "b.tif · heal", state: "running" },
      { key: "detect:0", kind: "detect", index: 0, label: "a.tif · detect", state: "queued" },
    ]);
  });

  it("preserves the snapshot's queue order for queued entries", () => {
    const snapshot = [
      { index: 2, kind: "export" as const },
      { index: 0, kind: "heal" as const },
    ];
    expect(composeQueueEntries([], snapshot, frames)).toEqual([
      { key: "export:2", kind: "export", index: 2, label: "c.tif · export", state: "queued" },
      { key: "heal:0", kind: "heal", index: 0, label: "a.tif · heal", state: "queued" },
    ]);
  });

  it("drops a snapshot row whose key matches a running row", () => {
    const running = [{ index: 0, kind: "heal" as const }];
    const snapshot = [
      { index: 0, kind: "heal" as const }, // same kind+index as running: dropped
      { index: 1, kind: "heal" as const }, // distinct: kept
    ];
    expect(composeQueueEntries(running, snapshot, frames)).toEqual([
      { key: "heal:0", kind: "heal", index: 0, label: "a.tif · heal", state: "running" },
      { key: "heal:1", kind: "heal", index: 1, label: "b.tif · heal", state: "queued" },
    ]);
  });

  it("attaches done/total progress to the running entry", () => {
    const running = [{ index: 0, kind: "heal" as const }];
    const progress = { done: 3, total: 10 };
    expect(composeQueueEntries(running, [], frames, progress)).toEqual([
      { key: "heal:0", kind: "heal", index: 0, label: "a.tif · heal", state: "running", progress },
    ]);
  });

  it("attaches stage progress to the running entry", () => {
    const running = [{ index: 1, kind: "export" as const }];
    const progress = { stage: "writing" };
    expect(composeQueueEntries(running, [], frames, progress)).toEqual([
      {
        key: "export:1",
        kind: "export",
        index: 1,
        label: "b.tif · export",
        state: "running",
        progress,
      },
    ]);
  });

  it("never attaches progress to a queued entry", () => {
    const snapshot = [{ index: 0, kind: "detect" as const }];
    const progress = { done: 1, total: 2 };
    const result = composeQueueEntries([], snapshot, frames, progress);
    expect(result).toEqual([
      { key: "detect:0", kind: "detect", index: 0, label: "a.tif · detect", state: "queued" },
    ]);
    expect(result[0]).not.toHaveProperty("progress");
  });

  it("omits progress when none is given", () => {
    const running = [{ index: 0, kind: "heal" as const }];
    const result = composeQueueEntries(running, [], frames);
    expect(result).toEqual([
      { key: "heal:0", kind: "heal", index: 0, label: "a.tif · heal", state: "running" },
    ]);
    expect(result[0]).not.toHaveProperty("progress");
  });

  it("marks the running entry cancelling when its key matches", () => {
    const running = [{ index: 1, kind: "heal" as const }];
    const snapshot = [{ index: 2, kind: "heal" as const }];
    expect(composeQueueEntries(running, snapshot, frames, null, "heal:1")).toEqual([
      {
        key: "heal:1",
        kind: "heal",
        index: 1,
        label: "b.tif · heal",
        state: "running",
        cancelling: true,
      },
      { key: "heal:2", kind: "heal", index: 2, label: "c.tif · heal", state: "queued" },
    ]);
  });

  it("never marks a queued entry cancelling", () => {
    // A queued job is removed instantly on cancel, so a stale cancelling key
    // pointing at a queued row must not flag it.
    const snapshot = [{ index: 0, kind: "detect" as const }];
    const result = composeQueueEntries([], snapshot, frames, null, "detect:0");
    expect(result[0]).not.toHaveProperty("cancelling");
  });

  it("ignores a cancelling key that matches nothing", () => {
    const running = [{ index: 0, kind: "heal" as const }];
    const result = composeQueueEntries(running, [], frames, null, "export:5");
    expect(result[0]).not.toHaveProperty("cancelling");
  });
});
