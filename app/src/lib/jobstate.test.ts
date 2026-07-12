import { describe, expect, it } from "vitest";
import { countQueuedJobs, isExportRunning, runningKindAt, type JobStates } from "./jobstate";

describe("runningKindAt", () => {
  const states: JobStates = {
    0: { state: "running", kind: "detect" },
    1: { state: "queued", kind: "heal" },
    2: { state: "running", kind: "export" },
  };

  it("returns the kind of a job running at the index", () => {
    expect(runningKindAt(states, 0, true)).toBe("detect");
    expect(runningKindAt(states, 2, true)).toBe("export");
  });

  it("returns null for a merely-queued job", () => {
    expect(runningKindAt(states, 1, true)).toBeNull();
  });

  it("returns null for an index with no job", () => {
    expect(runningKindAt(states, 5, true)).toBeNull();
  });

  it("returns null when no roll is open, whatever the map holds", () => {
    // Single-image mode never consults jobStates for current-frame status.
    expect(runningKindAt(states, 0, false)).toBeNull();
  });
});

describe("countQueuedJobs", () => {
  it("counts jobs of every non-prefetch kind and state", () => {
    const states: JobStates = {
      0: { state: "running", kind: "detect" },
      1: { state: "queued", kind: "heal" },
      2: { state: "queued", kind: "export" },
    };
    expect(countQueuedJobs(states)).toBe(3);
  });

  it("excludes prefetch jobs", () => {
    const states: JobStates = {
      0: { state: "running", kind: "detect" },
      1: { state: "queued", kind: "prefetch" },
      2: { state: "running", kind: "prefetch" },
    };
    expect(countQueuedJobs(states)).toBe(1);
  });

  it("is zero for an empty map", () => {
    expect(countQueuedJobs({})).toBe(0);
  });
});

describe("isExportRunning", () => {
  it("is true when an export job is running", () => {
    expect(isExportRunning({ 3: { state: "running", kind: "export" } })).toBe(true);
  });

  it("is false for a merely-queued export", () => {
    expect(isExportRunning({ 3: { state: "queued", kind: "export" } })).toBe(false);
  });

  it("is false when only non-export jobs run", () => {
    expect(
      isExportRunning({
        0: { state: "running", kind: "detect" },
        1: { state: "running", kind: "heal" },
      }),
    ).toBe(false);
  });

  it("is false for an empty map", () => {
    expect(isExportRunning({})).toBe(false);
  });
});
