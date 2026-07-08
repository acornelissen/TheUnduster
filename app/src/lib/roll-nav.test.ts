import { describe, expect, it } from "vitest";
import { nextUnapprovedIndex } from "./roll-nav";

describe("nextUnapprovedIndex", () => {
  it("advances to the next unapproved after the current frame", () => {
    expect(nextUnapprovedIndex([true, false, false], 0)).toBe(1);
  });

  it("wraps past the end to earlier unapproved frames", () => {
    expect(nextUnapprovedIndex([false, true, true], 2)).toBe(0);
  });

  it("returns -1 when everything is approved", () => {
    expect(nextUnapprovedIndex([true, true, true], 1)).toBe(-1);
  });

  it("never returns the current frame even when it is unapproved", () => {
    expect(nextUnapprovedIndex([true, false, true], 1)).toBe(-1);
  });

  it("handles a single-frame roll", () => {
    expect(nextUnapprovedIndex([false], 0)).toBe(-1);
    expect(nextUnapprovedIndex([true], 0)).toBe(-1);
  });
});
