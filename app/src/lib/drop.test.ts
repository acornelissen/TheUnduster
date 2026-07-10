import { describe, expect, it } from "vitest";
import { routeDrop } from "./drop";

describe("routeDrop", () => {
  it("routes a single dropped file to a scan open", () => {
    expect(routeDrop(["/scans/a.tif"], ["file"])).toEqual({
      action: "scan",
      path: "/scans/a.tif",
    });
  });

  it("routes a single dropped directory to a roll open", () => {
    expect(routeDrop(["/scans/roll-1"], ["dir"])).toEqual({
      action: "roll",
      path: "/scans/roll-1",
    });
  });

  it("errors when multiple paths are dropped", () => {
    expect(routeDrop(["/scans/a.tif", "/scans/b.tif"], ["file", "file"])).toEqual({
      error: "drop a single scan or one roll folder",
    });
  });

  it("errors when nothing is dropped", () => {
    expect(routeDrop([], [])).toEqual({
      error: "drop a single scan or one roll folder",
    });
  });
});
