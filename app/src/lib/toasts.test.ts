import { describe, expect, it } from "vitest";
import { pushToast, dismissToast, pushLog, type Toast } from "./toasts";

describe("pushToast", () => {
  it("appends a new toast to an empty list", () => {
    const result = pushToast([], "info", "exported 3 changed pixels", 1);
    expect(result).toEqual([
      { id: 1, level: "info", message: "exported 3 changed pixels", count: 1 },
    ]);
  });

  it("collapses an identical consecutive push into the last toast, incrementing count and keeping the same id", () => {
    const first = pushToast([], "error", "Frame raw0002.jpg: boom", 1);
    const second = pushToast(first, "error", "Frame raw0002.jpg: boom", 2);
    expect(second).toEqual([
      { id: 1, level: "error", message: "Frame raw0002.jpg: boom", count: 2 },
    ]);
  });

  it("appends a new toast (does not collapse) when the message differs from the last one", () => {
    const first = pushToast([], "error", "Frame raw0001.jpg: boom", 1);
    const second = pushToast(first, "error", "Frame raw0002.jpg: bang", 2);
    expect(second).toEqual([
      { id: 1, level: "error", message: "Frame raw0001.jpg: boom", count: 1 },
      { id: 2, level: "error", message: "Frame raw0002.jpg: bang", count: 1 },
    ]);
  });

  it("appends a new toast (does not collapse) when the level differs from the last one, even with the same message", () => {
    const first = pushToast([], "info", "same text", 1);
    const second = pushToast(first, "error", "same text", 2);
    expect(second).toEqual([
      { id: 1, level: "info", message: "same text", count: 1 },
      { id: 2, level: "error", message: "same text", count: 1 },
    ]);
  });

  it("only collapses against the last toast, not an earlier matching one", () => {
    const list: Toast[] = [
      { id: 1, level: "error", message: "boom", count: 1 },
      { id: 2, level: "info", message: "unrelated", count: 1 },
    ];
    const result = pushToast(list, "error", "boom", 3);
    expect(result).toEqual([
      { id: 1, level: "error", message: "boom", count: 1 },
      { id: 2, level: "info", message: "unrelated", count: 1 },
      { id: 3, level: "error", message: "boom", count: 1 },
    ]);
  });
});

describe("dismissToast", () => {
  it("removes the toast with the matching id", () => {
    const list: Toast[] = [
      { id: 1, level: "info", message: "a", count: 1 },
      { id: 2, level: "error", message: "b", count: 1 },
    ];
    expect(dismissToast(list, 1)).toEqual([{ id: 2, level: "error", message: "b", count: 1 }]);
  });

  it("returns the list unchanged when the id is not present", () => {
    const list: Toast[] = [{ id: 1, level: "info", message: "a", count: 1 }];
    expect(dismissToast(list, 99)).toEqual(list);
  });
});

describe("pushLog", () => {
  it("appends an entry under the cap", () => {
    const result = pushLog([1, 2], 3, 5);
    expect(result).toEqual([1, 2, 3]);
  });

  it("caps at N, dropping the oldest entries", () => {
    const result = pushLog([1, 2, 3], 4, 3);
    expect(result).toEqual([2, 3, 4]);
  });

  it("caps correctly when a single push would exceed the cap by more than one", () => {
    const result = pushLog([1, 2, 3, 4, 5], 6, 3);
    expect(result).toEqual([4, 5, 6]);
  });
});
