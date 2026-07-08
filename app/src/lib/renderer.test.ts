import { describe, expect, it } from "vitest";
import { TextureStore } from "./renderer";

describe("TextureStore", () => {
  it("evicts least-recently-used past the budget", () => {
    const store = new TextureStore<number>(2500);
    const dropped: number[] = [];
    store.onEvict = (t) => dropped.push(t);
    store.put("a", 1, 1000);
    store.put("b", 2, 1000);
    store.get("a");
    store.put("c", 3, 1000);
    expect(store.get("b")).toBeUndefined();
    expect(store.get("a")).toBe(1);
    expect(dropped).toEqual([2]);
  });
});
