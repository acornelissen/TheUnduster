import { describe, expect, it } from "vitest";
import { probPathFor, TextureStore } from "./renderer";

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

describe("probPathFor", () => {
  it("prefixes the layer and preserves the tile path", () => {
    expect(probPathFor("/3/0/1/2")).toBe("/probs/3/0/1/2");
  });
});

describe("TextureStore.clear", () => {
  it("runs onEvict for every entry and resets usage", () => {
    const store = new TextureStore<number>(10_000);
    const freed: number[] = [];
    store.onEvict = (t) => freed.push(t);
    store.put("a", 1, 1000);
    store.put("b", 2, 1000);
    store.clear();
    expect(freed.sort()).toEqual([1, 2]);
    expect(store.get("a")).toBeUndefined();
    store.put("c", 3, 1000); // usage reset, so no spurious eviction
    expect(store.get("c")).toBe(3);
  });
});
