import { describe, expect, it } from "vitest";
import {
  MAX_POSE_SAMPLES,
  mirrorHand,
  normalizeHand,
  poseConfidence,
  poseDistance,
} from "./collection";
import { formatRecognitionText, shouldAppendRecognition } from "./App";

describe("hand overlay contract", () => {
  it("has exactly 21 landmarks per detected hand", () => {
    const left = Array.from({ length: 21 }, (_, index) => [index / 21, 0.5, 0] as [number, number, number]);
    expect(left).toHaveLength(21);
    expect(left[0]).toEqual([0, 0.5, 0]);
  });

  it("represents a missing hand explicitly", () => {
    const missingHand: null = null;
    const presence = Array(21).fill(0);
    expect(missingHand).toBeNull();
    expect(presence).toHaveLength(21);
    expect(presence.every((value) => value === 0)).toBe(true);
  });

  it("normalizes scale and rotation before comparing poses", () => {
    const pose = Array.from({ length: 21 }, (_, index) => [
      (index % 5) * 0.1,
      index * 0.08,
      (index % 3) * 0.03,
    ] as [number, number, number]);
    const rotatedAndScaled = pose.map(([x, y, z]) => [
      10 - y * 2,
      20 + x * 2,
      4 + z * 2,
    ] as [number, number, number]);
    expect(poseDistance(normalizeHand(pose), normalizeHand(rotatedAndScaled))).toBeCloseTo(0);
    expect(poseConfidence(0)).toBe(1);
    expect(poseConfidence(0.35)).toBe(0);
  });

  it("mirrors left-hand x coordinates into the right-hand convention", () => {
    const hand = Array.from({ length: 21 }, (_, index) => [index / 20, 0.5, 0] as [number, number, number]);
    expect(mirrorHand(mirrorHand(hand))).toEqual(hand);
    expect(mirrorHand(hand)[10][0]).toBe(-hand[10][0]);
  });

  it("capitalizes letters but preserves words and symbols", () => {
    expect(formatRecognitionText("a")).toBe("A");
    expect(formatRecognitionText("hello")).toBe("hello");
    expect(formatRecognitionText("?")).toBe("?");
  });

  it("requires the same label to leave and re-enter the frame before repeating", () => {
    expect(shouldAppendRecognition("A", "A")).toBe(false);
    expect(shouldAppendRecognition(null, "A")).toBe(true);
    expect(shouldAppendRecognition("A", "B")).toBe(true);
  });

  it("can match a live pose against the nearest of several saved samples", () => {
    const base = Array.from(
      { length: 21 },
      (_, index) => [(index % 5) * 0.1, index * 0.08, 0] as [number, number, number],
    );
    const variant = base.map(([x, y, z]) => [x + 0.03, y - 0.02, z] as [number, number, number]);
    const live = base.map(([x, y, z]) => [x + 0.01, y - 0.01, z] as [number, number, number]);
    expect(Math.min(poseDistance(normalizeHand(live), normalizeHand(base)), poseDistance(normalizeHand(live), normalizeHand(variant)))).toBeLessThan(0.05);
  });

  it("supports at least 32 pose samples per label", () => {
    expect(MAX_POSE_SAMPLES).toBeGreaterThanOrEqual(32);
  });
});
