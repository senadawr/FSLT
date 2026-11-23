import { describe, expect, it } from "vitest";
import {
  MAX_POSE_SAMPLES,
  MAX_MOTION_SAMPLES,
  MOTION_RECORDING_MS,
  countPoseSamples,
  smoothTrajectory,
  isZZigZag,
  appendMotionSample,
  mirrorHand,
  normalizeHand,
  poseConfidence,
  poseDistance,
  normalizeTrajectory,
  resampleTrajectory,
  trajectoryDistance,
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

  it("counts each motion recording as one pose sample", () => {
    const pose = {
      label: "J",
      landmarks: [],
      motion_samples: [
        [[0, 0, 0]] as Array<[number, number, number]>,
        [[0.1, 0.1, 0]] as Array<[number, number, number]>,
      ],
      saved_at: "2026-11-23T00:00:00.000Z",
    };
    expect(countPoseSamples(pose)).toBe(2);
  });

  it("supports repeated 1.5-second motion recordings up to 32 samples", () => {
    expect(MOTION_RECORDING_MS).toBe(1500);
    expect(MAX_MOTION_SAMPLES).toBe(32);
  });

  it("appends repeated recordings to the current template state", () => {
    const initial: never[] = [];
    const first = appendMotionSample(initial, "J", [[0, 0, 0]], []);
    const second = appendMotionSample(first, "J", [[0.1, 0.1, 0]], []);
    expect(countPoseSamples(second[0])).toBe(2);
  });

  it("allows Z to accumulate the same 32 motion samples as J", () => {
    let templates: Parameters<typeof appendMotionSample>[0] = [];
    for (let index = 0; index < 32; index += 1) {
      templates = appendMotionSample(
        templates,
        "Z",
        [[index / 32, index / 64, 0]],
        [],
        [[[index / 32, index / 64, 0]]],
      );
    }
    expect(countPoseSamples(templates[0])).toBe(32);
  });

  it("smooths noisy motion samples with exponential filtering", () => {
    const noisy = [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
    ] as Array<[number, number, number]>;
    const smoothed = smoothTrajectory(noisy, 0.5);
    expect(smoothed).toHaveLength(noisy.length);
    expect(smoothed[1][0]).toBeCloseTo(0.5);
    expect(smoothed[2][1]).toBeCloseTo(0.5);
  });

  it("recognizes a Z-shaped horizontal zig-zag without exact template matching", () => {
    const zPath = [
      [0, 0, 0], [0.05, 0, 0], [0.1, 0, 0], [0.15, 0, 0],
      [0.2, 0, 0], [0.25, 0.05, 0], [0.2, 0.1, 0],
      [0.15, 0.15, 0], [0.1, 0.2, 0], [0.05, 0.25, 0],
      [0, 0.3, 0], [0.05, 0.3, 0], [0.1, 0.3, 0],
      [0.15, 0.3, 0], [0.2, 0.3, 0], [0.25, 0.3, 0],
    ] as Array<[number, number, number]>;
    expect(isZZigZag(zPath)).toBe(true);
    expect(isZZigZag([
      [0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0],
    ])).toBe(false);
  });

  it("normalizes motion trails before comparing them", () => {
    const trail = [
      [0.1, 0.2, 0],
      [0.2, 0.1, 0],
      [0.3, 0.2, 0],
    ] as Array<[number, number, number]>;
    const offset = trail.map(([x, y, z]) => [x + 0.4, y - 0.3, z] as [number, number, number]);

    const normalized = normalizeTrajectory(trail);
    expect(normalized[0]).toEqual([0, 0, 0]);
    expect(normalized[1][0]).toBeCloseTo(0.5);
    expect(normalized[1][1]).toBeCloseTo(-0.5);
    expect(normalized[2]).toEqual([1, 0, 0]);
    expect(resampleTrajectory(trail)).toHaveLength(24);
    expect(trajectoryDistance(trail, offset)).toBeLessThan(0.05);
  });
});
