export type CollectedHandFrame = {
  timestamp_ms: number;
  left_hand: Array<[number, number, number]> | null;
  right_hand: Array<[number, number, number]> | null;
  left_presence: number[];
  right_presence: number[];
};

export type CollectionSession = {
  schema_version: "fsl-hand-collection-v1";
  session_id: string;
  signer_id: string | null;
  sign_label: string;
  gloss_label: string;
  recorded_at: string;
  frame_interval_ms: number;
  coordinate_system: "MediaPipe normalized x/y/z";
  missing_hand_policy: "null hand with 21 zero presences";
  frames: CollectedHandFrame[];
};

export type PoseTemplate = {
  label: string;
  landmarks: Array<[number, number, number]>;
  samples?: Array<Array<[number, number, number]>>;
  motion_samples?: Array<Array<[number, number, number]>>;
  finger_motion_samples?: Array<Array<Array<[number, number, number]>>>;
  saved_at: string;
};

export const MAX_POSE_SAMPLES = 32;
export const MAX_MOTION_SAMPLES = 32;
export const MOTION_RECORDING_MS = 1500;

export function countPoseSamples(pose: PoseTemplate): number {
  const motionSamples = Math.max(
    pose.motion_samples?.length ?? 0,
    pose.finger_motion_samples?.length ?? 0,
  );
  if (motionSamples > 0) return motionSamples;
  return pose.samples?.length ?? (pose.landmarks.length === 21 ? 1 : 0);
}

export function appendMotionSample(
  templates: PoseTemplate[],
  label: string,
  motionSample: Array<[number, number, number]>,
  fallbackLandmarks: Array<[number, number, number]>,
  fingerMotionSample?: Array<Array<[number, number, number]>>,
): PoseTemplate[] {
  const existing = templates.find(
    (item) => item.label.toLowerCase() === label.toLowerCase(),
  );
  const updated: PoseTemplate = existing
    ? {
        ...existing,
        motion_samples: [...(existing.motion_samples ?? []), motionSample].slice(-MAX_MOTION_SAMPLES),
        finger_motion_samples: fingerMotionSample
          ? [...(existing.finger_motion_samples ?? []), fingerMotionSample].slice(-MAX_MOTION_SAMPLES)
          : existing.finger_motion_samples,
        samples: [],
        saved_at: new Date().toISOString(),
      }
    : {
        label,
        landmarks: fallbackLandmarks,
        samples: [],
        motion_samples: [motionSample],
        finger_motion_samples: fingerMotionSample ? [fingerMotionSample] : undefined,
        saved_at: new Date().toISOString(),
      };
  return existing
    ? templates.map((item) => item.label === existing.label ? updated : item)
    : [...templates, updated];
}

export function mirrorHand(
  hand: Array<[number, number, number]>,
): Array<[number, number, number]> {
  if (hand.length !== 21) {
    throw new Error("A pose must contain exactly 21 landmarks");
  }
  const wristX = hand[0][0];
  const wristZ = hand[0][2];
  return hand.map(([x, y, z]) =>
    [wristX * 2 - x, y, wristZ * 2 - z] as [number, number, number],
  );
}

export function normalizeHand(
  hand: Array<[number, number, number]>,
): Array<[number, number, number]> {
  if (hand.length !== 21) {
    throw new Error("A pose must contain exactly 21 landmarks");
  }
  const [wristX, wristY, wristZ] = hand[0];
  const scale = Math.hypot(
    hand[9][0] - wristX,
    hand[9][1] - wristY,
    hand[9][2] - wristZ,
  );
  const safeScale = scale > 1e-6 ? scale : 1;
  return hand.map(([x, y, z]) => [
    (x - wristX) / safeScale,
    (y - wristY) / safeScale,
    (z - wristZ) / safeScale,
  ]);
}

function dot(first: [number, number, number], second: [number, number, number]): number {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

function cross(first: [number, number, number], second: [number, number, number]): [number, number, number] {
  return [
    first[1] * second[2] - first[2] * second[1],
    first[2] * second[0] - first[0] * second[2],
    first[0] * second[1] - first[1] * second[0],
  ];
}

function unit(vector: [number, number, number]): [number, number, number] | null {
  const length = Math.hypot(...vector);
  return length > 1e-6 ? vector.map((value) => value / length) as [number, number, number] : null;
}

export function canonicalizeHand(
  hand: Array<[number, number, number]>,
): Array<[number, number, number]> {
  if (hand.length !== 21) {
    throw new Error("A pose must contain exactly 21 landmarks");
  }
  const middleDirection = unit(hand[9]);
  if (!middleDirection) return hand.map((point) => [...point] as [number, number, number]);
  const indexDirection = hand[5].map((value, index) =>
    value - middleDirection[index] * dot(hand[5], middleDirection),
  ) as [number, number, number];
  const xAxis = unit(indexDirection);
  if (!xAxis) return hand.map((point) => [...point] as [number, number, number]);
  const zAxis = unit(cross(xAxis, middleDirection));
  if (!zAxis) return hand.map((point) => [...point] as [number, number, number]);
  return hand.map((point) => [
    dot(point, xAxis),
    dot(point, middleDirection),
    dot(point, zAxis),
  ]);
}

export function poseDistance(
  first: Array<[number, number, number]>,
  second: Array<[number, number, number]>,
): number {
  if (first.length !== 21 || second.length !== 21) {
    throw new Error("Pose comparison requires exactly 21 landmarks");
  }
  const canonicalFirst = canonicalizeHand(first);
  const canonicalSecond = canonicalizeHand(second);
  const totalDistance = canonicalFirst.reduce((sum, point, index) => {
    const other = canonicalSecond[index];
    const isTip = [4, 8, 12, 16, 20].includes(index);
    const isThumbOrPip = index <= 4 || [5, 9, 13, 17].includes(index);
    const weight = isTip ? 1.25 : isThumbOrPip ? 1.1 : 0.85;
    return sum + Math.hypot(
      point[0] - other[0],
      point[1] - other[1],
      point[2] - other[2],
    ) * weight;
  }, 0);
  return totalDistance / first.length;
}

export function poseConfidence(distance: number): number {
  const confidence = 1 - distance / 0.35;
  return Math.max(0, Math.min(1, confidence));
}

export function smoothTrajectory(
  points: Array<[number, number, number]>,
  alpha = 0.35,
): Array<[number, number, number]> {
  if (points.length === 0) return [];
  const smoothed: Array<[number, number, number]> = [[...points[0]]];
  for (let index = 1; index < points.length; index += 1) {
    const previous = smoothed[index - 1];
    const current = points[index];
    smoothed.push([
      previous[0] + (current[0] - previous[0]) * alpha,
      previous[1] + (current[1] - previous[1]) * alpha,
      previous[2] + (current[2] - previous[2]) * alpha,
    ]);
  }
  return smoothed;
}

export function normalizeTrajectory(
  points: Array<[number, number, number]>,
): Array<[number, number, number]> {
  if (points.length === 0) return [];
  const [startX, startY, startZ] = points[0];
  const relative = points.map(([x, y, z]) =>
    [x - startX, y - startY, z - startZ] as [number, number, number],
  );
  let scale = 0;
  for (const [x, y, z] of relative) {
    scale = Math.max(scale, Math.abs(x), Math.abs(y), Math.abs(z));
  }
  if (scale < 1e-6) return relative;
  return relative.map(([x, y, z]) => [x / scale, y / scale, z / scale]);
}

export function isZZigZag(
  points: Array<[number, number, number]>,
  minimumTravel = 0.16,
): boolean {
  if (points.length < 12) return false;
  const normalized = resampleTrajectory(points, 32);
  const xs = normalized.map(([x]) => x);
  const ys = normalized.map(([, y]) => y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const horizontal = maxX - minX;
  if (horizontal < minimumTravel) return false;

  const firstDx = normalized[10][0] - normalized[0][0];
  const middleDx = normalized[21][0] - normalized[10][0];
  const lastDx = normalized[31][0] - normalized[21][0];
  if (Math.abs(firstDx) < horizontal * 0.2) return false;
  if (Math.abs(middleDx) < horizontal * 0.12) return false;
  if (Math.abs(lastDx) < horizontal * 0.2) return false;
  if (Math.sign(firstDx) === Math.sign(middleDx)) return false;
  if (Math.sign(middleDx) === Math.sign(lastDx)) return false;
  return true;
}

export function resampleTrajectory(
  points: Array<[number, number, number]>,
  targetCount = 24,
): Array<[number, number, number]> {
  const normalized = normalizeTrajectory(points);
  if (normalized.length === 0 || targetCount < 2) return normalized;
  if (normalized.length === 1) {
    return Array.from({ length: targetCount }, () => [...normalized[0]] as [number, number, number]);
  }

  const distances = [0];
  let total = 0;
  for (let index = 1; index < normalized.length; index += 1) {
    const previous = normalized[index - 1];
    const current = normalized[index];
    total += Math.hypot(
      current[0] - previous[0],
      current[1] - previous[1],
      current[2] - previous[2],
    );
    distances.push(total);
  }
  if (total < 1e-6) {
    return Array.from({ length: targetCount }, () => [...normalized[0]] as [number, number, number]);
  }

  return Array.from({ length: targetCount }, (_, index) => {
    const target = (index / (targetCount - 1)) * total;
    let segment = 1;
    while (segment < distances.length && distances[segment] < target) segment += 1;
    const startIndex = segment - 1;
    const endIndex = segment;
    const segmentLength = distances[endIndex] - distances[startIndex];
    const amount = segmentLength > 1e-6
      ? (target - distances[startIndex]) / segmentLength
      : 0;
    const start = normalized[startIndex];
    const end = normalized[endIndex];
    return [
      start[0] + (end[0] - start[0]) * amount,
      start[1] + (end[1] - start[1]) * amount,
      start[2] + (end[2] - start[2]) * amount,
    ] as [number, number, number];
  });
}

export function trajectoryDistance(
  first: Array<[number, number, number]>,
  second: Array<[number, number, number]>,
): number {
  const firstResampled = resampleTrajectory(first);
  const secondResampled = resampleTrajectory(second);
  if (firstResampled.length === 0 || secondResampled.length === 0) return 1;
  const total = firstResampled.reduce((sum, point, index) => {
    const other = secondResampled[index];
    return sum + Math.hypot(
      point[0] - other[0],
      point[1] - other[1],
      point[2] - other[2],
    );
  }, 0);
  return total / firstResampled.length;
}
