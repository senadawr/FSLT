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
  saved_at: string;
};

export const MAX_POSE_SAMPLES = 32;

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
