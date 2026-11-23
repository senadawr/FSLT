import { FilesetResolver, HolisticLandmarker, HandLandmarker } from "@mediapipe/tasks-vision";
import { FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION, LANDMARK_COUNT, LandmarkFrame } from "./landmarks";

type Point = { x: number; y: number; z: number };
type HolisticResult = {
  faceLandmarks?: Point[][];
  poseLandmarks?: Point[][];
  leftHandLandmarks?: Point[][];
  rightHandLandmarks?: Point[][];
};

const MODEL_ASSET_URL = import.meta.env.VITE_HOLISTIC_MODEL_URL
  ?? "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task";
const WASM_URL = "/mediapipe/wasm";
const HAND_MODEL_ASSET_URL =
  import.meta.env.VITE_HAND_MODEL_URL
  ?? "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export async function createHandExtractor(): Promise<HandLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL_ASSET_URL, delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.5,
  });
}

export async function createHolisticExtractor(): Promise<HolisticLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  return HolisticLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_ASSET_URL, delegate: "CPU" },
    runningMode: "VIDEO",
    outputFaceBlendshapes: false,
    minFaceDetectionConfidence: 0.5,
    minPoseDetectionConfidence: 0.5,
    minHandLandmarksConfidence: 0.5,
  });
}

function appendLandmarks(
  target: number[],
  presence: number[],
  groups: Point[][] | undefined,
  expectedCount: number,
): void {
  const landmarks = groups?.[0] ?? [];
  for (let index = 0; index < expectedCount; index += 1) {
    const point = landmarks[index];
    if (point) {
      target.push(point.x, point.y, point.z);
      presence.push(1);
    } else {
      target.push(0, 0, 0);
      presence.push(0);
    }
  }
}

export function extractFrame(
  landmarker: HolisticLandmarker,
  video: HTMLVideoElement,
  timestampMs: number,
): LandmarkFrame {
  const result = landmarker.detectForVideo(video, timestampMs) as HolisticResult;
  const features: number[] = [];
  const presence: number[] = [];
  appendLandmarks(features, presence, result.faceLandmarks, 478);
  appendLandmarks(features, presence, result.poseLandmarks, 33);
  appendLandmarks(features, presence, result.leftHandLandmarks, 21);
  appendLandmarks(features, presence, result.rightHandLandmarks, 21);
  if (features.length !== FEATURE_DIMENSION || presence.length !== LANDMARK_COUNT) {
    throw new Error("MediaPipe returned an incompatible landmark shape");
  }
  return { schema_version: FEATURE_SCHEMA_VERSION, timestamp_ms: timestampMs, features, presence };
}

export type HandFrame = {
  timestamp_ms: number;
  left_hand: Array<[number, number, number]> | null;
  right_hand: Array<[number, number, number]> | null;
  left_presence: number[];
  right_presence: number[];
};

function extractHand(groups: Point[][] | undefined): Array<[number, number, number]> | null {
  const landmarks = groups?.[0];
  if (!landmarks || landmarks.length !== 21) return null;
  return landmarks.map((point) => [point.x, point.y, point.z]);
}

export function extractHands(
  landmarker: HandLandmarker,
  video: HTMLVideoElement,
  timestampMs: number,
): HandFrame {
  const result = landmarker.detectForVideo(video, timestampMs);
  let left: Array<[number, number, number]> | null = null;
  let right: Array<[number, number, number]> | null = null;
  result.landmarks.forEach((landmarks, index) => {
    const handedness = result.handedness[index]?.[0]?.categoryName;
    const normalized = landmarks.map((point) => [point.x, point.y, point.z] as [number, number, number]);
    if (handedness?.toLowerCase() === "left") left = normalized;
    else if (handedness?.toLowerCase() === "right") right = normalized;
  });
  return {
    timestamp_ms: timestampMs,
    left_hand: left,
    right_hand: right,
    left_presence: left ? Array(21).fill(1) : Array(21).fill(0),
    right_presence: right ? Array(21).fill(1) : Array(21).fill(0),
  };
}
