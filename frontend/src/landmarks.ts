export const FEATURE_SCHEMA_VERSION = "fsl-landmarks-v1";
export const LANDMARK_COUNT = 553;
export const FEATURE_DIMENSION = 1659;

export type LandmarkFrame = {
  schema_version: string;
  timestamp_ms: number;
  features: number[];
  presence: number[];
};

export function emptyFrame(timestampMs: number): LandmarkFrame {
  return {
    schema_version: FEATURE_SCHEMA_VERSION,
    timestamp_ms: timestampMs,
    features: Array(FEATURE_DIMENSION).fill(0),
    presence: Array(LANDMARK_COUNT).fill(0),
  };
}
