import { useEffect, useRef, useState } from "react";
import { HandLandmarker } from "@mediapipe/tasks-vision";
import { createHandExtractor, extractHands } from "./holistic";
import {
  MAX_POSE_SAMPLES,
  MAX_MOTION_SAMPLES,
  MOTION_RECORDING_MS,
  PoseTemplate,
  appendMotionSample,
  countPoseSamples,
  mirrorHand,
  normalizeHand,
  poseConfidence,
  poseDistance,
  resampleTrajectory,
  smoothTrajectory,
  trajectoryDistance,
  isZZigZag,
} from "./collection";
import {
  calculateEvaluationResults,
  createEvaluationRun,
  currentEvaluationTarget,
  evaluationResultsCsv,
  isEvaluationLetter,
  recordEvaluationTrial,
  type EvaluationRun,
} from "./evaluation";

export function formatRecognitionText(label: string): string {
  if (/^[A-Za-z]$/.test(label)) return label.toUpperCase();
  return label;
}

export function shouldAppendRecognition(
  lastLabel: string | null,
  nextLabel: string,
): boolean {
  return lastLabel !== nextLabel;
}

export function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const frameIdRef = useRef<number | null>(null);
  const [cameraState, setCameraState] = useState("Camera stopped");
  const [translation, setTranslation] = useState("No translation");
  const [confidence, setConfidence] = useState(0);
  const [predictionStatus, setPredictionStatus] = useState("Model idle");
  const [history, setHistory] = useState<string[]>([]);
  const [typedHistory, setTypedHistory] = useState("");
  const typedHistoryRef = useRef("");
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [cameraStarting, setCameraStarting] = useState(false);
  const [trackedHands, setTrackedHands] = useState({ left: false, right: false });
  const [trackingFps, setTrackingFps] = useState(0);
  const [poseLabel, setPoseLabel] = useState("");
  const [poseTemplates, setPoseTemplates] = useState<PoseTemplate[]>([]);
  const [poseLoadError, setPoseLoadError] = useState("");
  const poseCandidateRef = useRef<{ label: string; since: number; frames: number } | null>(null);
  const lastTypedRef = useRef<{ label: string; at: number } | null>(null);
  const staticRepeatBlockRef = useRef<{ label: string; until: number } | null>(null);
  const currentHandRef = useRef<Array<[number, number, number]> | null>(null);
  const showLandmarksRef = useRef(true);
  const trackingActiveRef = useRef(false);
  const lastVideoTimeRef = useRef(-1);
  const videoCallbackIdRef = useRef<number | null>(null);
  const usingVideoFrameCallbackRef = useRef(false);
  const fpsWindowRef = useRef({ count: 0, startedAt: performance.now(), shown: 0 });
  const heldHandsRef = useRef({
    left: null as Array<[number, number, number]> | null,
    right: null as Array<[number, number, number]> | null,
  });
  const heldHandTimesRef = useRef({ left: 0, right: 0 });
  const motionRecordingRef = useRef<{
    active: boolean;
    label: string;
    since: number;
    points: Array<[number, number, number]>;
    fingertipPoints: Array<Array<[number, number, number]>>;
  }>({ active: false, label: "", since: 0, points: [], fingertipPoints: [] });
  const liveMotionRef = useRef<Array<[number, number, number]>>([]);
  const liveFingerMotionRef = useRef<Array<Array<[number, number, number]>>>([]);
  const visualMotionRef = useRef<Array<[number, number, number, number]>>([]);
  const visualFingerMotionRef = useRef<Array<Array<[number, number, number, number]>>>([]);
  const activeMotionFingerIndexRef = useRef<number | null>(null);
  const liveMotionTimeRef = useRef(0);
  const liveMotionUpdatedAtRef = useRef(0);
  const [motionRecording, setMotionRecording] = useState(false);
  const [motionSeconds, setMotionSeconds] = useState(0);
  const [evaluationRun, setEvaluationRun] = useState<EvaluationRun | null>(null);
  const evaluationRunRef = useRef<EvaluationRun | null>(null);
  const [evaluationClassCount, setEvaluationClassCount] = useState<10 | 20>(10);
  const [evaluationTrialsPerLetter, setEvaluationTrialsPerLetter] = useState(3);
  const compactTranslation = ["No close match", "Hold steady", "No translation"].includes(translation)
    ? "—"
    : translation;
  const evaluationTarget = evaluationRun ? currentEvaluationTarget(evaluationRun) : null;
  const evaluationResults = evaluationRun ? calculateEvaluationResults(evaluationRun) : null;

  useEffect(() => {
    evaluationRunRef.current = evaluationRun;
    if (evaluationRun) {
      window.localStorage.setItem("fslt-evaluation-run-v1", JSON.stringify(evaluationRun));
    }
  }, [evaluationRun]);

  useEffect(() => {
    try {
      const savedRun = window.localStorage.getItem("fslt-evaluation-run-v1");
      if (!savedRun) return;
      const parsed = JSON.parse(savedRun) as EvaluationRun;
      if (
        Array.isArray(parsed.selectedLetters) &&
        Array.isArray(parsed.targetOrder) &&
        Array.isArray(parsed.trials) &&
        typeof parsed.runId === "string" &&
        (parsed.selectedLetters.length === 10 || parsed.selectedLetters.length === 20) &&
        parsed.selectedLetters.every(isEvaluationLetter) &&
        new Set(parsed.selectedLetters).size === parsed.selectedLetters.length &&
        parsed.targetOrder.every(isEvaluationLetter) &&
        parsed.targetOrder.length === parsed.selectedLetters.length * parsed.trialsPerLetter &&
        parsed.trials.every((trial) => isEvaluationLetter(trial.expectedLetter))
      ) {
        evaluationRunRef.current = parsed;
        setEvaluationRun(parsed);
      }
    } catch {
      window.localStorage.removeItem("fslt-evaluation-run-v1");
    }
  }, []);

  useEffect(() => {
    showLandmarksRef.current = showLandmarks;
  }, [showLandmarks]);

  useEffect(() => {
    const target = history.map(formatRecognitionText).join("");
    let frame: number | null = null;
    let cancelled = false;
    let nextLength = target.startsWith(typedHistoryRef.current)
      ? typedHistoryRef.current.length
      : 0;

    if (!target.startsWith(typedHistoryRef.current)) {
      if (typedHistoryRef.current.length > 0 && target.length > typedHistoryRef.current.length) {
        nextLength = typedHistoryRef.current.length;
      } else {
        typedHistoryRef.current = "";
        setTypedHistory("");
      }
    }
    frame = window.setInterval(() => {
      if (cancelled) return;
      if (nextLength >= target.length) {
        if (frame !== null) window.clearInterval(frame);
        return;
      }
      nextLength += 1;
      const nextText = target.slice(0, nextLength);
      typedHistoryRef.current = nextText;
      setTypedHistory(nextText);
      if (nextLength >= target.length && frame !== null) {
        window.clearInterval(frame);
      }
    }, 55);
    return () => {
      cancelled = true;
      if (frame !== null) window.clearInterval(frame);
    };
  }, [history]);

  useEffect(() => () => {
    if (frameIdRef.current !== null) window.cancelAnimationFrame(frameIdRef.current);
    const videoWithCallbacks = videoRef.current as (HTMLVideoElement & {
      cancelVideoFrameCallback?: (id: number) => void;
    }) | null;
    if (videoCallbackIdRef.current !== null) {
      videoWithCallbacks?.cancelVideoFrameCallback?.(videoCallbackIdRef.current);
    }
    landmarkerRef.current?.close();
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((track) => track.stop());
  }, []);

  useEffect(() => {
    let active = true;
    async function loadPoseTemplates() {
      try {
        const response = await fetch("/api/poses");
        if (!response.ok) throw new Error(`Pose load failed (${response.status})`);
        const payload = await response.json() as { templates: PoseTemplate[] };
        if (!active) return;
        setPoseTemplates(payload.templates);
      } catch {
        if (active) {
          setPoseTemplates([]);
          setPoseLoadError("Backend unavailable");
          setPredictionStatus("Unable to load saved poses");
        }
      }
    }
    void loadPoseTemplates();
    return () => {
      active = false;
    };
  }, []);

  async function startCamera() {
    if (trackingActiveRef.current) return;
    setCameraStarting(true);
    let stage = "camera permission";
    try {
      if (!window.isSecureContext) {
        throw new Error("Camera blocked. Open this app through localhost or HTTPS.");
      }
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera API unavailable. Use localhost or HTTPS in a current browser.");
      }
      let stream: MediaStream;
      const firefox = navigator.userAgent.includes("Firefox");
      try {
        stream = firefox
          ? await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
          : await navigator.mediaDevices.getUserMedia({
              video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 60, max: 60 },
                facingMode: "user",
              },
              audio: false,
            });
      } catch (error) {
        if (error instanceof DOMException && error.name === "OverconstrainedError") {
          stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
        } else {
          throw error;
        }
      }
      if (!videoRef.current) return;
      stage = "starting video";
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      lastVideoTimeRef.current = -1;
      stage = "loading hand model";
      landmarkerRef.current = await createHandExtractor();
      trackingActiveRef.current = true;
      setCameraState("Camera running");
      heldHandsRef.current = { left: null, right: null };
      heldHandTimesRef.current = { left: 0, right: 0 };
      fpsWindowRef.current = { count: 0, startedAt: performance.now(), shown: 0 };
      setPredictionStatus(poseTemplates.length ? "Ready to match poses" : "No poses saved");
      startTrackingFrames();
    } catch (error) {
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((track) => track.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      trackingActiveRef.current = false;
      if (error instanceof DOMException) {
        if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
          setCameraState("Camera permission denied. Allow camera access in browser settings.");
        } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
          setCameraState("No camera found. Connect or enable a webcam.");
        } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
          setCameraState("Camera is busy or unavailable in another app.");
        } else if (error.name === "SecurityError") {
          setCameraState("Camera blocked. Open the app through localhost or HTTPS.");
        } else {
          setCameraState(`Unable to initialize camera: ${error.name}`);
        }
      } else if (error instanceof Error) {
        setCameraState(stage === "loading hand model"
          ? `MediaPipe initialization failed: ${error.message}`
          : error.message);
      } else {
        setCameraState("Unable to initialize camera");
      }
    } finally {
      setCameraStarting(false);
    }
  }

  function trackHands() {
    if (!trackingActiveRef.current) return;
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !landmarker) return;
    try {
      if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        const handFrame = extractHands(landmarker, video, performance.now());
        drawHands(handFrame);
        updateTrackingFps();
        const hand = handFrame.left_hand
          ? mirrorHand(handFrame.left_hand)
          : handFrame.right_hand;
        currentHandRef.current = hand;
        updateMotionRecording(hand);
        matchPose(hand);
        const nextTrackedHands = {
          left: handFrame.left_hand !== null,
          right: handFrame.right_hand !== null,
        };
        setTrackedHands((previous) =>
          previous.left === nextTrackedHands.left && previous.right === nextTrackedHands.right
            ? previous
            : nextTrackedHands,
        );
      }

    } catch (error) {
      console.error("Landmark extraction or matching error", error);
      setPredictionStatus("Frame skipped");
      setCameraState("Frame skipped — tracking continues");
    }
    if (!usingVideoFrameCallbackRef.current) {
      frameIdRef.current = window.requestAnimationFrame(trackHands);
    }
  }

  function getWristPoint(
    hand: Array<[number, number, number]> | null,
  ): [number, number, number] | null {
    return hand?.[0] ?? null;
  }

  const FINGER_TIP_INDICES = [4, 8, 12, 16, 20];

  function getMotionFingerIndex(label: string): number | null {
    const normalized = label.trim().toUpperCase();
    if (normalized === "J") return 4;
    if (normalized === "Z") return 1;
    return null;
  }

  function getFingerTips(
    hand: Array<[number, number, number]> | null,
  ): Array<[number, number, number]> {
    return hand
      ? FINGER_TIP_INDICES.map((index) => hand[index]).filter(
          (point): point is [number, number, number] => point !== undefined,
        )
      : [];
  }

  function updateMotionRecording(
    hand: Array<[number, number, number]> | null,
  ) {
    const now = performance.now();
    const wrist = getWristPoint(hand);
    if (!wrist || now - liveMotionUpdatedAtRef.current > 500) {
      liveMotionRef.current = [];
      liveFingerMotionRef.current = [];
      visualMotionRef.current = [];
      visualFingerMotionRef.current = [];
      poseCandidateRef.current = null;
    }
    if (wrist) {
      if (now - liveMotionTimeRef.current >= 50) {
        const previousWrist = liveMotionRef.current.at(-1);
        const smoothedWrist: [number, number, number] = previousWrist
          ? [
              previousWrist[0] + (wrist[0] - previousWrist[0]) * 0.45,
              previousWrist[1] + (wrist[1] - previousWrist[1]) * 0.45,
              previousWrist[2] + (wrist[2] - previousWrist[2]) * 0.45,
            ]
          : wrist;
        liveMotionRef.current = [...liveMotionRef.current, smoothedWrist].slice(-36);
        visualMotionRef.current = [
          ...visualMotionRef.current,
          [smoothedWrist[0], smoothedWrist[1], smoothedWrist[2], now] as [number, number, number, number],
        ].slice(-10);
        const fingerTips = getFingerTips(hand);
        if (fingerTips.length === 5) {
          liveFingerMotionRef.current = liveFingerMotionRef.current.length === 5
            ? liveFingerMotionRef.current.map((trail, fingerIndex) => {
                const previous = trail.at(-1);
                const point = fingerTips[fingerIndex];
                const smoothed: [number, number, number] = previous
                  ? [
                      previous[0] + (point[0] - previous[0]) * 0.45,
                      previous[1] + (point[1] - previous[1]) * 0.45,
                      previous[2] + (point[2] - previous[2]) * 0.45,
                    ]
                  : point;
                return [...trail, smoothed].slice(-36);
              })
            : fingerTips.map((tip) => [tip]);

          const activeLabel = motionRecordingRef.current.active
            ? motionRecordingRef.current.label
            : poseLabel.trim();
          const selectedFingerIndex = getMotionFingerIndex(activeLabel);
          const fingerIndices = selectedFingerIndex === null
            ? [0, 1, 2, 3, 4]
            : [selectedFingerIndex];
          visualFingerMotionRef.current = visualFingerMotionRef.current.length === fingerIndices.length
            ? visualFingerMotionRef.current.map((trail, sampleIndex) => {
                const fingerIndex = fingerIndices[sampleIndex];
                const previous = trail.at(-1);
                const point = fingerTips[fingerIndex];
                const smoothed: [number, number, number, number] = previous
                  ? [
                      previous[0] + (point[0] - previous[0]) * 0.45,
                      previous[1] + (point[1] - previous[1]) * 0.45,
                      previous[2] + (point[2] - previous[2]) * 0.45,
                      now,
                    ]
                  : [point[0], point[1], point[2], now];
                return [...trail, smoothed].slice(-10);
              })
            : fingerIndices.map((fingerIndex) =>
                [[fingerTips[fingerIndex][0], fingerTips[fingerIndex][1], fingerTips[fingerIndex][2], now] as [number, number, number, number]],
              );
        }
        liveMotionTimeRef.current = now;
        liveMotionUpdatedAtRef.current = now;
      }
    }

    const recording = motionRecordingRef.current;
    if (!recording.active) return;
    if (wrist) {
      recording.points.push(wrist);
      const fingerTips = getFingerTips(hand);
      if (fingerTips.length === 5) recording.fingertipPoints.push(fingerTips);
    }
    const elapsed = now - recording.since;
    setMotionSeconds(Math.min(elapsed / 1000, MOTION_RECORDING_MS / 1000));
    if (elapsed >= MOTION_RECORDING_MS) {
      finishMotionRecording();
    }
  }

  function startMotionRecording() {
    const label = poseLabel.trim();
    if (!label) {
      setPredictionStatus("Enter a label first");
      return;
    }
    if (!trackingActiveRef.current || !currentHandRef.current) {
      setPredictionStatus("Show a hand first");
      return;
    }
    motionRecordingRef.current = {
      active: true,
      label,
      since: performance.now(),
      points: [],
      fingertipPoints: [],
    };
    liveMotionRef.current = [];
    liveFingerMotionRef.current = [];
    visualMotionRef.current = [];
    visualFingerMotionRef.current = [];
    activeMotionFingerIndexRef.current = getMotionFingerIndex(label);
    setMotionRecording(true);
    setMotionSeconds(0);
    setPredictionStatus(`Recording motion: ${label}`);
  }

  function finishMotionRecording() {
    const recording = motionRecordingRef.current;
    if (!recording.active) return;
    recording.active = false;
    setMotionRecording(false);
    setMotionSeconds(0);

    const motionSample = resampleTrajectory(smoothTrajectory(recording.points, 0.35));
    const selectedFingerIndex = getMotionFingerIndex(recording.label);
    const fingerMotionSample = recording.fingertipPoints.length >= 8
      ? (selectedFingerIndex === null
          ? FINGER_TIP_INDICES.map((_, fingerIndex) =>
              resampleTrajectory(
                smoothTrajectory(
                  recording.fingertipPoints.map((tips) => tips[fingerIndex]),
                  0.35,
                ),
              ),
            )
          : [resampleTrajectory(
              smoothTrajectory(
                recording.fingertipPoints.map((tips) => tips[selectedFingerIndex]),
                0.35,
              ),
            )])
      : undefined;
    if (motionSample.length < 4 || recording.points.length < 8) {
      setPredictionStatus("Motion too short. Hold the hand steady and try again.");
      return;
    }

    setPoseTemplates((currentTemplates) => {
      const existing = currentTemplates.find(
        (item) => item.label.toLowerCase() === recording.label.toLowerCase(),
      );
      const fallbackLandmarks = currentHandRef.current
        ? normalizeHand(currentHandRef.current)
        : [];
      const nextTemplates = appendMotionSample(
        currentTemplates,
        recording.label,
        motionSample,
        fallbackLandmarks,
        fingerMotionSample,
      );
      void persistPoseTemplates(nextTemplates);

      const savedCount = Math.min(
        (existing?.motion_samples?.length ?? 0) + 1,
        MAX_MOTION_SAMPLES,
      );
      setPredictionStatus(
        `Saved motion: ${recording.label} (${savedCount}/${MAX_MOTION_SAMPLES} samples)`,
      );
      return nextTemplates;
    });
  }

  function matchPose(hand: Array<[number, number, number]> | null) {
    if (!hand || poseTemplates.length === 0) {
      setConfidence(0);
      setPredictionStatus(poseTemplates.length ? "Show a hand" : "No poses saved");
      poseCandidateRef.current = null;
      if (!hand) lastTypedRef.current = null;
      return;
    }
    const now = performance.now();
    const jTemplate = poseTemplates.find(
      (template) =>
        template.label.trim().toUpperCase() === "J" &&
        (template.motion_samples?.length ?? 0) > 0,
    );
    if (
      jTemplate &&
      !motionRecordingRef.current.active &&
      liveMotionRef.current.length >= 12
    ) {
      const wristStart = liveMotionRef.current[0];
      const wristEnd = liveMotionRef.current.at(-1);
      const pinkyStart = liveFingerMotionRef.current[4]?.[0];
      const pinkyEnd = liveFingerMotionRef.current[4]?.at(-1);
      const wristTravel = wristStart && wristEnd
        ? Math.hypot(wristEnd[0] - wristStart[0], wristEnd[1] - wristStart[1])
        : 0;
      const pinkyTravel = pinkyStart && pinkyEnd
        ? Math.hypot(pinkyEnd[0] - pinkyStart[0], pinkyEnd[1] - pinkyStart[1])
        : 0;
      if (wristTravel >= 0.12 && pinkyTravel >= 0.12) {
        const wristDistance = Math.min(
          ...jTemplate.motion_samples!.map((sample) =>
            trajectoryDistance(liveMotionRef.current, sample),
          ),
        );
        if (wristDistance <= 0.48) {
          if (poseCandidateRef.current?.label !== "J") {
            poseCandidateRef.current = { label: "J", since: now, frames: 1 };
          } else {
            poseCandidateRef.current.frames += 1;
          }
          if (
            poseCandidateRef.current.frames >= 5 &&
            shouldAppendRecognition(lastTypedRef.current?.label ?? null, "J")
          ) {
            setTranslation("J");
            setConfidence(Math.max(0.68, 1 - wristDistance));
            setPredictionStatus("J motion matched");
            lastTypedRef.current = { label: "J", at: now };
            staticRepeatBlockRef.current = { label: "I", until: now + 900 };
            setHistory((items) => [...items, "J"]);
            recordEvaluationPrediction("J", Math.max(0.68, 1 - wristDistance));
            liveMotionRef.current = [];
            liveFingerMotionRef.current = [];
            poseCandidateRef.current = null;
          }
          return;
        }
      }
    }
    if (
      !motionRecordingRef.current.active &&
      liveMotionRef.current.length >= 8 &&
      isZZigZag(liveMotionRef.current)
    ) {
      const zNow = performance.now();
      if (poseCandidateRef.current?.label !== "Z") {
        poseCandidateRef.current = { label: "Z", since: zNow, frames: 1 };
      } else {
        poseCandidateRef.current.frames += 1;
      }
      if (
        poseCandidateRef.current.frames >= 6 &&
        shouldAppendRecognition(lastTypedRef.current?.label ?? null, "Z")
      ) {
        setTranslation("Z");
        setConfidence(0.85);
        setPredictionStatus("Z zig-zag matched");
        lastTypedRef.current = { label: "Z", at: zNow };
        staticRepeatBlockRef.current = { label: "S", until: zNow + 900 };
        setHistory((items) => [...items, "Z"]);
        recordEvaluationPrediction("Z", 0.85);
        liveMotionRef.current = [];
        liveFingerMotionRef.current = [];
        poseCandidateRef.current = null;
      }
      return;
    }
    const normalized = normalizeHand(hand);
    const matches = poseTemplates
      .filter((template) => (template.motion_samples?.length ?? 0) === 0)
      .map((template) => {
      const samples = template.samples?.length
        ? template.samples
        : [template.landmarks];
      const distance = Math.min(
        ...samples.map((sample) => poseDistance(normalized, sample)),
      );
      return { template, distance };
    }).sort((first, second) => first.distance - second.distance);
    if (matches.length === 0) return;

    const closest = matches[0];
    const runnerUp = matches[1];
    const nextConfidence = poseConfidence(closest.distance);
    const margin = runnerUp
      ? runnerUp.distance - closest.distance
      : 1;
    const isAmbiguous = runnerUp !== undefined && margin < 0.018;
    const recognized = nextConfidence >= 0.68 && !isAmbiguous;
    const repeatBlock = staticRepeatBlockRef.current;
    if (repeatBlock && performance.now() >= repeatBlock.until) {
      staticRepeatBlockRef.current = null;
    }
    const blockedRepeat = staticRepeatBlockRef.current;
    if (
      blockedRepeat &&
      closest.template.label.trim().toUpperCase() === blockedRepeat.label &&
      performance.now() < blockedRepeat.until
    ) {
      setConfidence(0);
      setPredictionStatus("Motion finish — avoiding duplicate static letter");
      poseCandidateRef.current = null;
      return;
    }

    setConfidence(recognized ? nextConfidence : 0);
    setPredictionStatus(
      isAmbiguous
        ? `Too close: ${closest.template.label} / ${runnerUp.template.label}`
        : recognized ? "matched" : "uncertain",
    );
    if (!recognized) {
      setTranslation(isAmbiguous ? "Hold steady" : "No close match");
      poseCandidateRef.current = null;
      return;
    }

    if (poseCandidateRef.current?.label !== closest.template.label) {
      poseCandidateRef.current = { label: closest.template.label, since: now, frames: 1 };
    } else {
      poseCandidateRef.current.frames += 1;
    }
    const candidate = poseCandidateRef.current;
    if (candidate.frames < 2 || now - candidate.since < 90) return;

    const lastTyped = lastTypedRef.current;
    if (
      lastTyped?.label.trim().toUpperCase() === "J" &&
      closest.template.label.trim().toUpperCase() === "I" &&
      now - lastTyped.at < 900
    ) return;
    if (
      lastTyped?.label.trim().toUpperCase() === "Z" &&
      ["S", "X"].includes(closest.template.label.trim().toUpperCase()) &&
      now - lastTyped.at < 900
    ) return;
    if (!shouldAppendRecognition(lastTyped?.label ?? null, closest.template.label)) return;

    setTranslation(closest.template.label);
    lastTypedRef.current = { label: closest.template.label, at: now };
    setHistory((items) => [...items, closest.template.label]);
    recordEvaluationPrediction(closest.template.label, nextConfidence);
  }

  function saveCurrentPose() {
    const label = poseLabel.trim();
    if (!label || !currentHandRef.current) {
      setPredictionStatus(!label ? "Enter a label first" : "Show a hand first");
      return;
    }
    const template: PoseTemplate = {
      label,
      landmarks: normalizeHand(currentHandRef.current),
      samples: [normalizeHand(currentHandRef.current)],
      saved_at: new Date().toISOString(),
    };
    const nextTemplates = [
      ...poseTemplates.filter((item) => item.label.toLowerCase() !== label.toLowerCase()),
      template,
    ];
    setPoseTemplates(nextTemplates);
    void persistPoseTemplates(nextTemplates);
    setPoseLabel("");
    setPredictionStatus(`Saved pose: ${label}`);
  }

  function addPoseSample() {
    const label = poseLabel.trim();
    if (!label || !currentHandRef.current) {
      setPredictionStatus(!label ? "Enter a label first" : "Show a hand first");
      return;
    }
    const normalized = normalizeHand(currentHandRef.current);
    const existing = poseTemplates.find(
      (item) => item.label.toLowerCase() === label.toLowerCase(),
    );
    if (existing) {
      const existingSamples = existing.samples?.length
        ? existing.samples
        : [existing.landmarks];
      const nextTemplates = poseTemplates.map((item) =>
        item.label !== existing.label
          ? item
          : {
              ...item,
              landmarks: normalized,
              samples: [...existingSamples, normalized].slice(-MAX_POSE_SAMPLES),
              saved_at: new Date().toISOString(),
            },
      );
      setPoseTemplates(nextTemplates);
      void persistPoseTemplates(nextTemplates);
      setPredictionStatus(`Added sample: ${label} (${Math.min(existingSamples.length + 1, MAX_POSE_SAMPLES)}/${MAX_POSE_SAMPLES})`);
    } else {
      const template: PoseTemplate = {
        label,
        landmarks: normalized,
        samples: [normalized],
        saved_at: new Date().toISOString(),
      };
      const nextTemplates = [...poseTemplates, template];
      setPoseTemplates(nextTemplates);
      void persistPoseTemplates(nextTemplates);
      setPredictionStatus(`Added first sample: ${label}`);
    }
  }

  function deletePose(label: string) {
    const nextTemplates = poseTemplates.filter((item) => item.label !== label);
    setPoseTemplates(nextTemplates);
    void persistPoseTemplates(nextTemplates);
    setPredictionStatus(`Deleted pose: ${label}`);
  }

  async function persistPoseTemplates(templates: PoseTemplate[]) {
    try {
      const response = await fetch("/api/poses", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ templates }),
      });
      if (!response.ok) throw new Error(`Pose save failed (${response.status})`);
    } catch {
      setPredictionStatus("Unable to save poses to project");
    }
  }

  function updateTrackingFps() {
    const windowState = fpsWindowRef.current;
    windowState.count += 1;
    const elapsed = performance.now() - windowState.startedAt;
    if (elapsed >= 1000) {
      setTrackingFps(Math.round((windowState.count * 1000) / elapsed));
      windowState.count = 0;
      windowState.startedAt = performance.now();
    }
  }

  function startTrackingFrames() {
    const video = videoRef.current as (HTMLVideoElement & {
      requestVideoFrameCallback?: (callback: (now: number, metadata: unknown) => void) => number;
      cancelVideoFrameCallback?: (id: number) => void;
    }) | null;
    if (!video?.requestVideoFrameCallback) {
      usingVideoFrameCallbackRef.current = false;
      frameIdRef.current = window.requestAnimationFrame(trackHands);
      return;
    }
    usingVideoFrameCallbackRef.current = true;
    const callback = () => {
      if (!trackingActiveRef.current) return;
      trackHands();
      videoCallbackIdRef.current = video.requestVideoFrameCallback!(callback);
    };
    videoCallbackIdRef.current = video.requestVideoFrameCallback(callback);
  }

  function drawHands(frame: ReturnType<typeof extractHands>) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const rect = video.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
      canvas.width = Math.round(rect.width);
      canvas.height = Math.round(rect.height);
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);

    const intrinsicWidth = video.videoWidth || rect.width;
    const intrinsicHeight = video.videoHeight || rect.height;
    const coverScale = Math.max(rect.width / intrinsicWidth, rect.height / intrinsicHeight);
    const renderedWidth = intrinsicWidth * coverScale;
    const renderedHeight = intrinsicHeight * coverScale;
    const contentOffsetX = (rect.width - renderedWidth) / 2;
    const contentOffsetY = (rect.height - renderedHeight) / 2;
    const now = performance.now();
    const visibleTrailDuration = 600;
    const isTrailVisible = ([, , , timestamp]: [number, number, number, number]) =>
      now - timestamp < visibleTrailDuration;
    const wristTrail = visualMotionRef.current.filter(isTrailVisible);
    if (wristTrail.length >= 2) {
      context.lineWidth = 4;
      context.lineCap = "round";
      context.lineJoin = "round";
      const wristPoints = wristTrail.map(([x, y]) => ({
        x: rect.width - (contentOffsetX + x * renderedWidth),
        y: contentOffsetY + y * renderedHeight,
      }));
      for (let index = 1; index < wristPoints.length; index += 1) {
        const previous = wristPoints[index - 1];
        const point = wristPoints[index];
        const age = now - wristTrail[index][3];
        const opacity = Math.max(0, 1 - age / visibleTrailDuration);
        context.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.95})`;
        context.beginPath();
        context.moveTo(previous.x, previous.y);
        const midpointX = (previous.x + point.x) / 2;
        const midpointY = (previous.y + point.y) / 2;
        context.quadraticCurveTo(previous.x, previous.y, midpointX, midpointY);
        context.stroke();
      }
    }
    visualFingerMotionRef.current.forEach((trail) => {
      if (trail.length < 2) return;
      context.lineWidth = 2.5;
      context.lineCap = "round";
      context.lineJoin = "round";
      const visiblePoints = trail.filter(isTrailVisible);
      const fingerPoints = visiblePoints.map(([x, y]) => ({
        x: rect.width - (contentOffsetX + x * renderedWidth),
        y: contentOffsetY + y * renderedHeight,
      }));
      for (let index = 1; index < fingerPoints.length; index += 1) {
        const previous = fingerPoints[index - 1];
        const point = fingerPoints[index];
        const age = now - visiblePoints[index][3];
        const opacity = Math.max(0, 1 - age / visibleTrailDuration);
        context.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.9})`;
        context.beginPath();
        context.moveTo(previous.x, previous.y);
        const midpointX = (previous.x + point.x) / 2;
        const midpointY = (previous.y + point.y) / 2;
        context.quadraticCurveTo(previous.x, previous.y, midpointX, midpointY);
        context.stroke();
      }
    });
    if (!showLandmarksRef.current) return;
    const mapPoint = (x: number, y: number) => ({
      x: rect.width - (contentOffsetX + x * renderedWidth),
      y: contentOffsetY + y * renderedHeight,
    });

    const handConnections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16],
      [13, 17], [17, 18], [18, 19], [19, 20],
      [0, 17],
    ];

    const hands = [
      { key: "left" as const, detected: frame.left_hand },
      { key: "right" as const, detected: frame.right_hand },
    ];
    for (const { key, detected } of hands) {
      let hand = detected;
      if (hand) {
        const previous = heldHandsRef.current[key];
        if (previous) {
          hand = hand.map((point, index) => [
            previous[index][0] * 0.35 + point[0] * 0.65,
            previous[index][1] * 0.35 + point[1] * 0.65,
            previous[index][2] * 0.35 + point[2] * 0.65,
          ] as [number, number, number]);
        }
        heldHandsRef.current[key] = hand;
        heldHandTimesRef.current[key] = now;
      } else if (now - heldHandTimesRef.current[key] > 220) {
        heldHandsRef.current[key] = null;
        hand = null;
      } else {
        hand = heldHandsRef.current[key];
      }
      if (!hand) continue;
      const points = hand.map(([x, y]) => mapPoint(x, y));

      context.strokeStyle = "#a855f7";
      context.lineWidth = 3;
      context.lineCap = "round";
      context.lineJoin = "round";
      context.shadowColor = "rgba(168, 85, 247, 0.7)";
      context.shadowBlur = 5;
      for (const [startIndex, endIndex] of handConnections) {
        const start = points[startIndex];
        const end = points[endIndex];
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
      }

      context.shadowColor = "rgba(255, 255, 255, 0.45)";
      context.shadowBlur = 3;
      context.fillStyle = "#ffffff";
      for (const point of points) {
        context.beginPath();
        context.arc(point.x, point.y, 4, 0, Math.PI * 2);
        context.fill();
      }
    }
  }

  function stopCamera() {
    trackingActiveRef.current = false;
    setTrackedHands({ left: false, right: false });
    heldHandsRef.current = { left: null, right: null };
    (videoRef.current?.srcObject as MediaStream | null)?.getTracks().forEach((track) => track.stop());
    videoRef.current!.srcObject = null;
    if (frameIdRef.current !== null) window.cancelAnimationFrame(frameIdRef.current);
    frameIdRef.current = null;
    const videoWithCallbacks = videoRef.current as (HTMLVideoElement & {
      cancelVideoFrameCallback?: (id: number) => void;
    }) | null;
    if (videoCallbackIdRef.current !== null) {
      videoWithCallbacks?.cancelVideoFrameCallback?.(videoCallbackIdRef.current);
      videoCallbackIdRef.current = null;
    }
    landmarkerRef.current?.close();
    landmarkerRef.current = null;
    setCameraState("Camera stopped");
    setTranslation("No translation");
    setConfidence(0);
    setPredictionStatus("Model idle");
    currentHandRef.current = null;
    setTypedHistory("");
    typedHistoryRef.current = "";
  }

  function clearTyping() {
    setHistory([]);
    setTypedHistory("");
    typedHistoryRef.current = "";
    poseCandidateRef.current = null;
    lastTypedRef.current = null;
    staticRepeatBlockRef.current = null;
  }

  function startEvaluation() {
    if (!trackingActiveRef.current) {
      setPredictionStatus("Start the camera before starting an evaluation");
      return;
    }
    const run = createEvaluationRun(evaluationClassCount, evaluationTrialsPerLetter);
    evaluationRunRef.current = run;
    setEvaluationRun(run);
    poseCandidateRef.current = null;
    lastTypedRef.current = null;
    staticRepeatBlockRef.current = null;
    setPredictionStatus("Evaluation started — sign the target shown below");
  }

  /** Called only after the recognizer has produced a label; the target is never an input to matching. */
  function recordEvaluationPrediction(predictedLetter: string, predictionConfidence: number) {
    const run = evaluationRunRef.current;
    if (!run || run.completedAt) return;
    const nextRun = recordEvaluationTrial(run, predictedLetter, predictionConfidence);
    evaluationRunRef.current = nextRun;
    setEvaluationRun(nextRun);
    setPredictionStatus(
      nextRun.completedAt
        ? "Evaluation complete"
        : `Trial recorded — ${nextRun.trials.length}/${nextRun.targetOrder.length}`,
    );
  }

  function exportEvaluationResults() {
    if (!evaluationRun) return;
    const blob = new Blob([`\uFEFF${evaluationResultsCsv(evaluationRun)}`], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `fsl-evaluation-${evaluationRun.runId}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="app-shell">
      <aside className="rail">
        <div className="brand" aria-label="FSLT">
          <span className="brand-mark">F</span>
          <span>FSLT</span>
        </div>
        <div className="rail-status">
          <span className={`live-dot ${trackingActiveRef.current ? "active" : ""}`} />
          <span>{trackingActiveRef.current ? "Live" : "Offline"}</span>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Saved pose translator</p>
            <h1>Live recognition</h1>
          </div>
          <div className="camera-actions">
            <button
              className="primary-button"
              disabled={cameraStarting || trackingActiveRef.current}
              onClick={startCamera}
            >
              <span className="button-icon">●</span>
              {cameraStarting ? "Starting" : "Start camera"}
            </button>
            <button
              className="icon-button"
              disabled={!trackingActiveRef.current}
              onClick={stopCamera}
              aria-label="Stop camera"
              title="Stop camera"
            >
              ■
            </button>
            <button
              className="clear-button"
              onClick={clearTyping}
              disabled={history.length === 0}
              aria-label="Clear typing"
              title="Clear typing"
            >
              Clear
            </button>
          </div>
        </header>

        <div className="content-grid">
          <section className="camera-stage">
            <div className="video-wrap">
              <video ref={videoRef} muted playsInline aria-label="Webcam preview" />
              <canvas ref={canvasRef} />
              <div className="stage-topline">
                <span className="camera-state">
                  <span className={`live-dot ${trackingActiveRef.current ? "active" : ""}`} />
                  {cameraState}
                </span>
                <span className="fps">{trackingFps ? `${trackingFps} FPS` : "Measuring"}</span>
              </div>
              <div className="camera-history" aria-live="polite">
                {evaluationTarget ? (
                  <div className="evaluation-target">
                    <span>Sign this target</span>
                    <strong>{evaluationTarget}</strong>
                    <small>
                      Trial {(evaluationRun?.trials.length ?? 0) + 1} of {evaluationRun?.targetOrder.length}
                    </small>
                  </div>
                ) : (
                  <>
                    {typedHistory}
                    <span className="typing-caret" aria-hidden="true" />
                  </>
                )}
              </div>
            </div>
          </section>

          <section className="confidence-panel" aria-label="Recognition confidence">
            <div className="confidence-row">
              <div className="confidence-track">
                <div
                  className="confidence-fill"
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
              <strong>{(confidence * 100).toFixed(0)}%</strong>
            </div>
          </section>
        </div>

        <section className="bottom-deck">
          <div className="pose-trainer">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Trainer</p>
                <h2>Pose library</h2>
              </div>
              <div className="pose-tools">
                <span className="pose-count">{poseTemplates.length} saved</span>
                {poseLoadError && (
                  <span className="pose-load-error">{poseLoadError}</span>
                )}
                <label className="overlay-toggle">
                  <input
                    type="checkbox"
                    checked={showLandmarks}
                    onChange={(event) => setShowLandmarks(event.target.checked)}
                  />
                  <span />
                  Skeleton
                </label>
              </div>
            </div>
            <div className="pose-entry">
              <input
                value={poseLabel}
                onChange={(event) => setPoseLabel(event.target.value)}
                placeholder="Enter a letter or word"
                aria-label="Pose label"
              />
              <button
                className="secondary-button motion-button"
                disabled={!trackingActiveRef.current || motionRecording}
                onClick={startMotionRecording}
              >
                {motionRecording
                  ? `Recording ${Math.max(0, MOTION_RECORDING_MS / 1000 - motionSeconds).toFixed(1)}s`
                  : "Record motion"}
              </button>
              <button
                className="secondary-button"
                disabled={!trackingActiveRef.current}
                onClick={addPoseSample}
              >
                + Add sample
              </button>
              <button
                className="secondary-button"
                disabled={!trackingActiveRef.current}
                onClick={saveCurrentPose}
              >
                Replace
              </button>
            </div>
            <div className="pose-list" aria-label="Saved poses">
              {poseTemplates.map((pose) => (
                <div className="pose-chip" key={pose.label}>
                  <span className="pose-letter">{pose.label}</span>
                  <span
                    className="pose-samples"
                    title={`${pose.samples?.length ?? 1} static, ${pose.motion_samples?.length ?? 0} motion`}
                  >
                    {countPoseSamples(pose)} {countPoseSamples(pose) === 1 ? "sample" : "samples"}
                  </span>
                  <button
                    onClick={() => deletePose(pose.label)}
                    aria-label={`Delete ${pose.label}`}
                    title={`Delete ${pose.label}`}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="evaluation-panel">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Evaluation mode</p>
                <h2>Randomized static fingerspelling evaluation</h2>
              </div>
              <button
                className="secondary-button"
                onClick={exportEvaluationResults}
                disabled={!evaluationRun?.completedAt}
              >
                Export CSV
              </button>
            </div>

            <p className="evaluation-note">
              Targets are sampled only from the 24 supported static letters: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y.
              The live recognizer makes its prediction independently; every recognized result is recorded automatically.
            </p>

            <div className="evaluation-config" aria-label="Evaluation configuration">
              <div className="evaluation-config-group">
                <span className="evaluation-label">Selected letters</span>
                <div className="evaluation-choice-group" role="group" aria-label="Number of random letters">
                  {([10, 20] as const).map((count) => (
                    <button
                      key={count}
                      className={`evaluation-choice ${evaluationClassCount === count ? "selected" : ""}`}
                      disabled={Boolean(evaluationTarget)}
                      onClick={() => setEvaluationClassCount(count)}
                    >
                      {count} random letters
                    </button>
                  ))}
                </div>
              </div>
              <label className="evaluation-config-group" htmlFor="evaluation-trials">
                <span className="evaluation-label">Trials per selected letter</span>
                <input
                  id="evaluation-trials"
                  className="evaluation-trials-input"
                  type="number"
                  min="1"
                  max="20"
                  step="1"
                  value={evaluationTrialsPerLetter}
                  disabled={Boolean(evaluationTarget)}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    if (Number.isInteger(value) && value >= 1 && value <= 20) {
                      setEvaluationTrialsPerLetter(value);
                    }
                  }}
                />
              </label>
              <button
                className="primary-button evaluation-start"
                disabled={!trackingActiveRef.current || Boolean(evaluationTarget)}
                onClick={startEvaluation}
              >
                {evaluationRun ? "Start new randomized run" : "Start randomized evaluation"}
              </button>
            </div>

            {evaluationRun && evaluationResults && (
              <div className="evaluation-results" aria-live="polite">
                <div className="evaluation-progress">
                  <span>
                    {evaluationRun.completedAt ? "Completed" : "In progress"}
                  </span>
                  <strong>{evaluationRun.trials.length} / {evaluationRun.targetOrder.length} trials recorded</strong>
                </div>
                <div className="evaluation-summary-grid">
                  <div><span>Overall accuracy</span><strong>{(evaluationResults.overallAccuracy * 100).toFixed(1)}%</strong></div>
                  <div><span>Correct</span><strong>{evaluationResults.correctCount}</strong></div>
                  <div><span>Incorrect</span><strong>{evaluationResults.incorrectCount}</strong></div>
                  <div><span>Macro F1</span><strong>{evaluationResults.macroF1.toFixed(3)}</strong></div>
                  <div><span>Macro precision</span><strong>{evaluationResults.macroPrecision.toFixed(3)}</strong></div>
                  <div><span>Macro recall</span><strong>{evaluationResults.macroRecall.toFixed(3)}</strong></div>
                </div>

                <div className="evaluation-table-wrap">
                  <h3>Per-class metrics</h3>
                  <table>
                    <thead>
                      <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Correct</th><th>Incorrect</th></tr>
                    </thead>
                    <tbody>
                      {evaluationResults.perClass.map((metric) => (
                        <tr key={metric.letter}>
                          <td>{metric.letter}</td>
                          <td>{metric.precision.toFixed(3)}</td>
                          <td>{metric.recall.toFixed(3)}</td>
                          <td>{metric.f1Score.toFixed(3)}</td>
                          <td>{metric.correctCount}</td>
                          <td>{metric.incorrectCount}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="evaluation-table-wrap">
                  <h3>Confusion matrix</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Expected \ Predicted</th>
                        {evaluationResults.confusionLabels.map((label) => <th key={label}>{label}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {evaluationResults.confusionMatrix.map((row, rowIndex) => (
                        <tr key={evaluationResults.perClass[rowIndex].letter}>
                          <th>{evaluationResults.perClass[rowIndex].letter}</th>
                          {row.map((count, columnIndex) => <td key={evaluationResults.confusionLabels[columnIndex]}>{count}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="evaluation-table-wrap">
                  <h3>Recorded trials</h3>
                  <table>
                    <thead>
                      <tr><th>Trial ID</th><th>Expected</th><th>Predicted</th><th>Confidence</th><th>Timestamp</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {evaluationRun.trials.map((trial) => (
                        <tr key={trial.trialId} className={trial.correct ? "trial-correct" : "trial-incorrect"}>
                          <td>{trial.trialId}</td>
                          <td>{trial.expectedLetter}</td>
                          <td>{trial.predictedLetter ?? "No prediction"}</td>
                          <td>{(trial.confidence * 100).toFixed(1)}%</td>
                          <td>{new Date(trial.timestamp).toLocaleString()}</td>
                          <td>{trial.correct ? "Correct" : "Incorrect"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}
