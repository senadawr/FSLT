import { useEffect, useRef, useState } from "react";
import { HandLandmarker } from "@mediapipe/tasks-vision";
import { createHandExtractor, extractHands } from "./holistic";
import {
  MAX_POSE_SAMPLES,
  PoseTemplate,
  mirrorHand,
  normalizeHand,
  poseConfidence,
  poseDistance,
} from "./collection";

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
  const compactTranslation = ["No close match", "Hold steady", "No translation"].includes(translation)
    ? "—"
    : translation;

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

    } catch {
      setCameraState("Landmark extraction error");
    }
    if (!usingVideoFrameCallbackRef.current) {
      frameIdRef.current = window.requestAnimationFrame(trackHands);
    }
  }

  function matchPose(hand: Array<[number, number, number]> | null) {
    if (!hand || poseTemplates.length === 0) {
      setConfidence(0);
      setPredictionStatus(poseTemplates.length ? "Show a hand" : "No poses saved");
      poseCandidateRef.current = null;
      if (!hand) lastTypedRef.current = null;
      return;
    }
    const normalized = normalizeHand(hand);
    const matches = poseTemplates.map((template) => {
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
    const now = performance.now();
    const isAmbiguous = runnerUp !== undefined && margin < 0.018;
    const recognized = nextConfidence >= 0.68 && !isAmbiguous;

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
    if (!shouldAppendRecognition(lastTyped?.label ?? null, closest.template.label)) return;

    setTranslation(closest.template.label);
    lastTypedRef.current = { label: closest.template.label, at: now };
    setHistory((items) => [...items, closest.template.label]);
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
    if (!showLandmarksRef.current) return;

    const intrinsicWidth = video.videoWidth || rect.width;
    const intrinsicHeight = video.videoHeight || rect.height;
    const coverScale = Math.max(rect.width / intrinsicWidth, rect.height / intrinsicHeight);
    const renderedWidth = intrinsicWidth * coverScale;
    const renderedHeight = intrinsicHeight * coverScale;
    const contentOffsetX = (rect.width - renderedWidth) / 2;
    const contentOffsetY = (rect.height - renderedHeight) / 2;
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

    const now = performance.now();
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
                {typedHistory}
                <span className="typing-caret" aria-hidden="true" />
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
                  <span className="pose-samples">{pose.samples?.length ?? 1} samples</span>
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

        </section>
      </section>
    </main>
  );
}
