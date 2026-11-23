/**
 * Evaluation is deliberately limited to static FSL fingerspelling targets.
 * J and Z are dynamic signs, so they remain available to the recognizer but
 * are never evaluation targets or metric classes.
 */
export const EVALUATION_ALPHABET = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
] as const;

export type EvaluationLetter = typeof EVALUATION_ALPHABET[number];

export type EvaluationTrial = {
  trialId: string;
  expectedLetter: EvaluationLetter;
  predictedLetter: string | null;
  confidence: number;
  timestamp: string;
  correct: boolean;
};

export type EvaluationRun = {
  runId: string;
  selectedLetters: EvaluationLetter[];
  targetOrder: EvaluationLetter[];
  trialsPerLetter: number;
  startedAt: string;
  completedAt: string | null;
  trials: EvaluationTrial[];
};

export type PerClassMetric = {
  letter: EvaluationLetter;
  precision: number;
  recall: number;
  f1Score: number;
  correctCount: number;
  incorrectCount: number;
};

export type EvaluationResults = {
  overallAccuracy: number;
  correctCount: number;
  incorrectCount: number;
  perClass: PerClassMetric[];
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  confusionLabels: string[];
  confusionMatrix: number[][];
};

export function isEvaluationLetter(value: string): value is EvaluationLetter {
  return (EVALUATION_ALPHABET as readonly string[]).includes(value);
}

function shuffle<T>(items: readonly T[], random: () => number): T[] {
  const shuffled = [...items];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function createRunId(now: Date): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `evaluation-${now.getTime()}-${Math.random().toString(36).slice(2, 10)}`;
}

/** Selects unique classes and independently randomizes every scheduled trial. */
export function createEvaluationRun(
  selectedClassCount: 10 | 20,
  trialsPerLetter: number,
  options: { random?: () => number; now?: Date } = {},
): EvaluationRun {
  if (selectedClassCount !== 10 && selectedClassCount !== 20) {
    throw new Error("Evaluation must select either 10 or 20 letters");
  }
  if (!Number.isInteger(trialsPerLetter) || trialsPerLetter < 1) {
    throw new Error("Trials per letter must be a positive integer");
  }
  const random = options.random ?? Math.random;
  const now = options.now ?? new Date();
  const selectedLetters = shuffle(EVALUATION_ALPHABET, random).slice(0, selectedClassCount);
  const targetOrder = shuffle(
    selectedLetters.flatMap((letter) => Array<EvaluationLetter>(trialsPerLetter).fill(letter)),
    random,
  );
  return {
    runId: createRunId(now),
    selectedLetters,
    targetOrder,
    trialsPerLetter,
    startedAt: now.toISOString(),
    completedAt: null,
    trials: [],
  };
}

export function currentEvaluationTarget(run: EvaluationRun): EvaluationLetter | null {
  return run.targetOrder[run.trials.length] ?? null;
}

function normalizedPrediction(prediction: string | null): string | null {
  const value = prediction?.trim();
  return value ? value.toUpperCase() : null;
}

/**
 * Records exactly what the recognizer returned. In particular, dynamic J/Z
 * predictions are retained and evaluated as incorrect static-target results.
 */
export function recordEvaluationTrial(
  run: EvaluationRun,
  prediction: string | null,
  confidence: number,
  timestamp = new Date(),
): EvaluationRun {
  const expectedLetter = currentEvaluationTarget(run);
  if (!expectedLetter) {
    throw new Error("Evaluation run is already complete");
  }
  const predictedLetter = normalizedPrediction(prediction);
  const safeConfidence = Number.isFinite(confidence)
    ? Math.min(1, Math.max(0, confidence))
    : 0;
  const nextTrial: EvaluationTrial = {
    trialId: `${run.runId}-${String(run.trials.length + 1).padStart(4, "0")}`,
    expectedLetter,
    predictedLetter,
    confidence: safeConfidence,
    timestamp: timestamp.toISOString(),
    correct: predictedLetter === expectedLetter,
  };
  const trials = [...run.trials, nextTrial];
  return {
    ...run,
    trials,
    completedAt: trials.length === run.targetOrder.length ? timestamp.toISOString() : null,
  };
}

function divide(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
}

/**
 * Metrics have only selected static target letters as classes. Predictions of
 * an unselected letter, J, Z, or no label are represented as "Other prediction"
 * in the matrix while their full value remains in the trial log.
 */
export function calculateEvaluationResults(run: EvaluationRun): EvaluationResults {
  const classes = run.selectedLetters;
  const correctCount = run.trials.filter((trial) => trial.correct).length;
  const perClass = classes.map((letter) => {
    const truePositive = run.trials.filter(
      (trial) => trial.expectedLetter === letter && trial.predictedLetter === letter,
    ).length;
    const falsePositive = run.trials.filter(
      (trial) => trial.expectedLetter !== letter && trial.predictedLetter === letter,
    ).length;
    const falseNegative = run.trials.filter(
      (trial) => trial.expectedLetter === letter && trial.predictedLetter !== letter,
    ).length;
    const precision = divide(truePositive, truePositive + falsePositive);
    const recall = divide(truePositive, truePositive + falseNegative);
    return {
      letter,
      precision,
      recall,
      f1Score: divide(2 * precision * recall, precision + recall),
      correctCount: truePositive,
      incorrectCount: falseNegative,
    };
  });
  const confusionLabels = [...classes, "Other prediction"];
  const confusionMatrix = classes.map((expected) =>
    confusionLabels.map((predicted) => run.trials.filter((trial) => {
      if (trial.expectedLetter !== expected) return false;
      if (predicted === "Other prediction") {
        return !classes.includes(trial.predictedLetter as EvaluationLetter);
      }
      return trial.predictedLetter === predicted;
    }).length),
  );
  return {
    overallAccuracy: divide(correctCount, run.trials.length),
    correctCount,
    incorrectCount: run.trials.length - correctCount,
    perClass,
    macroPrecision: divide(perClass.reduce((sum, metric) => sum + metric.precision, 0), perClass.length),
    macroRecall: divide(perClass.reduce((sum, metric) => sum + metric.recall, 0), perClass.length),
    macroF1: divide(perClass.reduce((sum, metric) => sum + metric.f1Score, 0), perClass.length),
    confusionLabels,
    confusionMatrix,
  };
}

function csvCell(value: string | number | boolean | null): string {
  const text = value === null ? "" : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function csvRow(values: Array<string | number | boolean | null>): string {
  return values.map(csvCell).join(",");
}

/** Exports trial-level evidence plus every required aggregate result in one CSV. */
export function evaluationResultsCsv(run: EvaluationRun): string {
  const results = calculateEvaluationResults(run);
  const rows: string[] = [
    "Evaluation run",
    csvRow(["run_id", run.runId]),
    csvRow(["started_at", run.startedAt]),
    csvRow(["completed_at", run.completedAt]),
    csvRow(["selected_class_count", run.selectedLetters.length]),
    csvRow(["selected_letters", run.selectedLetters.join(" ")]),
    csvRow(["trials_per_letter", run.trialsPerLetter]),
    csvRow(["completed_trials", run.trials.length]),
    "",
    "Trial records",
    csvRow(["trial_id", "expected_letter", "predicted_letter", "prediction_confidence", "timestamp", "correct"]),
    ...run.trials.map((trial) => csvRow([
      trial.trialId,
      trial.expectedLetter,
      trial.predictedLetter,
      trial.confidence,
      trial.timestamp,
      trial.correct,
    ])),
    "",
    "Summary",
    csvRow(["overall_accuracy", results.overallAccuracy]),
    csvRow(["correct_predictions", results.correctCount]),
    csvRow(["incorrect_predictions", results.incorrectCount]),
    csvRow(["macro_precision", results.macroPrecision]),
    csvRow(["macro_recall", results.macroRecall]),
    csvRow(["macro_f1", results.macroF1]),
    "",
    "Per-class metrics",
    csvRow(["class", "precision", "recall", "f1_score", "correct_predictions", "incorrect_predictions"]),
    ...results.perClass.map((metric) => csvRow([
      metric.letter,
      metric.precision,
      metric.recall,
      metric.f1Score,
      metric.correctCount,
      metric.incorrectCount,
    ])),
    "",
    "Confusion matrix",
    csvRow(["expected \\ predicted", ...results.confusionLabels]),
    ...results.confusionMatrix.map((row, rowIndex) =>
      csvRow([results.perClass[rowIndex].letter, ...row]),
    ),
  ];
  return rows.join("\r\n");
}
