import { describe, expect, it } from "vitest";
import {
  EVALUATION_ALPHABET,
  calculateEvaluationResults,
  createEvaluationRun,
  evaluationResultsCsv,
  recordEvaluationTrial,
} from "./evaluation";

describe("Evaluation Mode", () => {
  it("uses exactly the 24 static fingerspelling targets and never selects dynamic letters", () => {
    expect(EVALUATION_ALPHABET).toHaveLength(24);
    expect(EVALUATION_ALPHABET).not.toContain("J");
    expect(EVALUATION_ALPHABET).not.toContain("Z");
    const run = createEvaluationRun(20, 2, { random: () => 0.37 });
    expect(run.selectedLetters).toHaveLength(20);
    expect(run.targetOrder).toHaveLength(40);
    expect(run.targetOrder.every((letter) => EVALUATION_ALPHABET.includes(letter))).toBe(true);
  });

  it("retains an out-of-alphabet prediction as an incorrect trial", () => {
    let run = createEvaluationRun(10, 1, { random: () => 0.5, now: new Date("2026-11-24T00:00:00Z") });
    run = recordEvaluationTrial(run, "J", 0.82, new Date("2026-11-24T00:00:01Z"));
    expect(run.trials[0]).toMatchObject({ predictedLetter: "J", confidence: 0.82, correct: false });
    const results = calculateEvaluationResults(run);
    expect(results.confusionLabels).not.toContain("J");
    expect(results.confusionMatrix.flat().reduce((sum, count) => sum + count, 0)).toBe(1);
    expect(evaluationResultsCsv(run)).toContain(",J,0.82,");
  });

  it("calculates class metrics from recorded trials without changing the selected classes", () => {
    let run = createEvaluationRun(10, 1, { random: () => 0.12 });
    for (const expected of run.targetOrder) {
      run = recordEvaluationTrial(run, expected, 0.9);
    }
    const results = calculateEvaluationResults(run);
    expect(run.completedAt).not.toBeNull();
    expect(results.overallAccuracy).toBe(1);
    expect(results.macroF1).toBe(1);
    expect(results.perClass.map((metric) => metric.letter)).toEqual(run.selectedLetters);
  });
});
