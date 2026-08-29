#!/usr/bin/env python3

import subprocess
import re
import time
import json
import csv
import os
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "./benchmarking"

PLUNDER_CMD = [
    "python3",
    "plunder.py",
    DATASET
]

TRUFFLEHOG_CMD = [
    "sudo",
    "trufflehog",
    "--no-verification",
    "filesystem",
    DATASET
]

PLUNDER_LOG = "benchmark_plunder.log"
TRUFFLEHOG_LOG = "benchmark_trufflehog.log"

# Synthetic ground-truth passwords planted in the dataset.
GROUND_TRUTH = {
    "pgSuperSecret!": "./benchmarking/postgres/pg.conf",
    "Repl123!": "./benchmarking/postgres/pg.conf",

    "SuperEnvPass123!": "./benchmarking/envs/.env",
    "jwtsecretkey123": "./benchmarking/envs/.env",
    "tok_9x8a7s6d5f": "./benchmarking/envs/.env",

    "Admin123!": "./benchmarking/apache/app.conf",
    "MySecretToken123!": "./benchmarking/apache/app.conf",
    "sk_test_51H8kLp9Z": "./benchmarking/apache/app.conf",

    "supersecret123": "./benchmarking/mysql/my.cnf",
    "R3pl!c@tionKey": "./benchmarking/mysql/my.cnf",
    "b@ckupP@ssw0rd!": "./benchmarking/mysql/my.cnf",
    "MyApp123!": "./benchmarking/mysql/my.cnf",

    "root123!": "./benchmarking/docker/docker-compose.yml",
    "MyDBPass!": "./benchmarking/docker/docker-compose.yml",
}

def run_program(command, output_file):
    print("\nRunning:")
    print(" ".join(command))

    start = time.perf_counter()

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    elapsed = time.perf_counter() - start

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    print(f"Finished in {elapsed:.2f} seconds")
    print(f"Output saved to {output_file}")

    return result.stdout, elapsed


def parse_plunder(output):
    """
    Parses lines such as:

    /path/file.yml -> SPIKE: MyDBPass! | entropy=50.38

    Returns a list of:
        {
            "secret": "...",
            "file": "..."
        }
    """

    detections = []

    pattern = re.compile(
        r"^(.*?)\s*->\s*SPIKE:\s*(.*?)\s*\|\s*entropy="
    )

    for line in output.splitlines():
        match = pattern.search(line)

        if match:
            filepath = match.group(1).strip()
            secret = match.group(2).strip()

            detections.append({
                "secret": secret,
                "file": filepath
            })

    return detections


def parse_trufflehog(output):
    """
    Parses TruffleHog output of the form:

    Raw result: http://user:pass@proxy.example.com
    File: /path/file
    Line: 2107

    We extract the Raw result and associated file.

    """

    detections = []

    current_result = None
    current_file = None

    for line in output.splitlines():

        if line.startswith("Raw result:"):
            current_result = line.split("Raw result:", 1)[1].strip()

        elif line.startswith("File:"):
            current_file = line.split("File:", 1)[1].strip()

        elif line.startswith("Line:"):

            if current_result is not None:

                detections.append({
                    "secret": current_result,
                    "file": current_file
                })

            current_result = None
            current_file = None

    return detections

def evaluate(detections):

    detected_truth = set()
    false_positives = []

    for detection in detections:

        detected_value = detection["secret"]

        # Exact match
        if detected_value in GROUND_TRUTH:
            detected_truth.add(detected_value)
        else:
            false_positives.append(detection)

    true_positives = len(detected_truth)

    false_negatives = len(
        set(GROUND_TRUTH.keys()) - detected_truth
    )

    false_positives_count = len(false_positives)

    total_truth = len(GROUND_TRUTH)

    # Precision
    if true_positives + false_positives_count > 0:
        precision = (
            true_positives /
            (true_positives + false_positives_count)
        )
    else:
        precision = 0.0

    # Recall / sensitivity
    if total_truth > 0:
        recall = true_positives / total_truth
    else:
        recall = 0.0

    # F1
    if precision + recall > 0:
        f1 = (
            2 * precision * recall /
            (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "TP": true_positives,
        "FP": false_positives_count,
        "FN": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detected_truth": sorted(detected_truth),
        "false_positives": false_positives
    }

def get_all_files(root):
    files = set()

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            files.add(
                os.path.normpath(
                    os.path.join(dirpath, filename)
                )
            )

    return files


def calculate_file_specificity(detections):

    all_files = get_all_files(DATASET)

    positive_files = set(
        os.path.normpath(path)
        for path in GROUND_TRUTH.values()
    )

    detected_files = set(
        os.path.normpath(d["file"])
        for d in detections
    )

    negative_files = all_files - positive_files

    # Negative files that were not flagged
    true_negative_files = negative_files - detected_files

    # Negative files incorrectly flagged
    false_positive_files = negative_files & detected_files

    TN = len(true_negative_files)
    FP = len(false_positive_files)

    if TN + FP > 0:
        specificity = TN / (TN + FP)
    else:
        specificity = None

    return {
        "TN_files": TN,
        "FP_files": FP,
        "specificity": specificity,
        "total_files": len(all_files),
        "positive_files": len(positive_files),
        "negative_files": len(negative_files)
    }

def print_results(name, metrics, file_metrics, runtime):

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Runtime:              {runtime:.2f} s")
    print()

    print(f"True Positives:       {metrics['TP']}")
    print(f"False Positives:      {metrics['FP']}")
    print(f"False Negatives:      {metrics['FN']}")
    print()

    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Recall/Sensitivity:   {metrics['recall']:.4f}")
    print(f"F1 Score:             {metrics['f1']:.4f}")
    print()

    print(f"True Negative Files:  {file_metrics['TN_files']}")
    print(f"False Positive Files: {file_metrics['FP_files']}")

    if file_metrics["specificity"] is not None:
        print(
            f"Specificity:          "
            f"{file_metrics['specificity']:.4f}"
        )
    else:
        print("Specificity:          N/A")

    print()

    print("Detected ground-truth passwords:")

    for password in metrics["detected_truth"]:
        print(f"    [TP] {password}")

    print()

    print("False-positive detections:")

    for fp in metrics["false_positives"]:
        print(
            f"    [FP] {fp['secret']} "
            f"-> {fp['file']}"
        )


def save_summary(results):

    with open(
        "benchmark_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4
        )

    with open(
        "benchmark_metrics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "tool",
            "runtime_seconds",
            "TP",
            "FP",
            "FN",
            "TN_files",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "f1"
        ])

        for tool, result in results.items():

            metrics = result["metrics"]
            file_metrics = result["file_metrics"]

            writer.writerow([
                tool,
                result["runtime"],
                metrics["TP"],
                metrics["FP"],
                metrics["FN"],
                file_metrics["TN_files"],
                metrics["precision"],
                metrics["recall"],
                metrics["recall"],
                file_metrics["specificity"],
                metrics["f1"]
            ])

    print("\nResults saved to:")
    print("    benchmark_results.json")
    print("    benchmark_metrics.csv")


def main():

    print("=" * 60)
    print("PLUNDER vs TRUFFLEHOG BENCHMARK")
    print("=" * 60)

    plunder_output, plunder_time = run_program(
        PLUNDER_CMD,
        PLUNDER_LOG
    )

    plunder_detections = parse_plunder(
        plunder_output
    )

    plunder_metrics = evaluate(
        plunder_detections
    )

    plunder_file_metrics = calculate_file_specificity(
        plunder_detections
    )

    print_results(
        "PLUNDER",
        plunder_metrics,
        plunder_file_metrics,
        plunder_time
    )

    truffle_output, truffle_time = run_program(
        TRUFFLEHOG_CMD,
        TRUFFLEHOG_LOG
    )

    truffle_detections = parse_trufflehog(
        truffle_output
    )

    truffle_metrics = evaluate(
        truffle_detections
    )

    truffle_file_metrics = calculate_file_specificity(
        truffle_detections
    )

    print_results(
        "TRUFFLEHOG",
        truffle_metrics,
        truffle_file_metrics,
        truffle_time
    )
  
    results = {

        "Plunder": {
            "runtime": plunder_time,
            "metrics": plunder_metrics,
            "file_metrics": plunder_file_metrics
        },

        "TruffleHog": {
            "runtime": truffle_time,
            "metrics": truffle_metrics,
            "file_metrics": truffle_file_metrics
        }
    }

    save_summary(results)


if __name__ == "__main__":
    main()
