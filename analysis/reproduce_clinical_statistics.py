import argparse
import collections
import json
from pathlib import Path

import numpy as np

SEED = 20260804
NBOOT = 10000
THRESHOLDS = [25, 50, 70]
CANON = {
    "proximal": {"proximal"},
    "proximal-mid": {"proximal", "mid"},
    "mid": {"mid"},
    "mid-distal": {"mid", "distal"},
    "distal": {"distal"},
}

def category(x):
    x = np.asarray(x)
    return np.where(
        (x == 0),
        0,
        np.where(
            (x < 25), 1, np.where((x < 50), 2, np.where((x < 70), 3, np.where((x < 100), 4, 5)))
        ),
    ).astype(int)

def ratio(a, b):
    a, b = np.asarray(a, dtype = float), np.asarray(b, dtype = float)
    return np.divide(
        a, b, out = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan), where = (b != 0)
    )

def weighted_ranks(values, weights):
    values = np.asarray(values)
    if (values.size == 0):
        return np.zeros_like(weights, dtype = float)
    _, inverse = np.unique(values, return_inverse = True)
    grouped = np.stack(
        [weights[:, (inverse == i)].sum(axis = 1) for i in range((inverse.max() + 1))], axis = 1
    )
    ranks = ((np.cumsum(grouped, axis = 1) - (grouped / 2)) + 0.5)
    return ranks[:, inverse]

def weighted_spearman(x, y, weights):
    rx, ry = weighted_ranks(x, weights), weighted_ranks(y, weights)
    center = ((weights.sum(axis = 1) + 1)[:, None] / 2)
    dx, dy = (rx - center), (ry - center)
    return ratio(
        ((weights * dx) * dy).sum(axis = 1),
        np.sqrt((((weights * dx) * dx).sum(axis = 1) * ((weights * dy) * dy).sum(axis = 1))),
    )

def weighted_kappa(xcat, ycat, weights, power = 1):
    n = weights.sum(axis = 1)
    obs = ratio((weights * (np.abs((xcat - ycat)) ** power)).sum(axis = 1), n)
    xm = np.stack([weights[:, (xcat == i)].sum(axis = 1) for i in range(6)], axis = 1)
    ym = np.stack([weights[:, (ycat == i)].sum(axis = 1) for i in range(6)], axis = 1)
    dist = (np.abs((np.arange(6)[:, None] - np.arange(6)[None, :])) ** power)
    exp = ratio(np.einsum("bi,ij,bj->b", xm, dist, ym), (n * n))
    return (1 - ratio(obs, exp))

def weighted_auc(score, reference_positive, weights):
    ranks = weighted_ranks(score, weights)
    npos = weights[:, reference_positive].sum(axis = 1)
    nneg = weights[:, ~reference_positive].sum(axis = 1)
    score_sum = (weights[:, reference_positive] * ranks[:, reference_positive]).sum(axis = 1)
    return ratio((score_sum - ((npos * (npos + 1)) / 2)), (npos * nneg))

def bootstrap_weights(rows, cohort_cases):
    if not cohort_cases:
        if rows:
            raise ValueError("Nonempty analysis rows require a patient cohort.")
        return np.empty((0, 0), dtype = float)
    rng = np.random.default_rng(SEED)
    ids = {c: i for i, c in enumerate(cohort_cases)}
    draws = rng.integers(0, len(cohort_cases), size = (NBOOT, len(cohort_cases)))
    counts = np.stack([(draws == i).sum(axis = 1) for i in range(len(cohort_cases))], axis = 1)
    return counts[:, [ids[r["case"]] for r in rows]].astype(float)

def interval(v):
    valid = np.asarray(v)[np.isfinite(v)]
    return {
        "lower": (float(np.percentile(valid, 2.5)) if len(valid) else None),
        "upper": (float(np.percentile(valid, 97.5)) if len(valid) else None),
        "valid_resamples": len(valid),
        "attempted_resamples": len(v),
    }

def number(x):
    return (float(x) if np.isfinite(x) else None)

def evaluate(rows, left, right, cohort_cases, bootstrap = True, features = False):
    rs = [r for r in rows if ((r[left] is not None) and (r[right] is not None))]
    x = np.array([r[left] for r in rs], float)
    y = np.array([r[right] for r in rs], float)
    w = np.ones((1, len(rs)))
    xc, yc = category(x), category(y)
    out = {
        "left_source": left,
        "reference_source": right,
        "n": len(rs),
        "cohort_cases": len(cohort_cases),
        "contributing_cases": len({r["case"] for r in rs}),
        "case_ids": sorted({r["case"] for r in rs}),
        "denominator_by_territory": dict(collections.Counter(r["territory"] for r in rs)),
        "rho_DS": number(weighted_spearman(x, y, w)[0]),
        "rho_category": number(weighted_spearman(xc, yc, w)[0]),
        "MAE": (float(np.mean(abs((x - y)))) if len(rs) else None),
        "bias_left_minus_reference": (float(np.mean((x - y))) if len(rs) else None),
        "linear_kappa": number(weighted_kappa(xc, yc, w)[0]),
        "quadratic_kappa": number(weighted_kappa(xc, yc, w, 2)[0]),
        "exact_n": int(np.sum((xc == yc))),
        "within_one_n": int(np.sum((abs((xc - yc)) <= 1))),
        "exact_fraction": (float(np.mean((xc == yc))) if len(rs) else None),
        "within_one_fraction": (float(np.mean((abs((xc - yc)) <= 1))) if len(rs) else None),
        "category_confusion": [
            [int(np.sum(((yc == i) & (xc == j)))) for j in range(6)] for i in range(6)
        ],
        "category_confusion_axes": "rows=reference category, columns=left category",
        "thresholds": {},
        "paired_keys": [((r["case"] + ":") + r["territory"]) for r in rs],
    }
    for cutoff in THRESHOLDS:
        pp, yp = (x >= cutoff), (y >= cutoff)
        tp, tn, fp, fn = [int(np.sum(z)) for z in [(pp & yp), (~pp & ~yp), (pp & ~yp), (~pp & yp)]]
        out["thresholds"][str(cutoff)] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "accuracy": number(ratio((tp + tn), len(rs))),
            "sensitivity": number(ratio(tp, (tp + fn))),
            "specificity": number(ratio(tn, (tn + fp))),
            "precision": number(ratio(tp, (tp + fp))),
            "F1": number(ratio((2 * tp), (((2 * tp) + fp) + fn))),
            "AUC": number(weighted_auc(x, yp, w)[0]),
        }
    if bootstrap:
        wb = bootstrap_weights(rs, cohort_cases)
        n = wb.sum(axis = 1)
        out["CI95"] = {
            "rho_DS": interval(weighted_spearman(x, y, wb)),
            "rho_category": interval(weighted_spearman(xc, yc, wb)),
            "MAE": interval(ratio((wb * abs((x - y))).sum(axis = 1), n)),
            "bias_left_minus_reference": interval(ratio((wb * (x - y)).sum(axis = 1), n)),
            "linear_kappa": interval(weighted_kappa(xc, yc, wb)),
            "quadratic_kappa": interval(weighted_kappa(xc, yc, wb, 2)),
        }
        for cutoff in THRESHOLDS:
            out["thresholds"][str(cutoff)]["AUC_CI95"] = interval(
                weighted_auc(x, (y >= cutoff), wb)
            )
    if features:
        out["features"] = {}
        for feature in ["AS", "length"]:
            fr = [r for r in rs if (r[feature] is not None)]
            fx = np.array([r[feature] for r in fr])
            fy = np.array([r[right] for r in fr])
            f = {
                "n": len(fr),
                "contributing_cases": len({r["case"] for r in fr}),
                "rho": number(weighted_spearman(fx, fy, np.ones((1, len(fr))))[0]),
                "paired_keys": [((r["case"] + ":") + r["territory"]) for r in fr],
            }
            if bootstrap:
                f["CI95"] = interval(weighted_spearman(fx, fy, bootstrap_weights(fr, cohort_cases)))
            out["features"][feature] = f
    return out

def localization(rows, left, right):
    out = {"n": 0, "exact_n": 0, "overlap_n": 0, "included": [], "dual_positive_unlocalized": []}
    for r in rows:
        if ((r[left] is None) or (r[right] is None) or (min(r[left], r[right]) < 25)):
            continue
        la, lb = r["locations"][left], r["locations"][right]
        key = ((r["case"] + ":") + r["territory"])
        if (not la or not lb):
            out["dual_positive_unlocalized"].append(key)
            continue
        ex = bool((set(la) & set(lb)))
        ov = any((CANON[a] & CANON[b]) for a in la for b in lb)
        out["n"] += 1
        out["exact_n"] += int(ex)
        out["overlap_n"] += int(ov)
        out["included"].append(
            {"key": key, "left_labels": la, "reference_labels": lb, "exact": ex, "overlap": ov}
        )
    out["exact_fraction"] = number(ratio(out["exact_n"], out["n"]))
    out["overlap_fraction"] = number(ratio(out["overlap_n"], out["n"]))
    return out

def add_proportion_cis(result, rows, ids):
    rs = [
        r
        for r in rows
        if ((r[result["left_source"]] is not None) and (r[result["reference_source"]] is not None))
    ]
    x = np.asarray([r[result["left_source"]] for r in rs])
    y = np.asarray([r[result["reference_source"]] for r in rs])
    wb = bootstrap_weights(rs, ids)
    n = wb.sum(axis = 1)
    for name, test in [
        ("exact_fraction", (category(x) == category(y))),
        ("within_one_fraction", (abs((category(x) - category(y))) <= 1)),
    ]:
        result["CI95"][name] = interval(ratio((wb * test).sum(axis = 1), n))
    for cutoff in (25, 50, 70):
        pp, yp = (x >= cutoff), (y >= cutoff)
        tp, tn, fp, fn = [
            (wb * v).sum(axis = 1) for v in [(pp & yp), (~pp & ~yp), (pp & ~yp), (~pp & yp)]
        ]
        result["thresholds"][str(cutoff)]["CI95"] = {
            "accuracy": interval(ratio((tp + tn), n)),
            "sensitivity": interval(ratio(tp, (tp + fn))),
            "specificity": interval(ratio(tn, (tn + fp))),
            "precision": interval(ratio(tp, (tp + fp))),
            "F1": interval(ratio((2 * tp), (((2 * tp) + fp) + fn))),
        }
    loc = result["localization"]
    included = {r["key"]: r for r in loc["included"]}
    lr = [r for r in rows if (r["source_key"] in included)]
    if lr:
        lw = bootstrap_weights(lr, ids)
        loc["CI95"] = {
            name: interval(
                ratio(
                    (lw * np.array([included[r["source_key"]][field] for r in lr])).sum(axis = 1),
                    lw.sum(axis = 1),
                )
            )
            for name, field in [("exact_fraction", "exact"), ("overlap_fraction", "overlap")]
        }

def numeric_check(actual, expected, path = "analyses"):
    if isinstance(expected, dict):
        if (not isinstance(actual, dict) or (set(actual) != set(expected))):
            raise ValueError(f"Reference structure differs at {path}.")
        return sum(
            numeric_check(actual[key], value, ((path + ".") + key))
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if (not isinstance(actual, list) or (len(actual) != len(expected))):
            raise ValueError(f"Reference list differs at {path}.")
        return sum(
            numeric_check(a, b, (path + f"[{i}]")) for i, (a, b) in enumerate(zip(actual, expected))
        )
    if (isinstance(expected, (float, int)) and not isinstance(expected, bool)):
        if (
            not isinstance(actual, (float, int))
            or isinstance(actual, bool)
            or not np.isfinite(expected)
            or not np.isfinite(actual)
            or not np.isclose(actual, expected, rtol = 1e-12, atol = 1e-12)
        ):
            raise ValueError(f"Reference numeric value differs at {path}.")
        return 1
    if ((type(actual) is not type(expected)) or (actual != expected)):
        raise ValueError(f"Reference value differs at {path}.")
    return 0

def read_rows(path, original_prefix, reviewed_prefix):
    ledger = json.loads(path.read_text(encoding = "utf-8"))
    if (not isinstance(ledger, dict) or not isinstance(ledger.get("records"), list)):
        raise ValueError("Input must be a JSON object containing a records list.")
    if (
        not original_prefix
        or not reviewed_prefix
        or original_prefix.startswith(reviewed_prefix)
        or reviewed_prefix.startswith(original_prefix)
    ):
        raise ValueError("Cohort prefixes must be nonempty and non-overlapping.")
    rows = []
    seen = set()
    for index, record in enumerate(ledger["records"]):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")
        case = record.get("study_code")
        territory = record.get("territory")
        if (not isinstance(case, str) or not case or (":" in case)):
            raise ValueError(f"Record {index} needs a nonempty study_code without ':'.")
        if not case.startswith((original_prefix, reviewed_prefix)):
            raise ValueError(f"Record {index} does not match either cohort prefix.")
        if (not isinstance(territory, str) or not territory or (":" in territory)):
            raise ValueError(f"Record {index} needs a nonempty territory without ':'.")
        if ((case, territory) in seen):
            raise ValueError(f"Record {index} duplicates a study/territory endpoint.")
        seen.add((case, territory))
        fully = record.get("fully_assessable_raw")
        if not isinstance(fully, bool):
            raise ValueError(f"Record {index} needs a boolean fully_assessable_raw.")
        row = {
            "case": case,
            "source_key": ((case + ":") + territory),
            "territory": territory,
            "fully": fully,
            "locations": {},
        }
        locations = record.get("locations", {})
        if not isinstance(locations, dict):
            raise ValueError(f"Record {index} locations must be an object.")
        for source in ("P", "H", "CAG"):
            component = record.get(source)
            if ((component is not None) and not isinstance(component, dict)):
                raise ValueError(f"Record {index} {source} must be an object or null.")
            component = (component or {})
            fields = [("analysis_value_pct", source, 100)]
            if (source == "P"):
                fields += [("area_stenosis_pct", "AS", 100), ("lesion_length_mm", "length", None)]
            for field, output_key, upper in fields:
                value = component.get(field)
                if (value is not None):
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not np.isfinite(value)
                        or (value < 0)
                        or ((upper is not None) and (value > upper))
                    ):
                        raise ValueError(f"Record {index} {source}.{field} is invalid.")
                row[output_key] = value
            labels = locations.get(source)
            if (labels is None):
                labels = []
            if (not isinstance(labels, list) or any(
                (not isinstance(label, str) or (label not in CANON)) for label in labels
            )):
                raise ValueError(f"Record {index} {source} locations need canonical labels.")
            row["locations"][source] = labels
        rows.append(row)
    return rows

def main(argv = None):
    parser = argparse.ArgumentParser(
        description = "Reproduce clinical agreement statistics from an external endpoint ledger."
    )
    parser.add_argument("--input", required = True, type = Path, help = "Input JSON ledger.")
    parser.add_argument("--output", required = True, type = Path, help = "Output JSON statistics.")
    parser.add_argument("--expected", type = Path, help = "Optional JSON aggregate to validate.")
    parser.add_argument(
        "--original-prefix", default = "O", help = "Original-output study prefix (default: O)."
    )
    parser.add_argument(
        "--reviewed-prefix", default = "R", help = "Reviewed-register study prefix (default: R)."
    )
    args = parser.parse_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    expected_path = (args.expected.resolve() if (args.expected is not None) else None)
    if (output_path in (input_path, expected_path)):
        raise ValueError("Output must not overwrite the input ledger or expected aggregate.")
    if (output_path.exists() and any(
        output_path.samefile(path)
        for path in (input_path, expected_path)
        if ((path is not None) and path.exists())
    )):
        raise ValueError("Output must not alias an input file.")
    rows = read_rows(input_path, args.original_prefix, args.reviewed_prefix)
    original = [r for r in rows if r["case"].startswith(args.original_prefix)]
    reviewed = [r for r in rows if r["case"].startswith(args.reviewed_prefix)]
    original_ids = list(dict.fromkeys(r["case"] for r in original))
    reviewed_ids = list(dict.fromkeys(r["case"] for r in reviewed))
    specs = [
        ("case1_strict", original, "P", "H", original_ids, True),
        (
            "case1_fully_assessable",
            [r for r in original if r["fully"]],
            "P",
            "H",
            original_ids,
            True,
        ),
        ("triad_P_H", reviewed, "P", "H", reviewed_ids, True),
        ("triad_P_CAG", reviewed, "P", "CAG", reviewed_ids, True),
        ("triad_H_CAG", reviewed, "H", "CAG", reviewed_ids, False),
        ("same_modality_46", rows, "P", "H", (original_ids + reviewed_ids), True),
        (
            "triad_P_CAG_CAG_positive_only",
            [r for r in reviewed if ((r["CAG"] is not None) and (r["CAG"] >= 25))],
            "P",
            "CAG",
            reviewed_ids,
            True,
        ),
    ]
    analyses = {}
    for name, rs, left, right, ids, features in specs:
        result = evaluate(rs, left, right, ids, features = features)
        result["localization"] = localization(rs, left, right)
        add_proportion_cis(result, rs, ids)
        analyses[name] = result
    checked = None
    if (expected_path is not None):
        expected_document = json.loads(expected_path.read_text(encoding = "utf-8"))
        if (not isinstance(expected_document, dict) or ("analyses" not in expected_document)):
            raise ValueError("Expected aggregate must contain an analyses object.")
        checked = numeric_check(analyses, expected_document["analyses"])
        if (checked == 0):
            raise ValueError("Expected aggregate contains no comparable numerical values.")
    out = {
        "schema": "trace23_recomputed_anonymous_v1",
        "analyses": analyses,
        "validation": {
            "expected_reference_supplied": (expected_path is not None),
            "all_numeric_estimates_counts_and_intervals_match": (
                True if (checked is not None) else None
            ),
            "numeric_values_checked": checked,
        },
    }
    serialized = json.dumps(out, indent = 2, allow_nan = False)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    output_path.write_text(serialized, encoding = "utf-8")

if (__name__ == "__main__"):
    main()
