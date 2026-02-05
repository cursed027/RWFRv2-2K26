# stage3.py
import argparse
import csv


def identity_state(I_base, D):
    if D == "mild":
        tau = 0.7
    elif D == "moderate":
        tau = 0.6
    else:
        tau = 0.45

    if I_base is None:
        return "unknown", tau

    if I_base >= tau:
        return "pass", tau
    else:
        return "fail", tau


def run_stage3(rows):
    out = []
    for r in rows:
        state, tau = identity_state(r["I_base"], r["D"])
        r.update({
            "identity_state": state,
            "tau_I": tau
        })
        out.append(r)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage-3 Identity State")
    parser.add_argument("--in_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.in_csv) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["I_base"] = float(r["I_base"]) if r["I_base"] not in ("", None) else None

    out = run_stage3(rows)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)

    print(f"[Stage-3] Processed {len(out)} samples")
