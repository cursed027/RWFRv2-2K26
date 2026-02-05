# stage4.py
import argparse
import csv


def route(identity_state, D):
    # Case A
    if identity_state == "pass" and D == "mild":
        return "A", ["diffbir_0.35", "codeformer_0.7"]
    
    if identity_state == "pass" and D == "moderate":
        return "A2", ["diffbir_0.25", "codeformer_0.6"]

    # Case B
    if identity_state == "pass" and D == "severe":
        return "B", ["diffbir_0.25"]

    # Case C
    if identity_state == "fail" and D == "mild":
        return "C", ["use_anchor", "diffbir_0.1"]

    # Case D1: UNKNOWN identity (key fix)
    if identity_state == "unknown" and D in ("mild", "moderate"):
        return "D1", ["diffbir_0.15"]

    # Case D2: UNKNOWN + severe
    if identity_state == "unknown" and D == "severe":
        return "D2", ["diffbir_0.1"]

    # True worst case
    return "E", ["use_anchor"]


def run_stage4(rows):
    out = []
    for r in rows:
        route_id, actions = route(r["identity_state"], r["D"])
        r.update({
            "route": route_id,
            "actions": ";".join(actions)
        })
        out.append(r)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage-4 Routing")
    parser.add_argument("--in_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.in_csv) as f:
        rows = list(csv.DictReader(f))

    out = run_stage4(rows)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)

    print(f"[Stage-4] Routed {len(out)} samples")
