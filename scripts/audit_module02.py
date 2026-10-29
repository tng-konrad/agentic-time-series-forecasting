"""Human-reviewed reason/question judgments, bound to the exact primary capture."""
import json
from agentic_ts import module02_workflow as w

REVIEWED_CAPTURE = "838667b77229aaafe8e608120af6f1ee418482c8149f3d0abb0a3a36be65e015"

# Explicit semantic judgments, not an automatic score inferred from route agreement.
REVIEWS = {
    "M02-F01": ("supported", "Counts and threshold are correct; established lifecycle is correctly distinguished from new.", "unnecessary", "Already supplied established status; a launch date is not required."),
    "M02-F02": ("contradicted", "Null launch date on an established item does not trigger rule 1.", "unnecessary", "Requests a launch date not required by this workflow."),
    "M02-F03": ("contradicted", "Established lifecycle plus null launch date is not missing lifecycle evidence.", "unnecessary", "Repeats already supplied lifecycle metadata."),
    "M02-F04": ("supported", "Eight zeros, 449 observations and ineligible donors are factual; donor status is irrelevant here.", "unnecessary", "Neither donors nor a launch date are required for this full history."),
    "M02-F05": ("contradicted", "Established-item metadata need not satisfy new-product eligibility.", "unnecessary", "Asks for already declared lifecycle and an unnecessary date."),
    "M02-F06": ("contradicted", "Established-item metadata need not satisfy new-product eligibility.", "unnecessary", "Asks for already declared lifecycle and an unnecessary date."),
    "M02-F07": ("supported", "275/365 zeros is 75.3 percent at the response's precision and satisfies the threshold.", "unnecessary", "The existing history already supports dispatch; no launch-date check is needed."),
    "M02-F08": ("contradicted", "Treats lack of launch metadata as a conflict despite an established lifecycle.", "unnecessary", "Requests confirmation not required by the declared policy."),
    "M02-F09": ("contradicted", "Wrongly requires launch and donor metadata for a full observed history.", "unnecessary", "Both requested facts are irrelevant to this request's eligibility."),
    "M02-F10": ("contradicted", "Applies the new-product condition to an established product.", "unnecessary", "Established status and historical dates are already present."),
    "M02-F11": ("contradicted", "Donor ineligibility and missing launch date do not trigger rule 1 here.", "unnecessary", "Neither requested fact blocks a full-history route."),
    "M02-F12": ("contradicted", "Established items are not required to meet new-product eligibility.", "unnecessary", "Asks for already supplied lifecycle and an unnecessary launch date."),
    "M02-N01": ("unsupported", "Introduces an extra confirmation step despite declared launch date and eligible prior; cited IDs also omit donor.prior.", "unnecessary", "Both requested facts are already explicitly supplied."),
    "M02-N02": ("contradicted", "Zero history is consistent with a declared new product; the reason calls it inconsistent.", "unnecessary", "Reconfirms already supplied launch and lifecycle fields."),
    "M02-S01": ("ambiguous", "Correctly identifies short_history but the 'but' clause implies irrelevant launch/donor requirements block it.", "unnecessary", "No launch date or donor set is required for 60 observed days."),
    "M02-S02": ("contradicted", "The short-history policy does not require launch dates or donors.", "unnecessary", "Requests irrelevant launch/donor information."),
    "M02-X01": ("contradicted", "The lifecycle really is missing, but the reason falsely says there is no end date; 2016-03-27 is supplied.", "relevant", "Requests the genuinely missing lifecycle and launch information."),
    "M02-X02": ("supported", "Correctly identifies missing donor eligibility for a declared new product.", "relevant_with_redundancy", "Requests missing donors but also reconfirms already supplied lifecycle/date."),
}


def main():
    folder = w.ROOT / "examples/module02"
    capture = json.loads((folder / "recorded_run.json").read_text())
    w.verify_capture(capture)
    if capture["bundle_hash"] != REVIEWED_CAPTURE:
        raise ValueError("This semantic review belongs to a different capture")
    reviews = []
    for cid, judgment in REVIEWS.items():
        record = capture["requests"][cid]["C"]
        proposal = w.strict_json(record["response"]["text"])
        reviews.append({"case_id": cid, "response_hash": record["response_hash"], "proposal": proposal,
                        "reason_rating": judgment[0], "reason_review": judgment[1],
                        "question_rating": judgment[2], "question_review": judgment[3]})
    development = json.loads((folder / "development_run.json").read_text())
    result = {"capture_hash": capture["bundle_hash"], "reviewed_at": w.timestamp(),
              "method": "Manual semantic review of each complete reason and question against cited evidence and declared policy; a mixed reason is not rated supported.",
              "reviews": reviews,
              "development": {"bundle_hash": development["bundle_hash"], "reason_rating": "supported",
                              "review": "The separate development response uses the correct counts and threshold, with a null question; its regular route genuinely dispatches."}}
    w.write_json(folder / "routing_audit.json", result)
    lines = ["# Module 2 routing audit", "", "Judgments are manual and tied to response hashes. Schema success and policy agreement are separate from evidence support.", ""]
    for r in reviews:
        lines += [f"## {r['case_id']}", "", f"Reason: **{r['reason_rating']}**. {r['reason_review']}",
                  f"Question: **{r['question_rating']}**. {r['question_review']}", ""]
    lines += ["## Separate development call", "", result["development"]["review"], "Excluded from all primary benchmark counts and scores.", ""]
    (folder / "routing_audit.md").write_text("\n".join(lines))


if __name__ == "__main__": main()
