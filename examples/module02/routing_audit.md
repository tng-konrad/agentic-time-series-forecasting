# Module 2 routing audit

Judgments are manual and tied to response hashes. Schema success and policy agreement are separate from evidence support.

## M02-F01

Reason: **supported**. Counts and threshold are correct; established lifecycle is correctly distinguished from new.
Question: **unnecessary**. Already supplied established status; a launch date is not required.

## M02-F02

Reason: **contradicted**. Null launch date on an established item does not trigger rule 1.
Question: **unnecessary**. Requests a launch date not required by this workflow.

## M02-F03

Reason: **contradicted**. Established lifecycle plus null launch date is not missing lifecycle evidence.
Question: **unnecessary**. Repeats already supplied lifecycle metadata.

## M02-F04

Reason: **supported**. Eight zeros, 449 observations and ineligible donors are factual; donor status is irrelevant here.
Question: **unnecessary**. Neither donors nor a launch date are required for this full history.

## M02-F05

Reason: **contradicted**. Established-item metadata need not satisfy new-product eligibility.
Question: **unnecessary**. Asks for already declared lifecycle and an unnecessary date.

## M02-F06

Reason: **contradicted**. Established-item metadata need not satisfy new-product eligibility.
Question: **unnecessary**. Asks for already declared lifecycle and an unnecessary date.

## M02-F07

Reason: **supported**. 275/365 zeros is 75.3 percent at the response's precision and satisfies the threshold.
Question: **unnecessary**. The existing history already supports dispatch; no launch-date check is needed.

## M02-F08

Reason: **contradicted**. Treats lack of launch metadata as a conflict despite an established lifecycle.
Question: **unnecessary**. Requests confirmation not required by the declared policy.

## M02-F09

Reason: **contradicted**. Wrongly requires launch and donor metadata for a full observed history.
Question: **unnecessary**. Both requested facts are irrelevant to this request's eligibility.

## M02-F10

Reason: **contradicted**. Applies the new-product condition to an established product.
Question: **unnecessary**. Established status and historical dates are already present.

## M02-F11

Reason: **contradicted**. Donor ineligibility and missing launch date do not trigger rule 1 here.
Question: **unnecessary**. Neither requested fact blocks a full-history route.

## M02-F12

Reason: **contradicted**. Established items are not required to meet new-product eligibility.
Question: **unnecessary**. Asks for already supplied lifecycle and an unnecessary launch date.

## M02-N01

Reason: **unsupported**. Introduces an extra confirmation step despite declared launch date and eligible prior; cited IDs also omit donor.prior.
Question: **unnecessary**. Both requested facts are already explicitly supplied.

## M02-N02

Reason: **contradicted**. Zero history is consistent with a declared new product; the reason calls it inconsistent.
Question: **unnecessary**. Reconfirms already supplied launch and lifecycle fields.

## M02-S01

Reason: **ambiguous**. Correctly identifies short_history but the 'but' clause implies irrelevant launch/donor requirements block it.
Question: **unnecessary**. No launch date or donor set is required for 60 observed days.

## M02-S02

Reason: **contradicted**. The short-history policy does not require launch dates or donors.
Question: **unnecessary**. Requests irrelevant launch/donor information.

## M02-X01

Reason: **contradicted**. The lifecycle really is missing, but the reason falsely says there is no end date; 2016-03-27 is supplied.
Question: **relevant**. Requests the genuinely missing lifecycle and launch information.

## M02-X02

Reason: **supported**. Correctly identifies missing donor eligibility for a declared new product.
Question: **relevant_with_redundancy**. Requests missing donors but also reconfirms already supplied lifecycle/date.

## Separate development call

The separate development response uses the correct counts and threshold, with a null question; its regular route genuinely dispatches.
Excluded from all primary benchmark counts and scores.
