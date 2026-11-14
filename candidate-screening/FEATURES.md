# Historical candidate-screening features

A synchronization peak is only a possible FT8 signal: noise and interference can
also produce plausible matches. These features give the classifier several views
of that candidate before expensive LDPC/OSD decoding: agreement with the known
sync pattern, support elsewhere in the message, and evidence for competing tones.

## Signal estimates

Columns: `snr`, `abs_dt`, `freq`.

`snr` describes estimated strength relative to noise, `abs_dt` gives the magnitude
of the candidate's timing offset in seconds, and `freq` locates it in the audio
passband. A weak real signal may have only modest sync evidence, so these inputs
let the classifier interpret that
evidence in context rather than treating weakness alone as evidence of noise.

## Sync counts and quality

Columns: `nsync`, `nsync2`, `sync_quality`.

`sync_quality` measures background-normalized agreement with the known sync
pattern during the coarse time/frequency search, making search strength comparable
across recordings; after fine alignment, `nsync` counts expected-tone wins and
`nsync2` counts placements exactly second to the strongest tone. Together they
connect the search evidence used to select candidates with refined symbol-level
agreement, retaining evidence from weak signals whose expected tones repeatedly
finish as runners-up under noise or interference.

## Per-block scores and ratios

Columns: `nsync_score`, `nsync_score_a` through `nsync_score_c`,
`nsync_score_wild`, `score_ratio`, `score_ratio_a` through `score_ratio_c`.

The `nsync_score` family counts sync symbols with expected-tone support relative
to the competing tones' average energy, the `score_ratio` family measures that
contrast among the supported symbols, and `nsync_score_wild` pools evidence for
each expected tone across observable sync blocks. The per-block versions preserve
where support occurs, while pooling can reveal a recurring pattern that is weak
in individual blocks, helping distinguish distributed signal evidence from a
local coincidence or disruption.

## CQ-shape score

Column: `rscq`.

`rscq` measures agreement with the data-tone pattern associated with CQ for
candidates considered possible CQ calls, and is inactive for other candidates.
It supplies evidence beyond synchronization: a noise peak that resembles the
sync pattern need not also fit the message structure of a CQ call.

## Relative timing and frequency

Columns: `rrxdt`, `dfqso`.

`rrxdt` is the candidate's timing offset from the nominal frame start, helping the
model distinguish a weak sync block from one missing because the recording does
not contain that part of an early or late signal. `dfqso` is the signed frequency
offset from the receiver's QSO-frequency setting; the historical offline captures
used a fixed default, so it provides a re-centered frequency coordinate alongside
the sync evidence.

## Timing and frequency curvature

Columns: `freq_curvature`, `timing_curvature`.

These features describe the sharpness of a smoothed fine-sync peak around the
refined frequency and time estimates, with more negative curvature indicating a
sharper peak and near-zero curvature indicating a flatter one. A distinct peak
can support a well-localized signal candidate, while broad or ambiguous peaks
can reflect noise or interference.

## Expected-tone rank counts

Columns: `nsmax_rank_1` through `nsmax_rank_8`.

These counts describe how often the expected Costas tone occupies each strength
rank at the sync positions, retaining the distribution beyond wins and
runner-up placements. A weak real signal can repeatedly put the expected tone
near the top without making it the strongest, whereas noise has no reason to
favor the expected tone across the known pattern.

## Alternative-tone indicator

Column: `alt_tone_energy_skip`.

`alt_tone_energy_skip` summarizes `nsync` and the expected-tone rank counts as a
boolean indication that the observed energy favors competing tones, adding no
independent measurement. It presents the classifier with an existing warning
pattern: energy is present, but its distribution may fit interference better
than the candidate's assumed FT8 alignment.

The [historical result](README.md#historical-result) supports the usefulness of
ranking candidates with the combined feature set.
