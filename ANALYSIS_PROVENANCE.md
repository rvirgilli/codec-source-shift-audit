# Analysis provenance and decision chronology

This document distinguishes prospectively repository-recorded decisions from
retrospective and revision-stage analyses. It supersedes the broader claims in the
older `PREREGISTRATION.md`.

## Repository-backed chronology

All times below are the commit timestamps in this repository (UTC-03:00):

| Commit | Time | Event | Status for v9 |
|---|---|---|---|
| `370ade6` | 2026-07-02 00:29 | Release containing the targeted 15-seed MLAAD/TORTOISE outputs | Run predates any public rule file; retrospective |
| `9c708c4` | 2026-07-02 03:37 | ASVspoof5 second-instrument sweep and outcome | Run predates any public rule file; retrospective |
| `0017716` | 2026-07-02 11:31 | First repository appearance of `PREREGISTRATION.md` | Does not establish prior timing for the two earlier runs |
| `0d1b7d0` | 2026-07-02 12:02 | Rule committing to a uniform nine-fold MLAAD extension | Prospectively repository-recorded |
| `f419e53` | 2026-07-02 17:16 | Uniform nine-fold MLAAD extension outputs | After the public rule commit |

## V9 claim status

The following are revision-stage robustness analyses and were **not preregistered**:

- the CodecFake+ crossed seed-and-item bootstrap;
- the eight-fold Bonferroni and max-T family analyses;
- fixed-source macro and leave-source-out sensitivity analyses;
- simultaneous max-T inference over the 28 source pairs;
- tie-correct Mann--Whitney AUROC validation; and
- the crossed reanalysis of the ASVspoof5 and MLAAD external families.

These analyses are reported completely for their declared families, including null and
negative results, but their timing should not be presented as protection against
outcome selection.

The uniform nine-fold MLAAD extension is the only analysis in this repository with a
public rule commit that predates its run. V9 treats the external MLAAD family as a scope
check, not as independent confirmation: its crossed global test is unresolved
(`p=0.520`) and no fold survives familywise correction.

## Superseded claims

The old `PREREGISTRATION.md` called the targeted TORTOISE and ASVspoof5 analyses
pre-registered and reported older seed-focused intervals. The repository history does
not substantiate those timing claims, and v9 does not use those intervals. The
authoritative v9 values are in:

- `outputs/crossed_global_analysis.json`;
- `outputs/rank_crossed_audit_200k.json`; and
- `outputs/external_crossed_effects_200k.json`.
