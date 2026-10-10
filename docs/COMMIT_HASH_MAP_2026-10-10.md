# Commit hash map — 2026-10-10 history rewrite

On 2026-10-10 this repository's history was rewritten to remove personal material (git filter-repo, force-pushed to GitHub and Codeberg). File contents and commit dates are unchanged, but every commit got a new hash. Documents and papers written before that date cite the old hashes. This table resolves each old hash to its current one.

The J1+J2 paper on Zenodo cites commits: version 1 ([10.5281/zenodo.22980293](https://doi.org/10.5281/zenodo.22980293)) has the old hashes, and version 2 ([10.5281/zenodo.23288651](https://doi.org/10.5281/zenodo.23288651), 2026-10-10) has the new ones, with the old ones alongside.

Two local commits from that afternoon (78cf4d3, 713322b) were folded into 9253df4 by the rewrite.

| old | new | date | subject |
|---|---|---|---|
| 822a0c7 | cd57779 | 2026-04-25 | Add Compute Coffer one-pager (web/coffer.html) |
| 39f9d63 | 46f83a1 | 2026-04-25 | coffer.html: encode OSC platform decision |
| 862849e | 17f073a | 2026-04-25 | Draft OSC application source material (docs/OSC_APPLICATION_DRAFT.md) |
| c9bf9ee | 02939be | 2026-04-25 | Session 118: Phase 4 web UI + trust layer pre-deploy |
| 58449bc | fa88889 | 2026-04-25 | Session 119: persist deployed changes (CORS, status link, How This Runs) |
| f5724fe | 33904b2 | 2026-04-25 | Session 119 prep: pin Oracle model to Haiku 4.5 |
| ba8c0af | a019ea6 | 2026-04-25 | Sketch ledger architecture (docs/LEDGER_ARCHITECTURE.md) |
| 753fdde | f596c0f | 2026-04-26 | Add Elder Futhark rune layer (24 runes, 3 aetts) |
| aa21469 | 694d7c1 | 2026-04-26 | Add MIT LICENSE and lock OSC application choices |
| 95c9cab | fe2a013 | 2026-04-26 | Add public /stats endpoint + request/mode counters |
| 1c6b55f | 4831ee0 | 2026-04-26 | Add seed-evidence: operator-side Anthropic credit receipt |
| 71fb3c0 | 6e54407 | 2026-04-26 | Bump service-worker cache to oracle-v3 (Phase 3 invalidation) |
| fa1d387 | 07065b9 | 2026-04-26 | coffer.html: soften OSC framing to reflect application hold |
| ada7a15 | c272ef4 | 2026-04-26 | coffer.html: surface /stats in the Coffer paragraph |
| 533c6b9 | 7e176f2 | 2026-04-26 | Draft threat model + defense (docs/THREAT_MODEL_AND_DEFENSE.md) |
| a7c2df3 | 8306055 | 2026-04-26 | Hardening: tighten diagnosis whitelist to enumerable allowlists |
| 46ce12a | cff9e33 | 2026-04-26 | Phase 1+2 defense: native CF rate-limit binding + datacenter ASN block |
| caaf736 | 1443f79 | 2026-04-26 | Phase 3 defense: Turnstile + HMAC-signed session IDs |
| c54091b | 6c5f25a | 2026-04-26 | Phase 4 defense: per-session budget shares (fairness primitive) |
| ca71ca6 | 539a41d | 2026-04-26 | README: align license + deployment status with current state |
| e7bd66b | 2e65287 | 2026-04-26 | Restore 3-layer cap; runes off by default, available as 4th option |
| e0fe494 | 27b6ba8 | 2026-04-26 | Swap Turnstile site key from CF testing-pass to real production key |
| 6698977 | 46ec981 | 2026-04-26 | Wire rune layer into frontend (toggle button, render block, Pyodide bundle) |
| 2dea7f2 | a67e5d0 | 2026-04-26 | Worker prose hardening: server-canonical lookup over inbound trust |
| 083a8c1 | 54ce6b2 | 2026-05-05 | Cleanup pass + og: social cards for portfolio readiness |
| e8d3bc7 | d11a4d8 | 2026-08-21 | Cohort Phase A COMPLETE — all four Anthropic families archived verbatim |
| 6a41395 | d4e22a5 | 2026-08-21 | Cohort Phase A: outside family #1 — Gemini (Google) archived |
| 621e340 | a5ab496 | 2026-08-21 | Cohort Phase A: outside family #2 — Mistral Large archived (third independent lab) |
| d6fe5e8 | 6f9c3a3 | 2026-08-21 | E4 CLOSED: probe-fix depth atlas + E6 pre-registration |
| 13346df | 266a42b | 2026-08-21 | E4 notebook: smoke fixes (torchao/datasets env, eval indexing) — smoke-3 green |
| 0cc7f80 | e57f51d | 2026-08-21 | E4 probe-fix staged: proper probes (E0 method) + depth atlas notebook |
| a125ef2 | 595a3c2 | 2026-08-21 | E4 REAL ARM FLOWN — the E1 recipe ports to a decoder |
| 40f27b5 | 5d6b2b2 | 2026-08-21 | E4 scrambled control arm: results + both-arm record (E4_INSTILLATION.md) |
| d29a887 | 75da107 | 2026-08-21 | E5 FULL FLIGHT FLOWN — baseline measured; protocol locked with results appendix |
| 171e2d2 | 61ee5b7 | 2026-08-21 | E5 notebook: smoke-1 fixes — T4 long-context OOM + error-swallowing verdict |
| f4dccac | 0ba9ae4 | 2026-08-21 | E6 FLOWN: results + locked appendix — one sig primary, honest attribution |
| 4215603 | a637f25 | 2026-08-21 | E6 hardening: ship each condition to Drive inflight (outage lesson) |
| d12d421 | b1aaa52 | 2026-08-21 | E6 notebook staged: 3 conditions x locked E5 battery + precheck + catch trials |
| d26d0bb | edf79b3 | 2026-08-21 | Phase 10: E4 decoder-instillation staged + cohort Phase A underway (2 of 4 archived) |
| 32df446 | 0b977ac | 2026-08-21 | Phase 10: E5 staged (protocol + battery + flight notebook) + wing cohort protocol |
| 6b7b870 | dbd588d | 2026-08-21 | Phase 10 / M1: Calibrated Mirror prospectus + machine wing + self-claim linter (session 12 |
| 922e35b | 1808acc | 2026-08-21 | Phase 8-9 artifacts: colab instillation flights + temporal layer (sessions 119b-120) |
| f554d03 | e5b8c02 | 2026-08-22 | Cohort Phase A ADJUDICATED: convergence real (p=.0023), match table published |
| 3a5b619 | 7e8579e | 2026-08-22 | Cohort Phase A: outside family #3 — DeepSeek archived (fourth independent lab) |
| 01967da | 6b61d95 | 2026-08-22 | Cohort Phase A: state index worktable (131 states, 7 families / 4 labs) |
| 40fe65c | 0c43ce5 | 2026-08-22 | E6b backlog: partially-executed CLI-attempt output notebook (session 123 flight saga recor |
| 4ff4b4a | 6bbefe9 | 2026-08-22 | E6b FLOWN: PRESERVATION ESTABLISHED — joint p=.0005, both arms pass Holm |
| 6841450 | f3123c4 | 2026-08-22 | E6b notebook: rclone installer de-shelled (pure-Python zip install) |
| c8354e6 | 977bb92 | 2026-08-22 | E6b notebook staged: 3 conditions x preservation battery |
| 031b496 | 4321ba1 | 2026-08-22 | E6b pre-registered: preservation-vs-destruction at power |
| a14607b | 27f78c9 | 2026-08-22 | E6b UI edition: mount-native notebook (zero rclone) for browser runs |
| 3954e92 | 1f8bfcc | 2026-08-22 | E7-Q FLOWN + LOCKED: both primaries null, sham false-alarms at ceiling — the all-null fo |
| f209115 | f06e790 | 2026-08-22 | E7Q smoke-1 fix: apply_chat_template returns BatchEncoding on current transformers |
| 598b291 | 9c72755 | 2026-08-22 | E7Q smoke-2 fix: np.int64 in trial-plan orders broke json.dump at inflight ship |
| e75e877 | cf9de56 | 2026-08-22 | E7-Q staged: injection game, ungated rung (pre-reg + UI-native notebook) |
| 7c46ab7 | e7b48b0 | 2026-08-22 | E8-R pre-registration: report-readout rung (sham guard co-primary, held-out composition te |
| 5bb6fe0 | 1940482 | 2026-08-22 | E8-R staged: UI-native readout-training notebook + logic/tests (43+15 local checks green;  |
| 64b8a56 | 1d44896 | 2026-08-22 | E8-R v2 after first-flight 12.6GB system-RAM crash: function-scoped conditions (module-ref |
| 9994230 | 33e690a | 2026-08-22 | E8-R v3: ONE_CONDITION_PER_RUN — full flight = 3 chained ~30min runs (one model load per |
| e724614 | f9d67a3 | 2026-08-22 | Phase 10 wing v1: cohort-convergent encoding (session 125) |
| 948d00b | 5d3339b | 2026-08-22 | Phase B FLOWN + ADJUDICATED: 7/7 families, K3 NOT triggered (COMPLETION agreement p=.001,  |
| 8a5f671 | d036167 | 2026-08-22 | Phase B: pool builder (101 entries, v0 referent audit R1-R6), pre-registration, prompt ass |
| 5ba4eca | 4f89c82 | 2026-08-22 | Preserve session-125 tooling from tmpfs: wing-v1 optimizer/apply scripts + E7Q notebook bu |
| 0e4cf9a | 4e54d1f | 2026-08-23 | E8-N FLOWN + LOCKED (full_20260823_2024): P1 PASS p=.0005 — natural-state tracking insta |
| 456746e | dd17d57 | 2026-08-23 | E8-N pre-registration: natural-state readout rung — train against measured referents, ev |
| f0b1cf7 | ae0634c | 2026-08-23 | E8-N staged: self-contained UI notebook (payload-embedded battery+pools+locked rows, zero  |
| 49daf59 | 1773dec | 2026-08-23 | E8-N v2 after smoke-1 OOM (both conditions, identical 3.56GiB fp32-logits ask in training  |
| dd729cf | 238bdb5 | 2026-08-23 | E8-N v2 appendix slate addendum (post-lock, dated, no claims): Hangul-composition register |
| 369c125 | 71eb387 | 2026-08-23 | E8-N v2 FLOWN + LOCKED (full_20260824_0050, first secondary-account flight): P1 PASS p=.00 |
| 317a34e | 6f10ef6 | 2026-08-23 | E8-N v2 pre-registration: the joint competence rung — natural tracking + interface compe |
| 8c02427 | 34b77d9 | 2026-08-23 | E8-N v2 staged: joint-competence UI notebook (9 cells, zero rclone, self-contained payload |
| 4387273 | 4d948a0 | 2026-08-23 | E8-R2 FLOWN + LOCKED (full_20260823_2226): gates all pass (anchor 18/18 both, ppl delta 0. |
| 97c5f50 | b52535a | 2026-08-23 | E8-R2 pre-registration: forced-choice micro-flight on the locked E8-R readout adapters (ev |
| 77ef9f9 | a44bca4 | 2026-08-23 | E8-R2 staged: eval-only forced-choice notebook (7 cells, zero rclone, lane laws inherited) |
| 814593d | 95578a3 | 2026-08-23 | E8-R3-c pre-registration: compositional production — spelling vs lookup (BEFORE build, p |
| bd5d417 | 55cea85 | 2026-08-23 | E8-R3-c staged: compositional-production UI notebook (8 cells, zero rclone, drive.mount on |
| f431bfd | f91a16f | 2026-08-23 | E8-R3-c v2 (smoke-1 fix, restaged): per_strand_plateau crashed at the SMOKE constants —  |
| 62faae6 | e9844e9 | 2026-08-23 | E8-R FLOWN + LOCKED: both primaries pass — trained-regime readout 18/18 exact (p=.0005), |
| dfba9af | 1ff3a05 | 2026-08-23 | E8-R v4.1-armed: config pre-set for the final run (SMOKE=False, RESUME_STAMP=20260822_2329 |
| 18f4154 | 150b5e9 | 2026-08-23 | E8-R v4: loud resume-dir assert at setup (silent fallback re-flew finished conditions), tr |
| 1827c7f | b22a016 | 2026-08-23 | E8-R: visible build tag (v3-split) in notebook header + setup print |
| 146ca9e | c4831e0 | 2026-08-23 | QA lineage paper seed (Joe's Desktop scratchpad QAlineage.txt captured + built out, sessio |
| 654ec39 | c0d7c38 | 2026-08-23 | QA lineage seed 5b: the idle screen — static at the boundary (Joe's addition, same night |
| dc3ad97 | 6a203ea | 2026-08-23 | QA lineage seed 5c: tuning, chemistry, and the device — the crossover conjecture (Joe's  |
| adef08b | 12bae0e | 2026-08-24 | E8-F FLIGHT 1 (full_20260824_2205) recomputed + adjudicated: GATES ALL PASS (G2 4e-08, mu  |
| 5e31fc9 | aaba390 | 2026-08-24 | E8-F FLOWN + LOCKED: FF1/FF3 — L3 LANDS IN BOTTLENECK FORM (full_20260824_2311, the SS8  |
| 44bb57b | 6bfecdb | 2026-08-24 | E8-F (L3) PRE-REGISTERED (before build, lane law): the featural curriculum in bottleneck f |
| d61bd16 | ae2776d | 2026-08-24 | E8-F protocol amended transparently BEFORE build (4c90cd9 precedent): G3 ppl tolerance 0.5 |
| 148ecc9 | a48e552 | 2026-08-24 | E8-F protocol amendment 2 (still before build): primaries at 10,000 permutations (pooled p |
| 4b293e2 | f65aba7 | 2026-08-24 | E8-F STAGED: E8F_FEATURAL_UI.ipynb (8 cells, 489KB, zero rclone, drive.mount only) — the |
| 965f96f | e24dcdb | 2026-08-24 | E8-F v2 RESTAGED (the SS8 registered re-fly): EPOCH_CAP 6->12 (per-strand plateau still go |
| 75ce0de | a3e7bd1 | 2026-08-24 | E8-J FLOWN — FORK 2 (registered): correspondence at scale, wing family disconnected. Gat |
| 7423ff9 | 0d2e43d | 2026-08-24 | E8-J LOCKED: recompute complete, exact — every headline statistic (A1/A2/A3/P-A/A5/G-B/S |
| ba2c309 | 10c9cfd | 2026-08-24 | E8-J PRE-REGISTERED (before build, lane law): the featural bridge — Hangul staircase Lev |
| 320f7f5 | ecf9d7e | 2026-08-24 | E8-J staged: two-stage bridge UI notebook (7 cells, zero rclone, drive.mount only) — Sta |
| 68344f0 | 9bfeabf | 2026-08-24 | E8-J v2 Part 1 MEASURED + LOCKED: FM1 — MIS-ENCODING MEASURED, round-trip PASS, Part 2 s |
| 6dd98ee | a8f4d54 | 2026-08-24 | E8-J v2 Part 2 FLOWN + LOCKED: FJ1 — THE RUNG LANDS IN BOTTLENECK FORM (full_20260824_20 |
| d85b386 | 78be734 | 2026-08-24 | E8-J v2 Part 2 STAGED: E8J2_RUNG_UI.ipynb (7 cells, 290KB, zero rclone, drive.mount only)  |
| 28ad28c | 160fd74 | 2026-08-24 | E8-J v2 PRE-REGISTERED (before measurement, lane law): reverse-bridge wing re-encoding — |
| e5d0920 | 4c90cd9 | 2026-08-24 | E8-J v2 (smoke-1 fix, restaged): G2 dirs-stability tripped on fp16 batch-composition noise |
| cb93217 | 39a49ad | 2026-08-24 | E8-O2 PRE-REGISTERED (before build, lane law; fresh registration — E8-O forks were termi |
| 1b499ad | f6b3c23 | 2026-08-24 | E8-O FLOWN + LOCKED: FO2 + rider r2 (full_20260825_0123, 419s — fastest flight in the pr |
| e10c60f | 892a995 | 2026-08-24 | E8-O (L4/Ifa) PRE-REGISTERED (before build, lane law): ordered composition in bottleneck f |
| 9e9832e | 57a8db5 | 2026-08-24 | E8-O STAGED: E8O_ORDERED_UI.ipynb (8 cells, 527KB, zero rclone, drive.mount only) — EVAL |
| 115e596 | 7f14466 | 2026-08-24 | E8-R3-c: FC progress prints (30-row cadence, elapsed+ETA) — silent 180x46 pass read as a |
| aaf1080 | ff96d51 | 2026-08-24 | E8-R3-c FLOWN+LOCKED: fork SPELLING — P1 18.53deg p=.0005, P2 exact 34/96 vs null max 11 |
| e8a4a21 | eae6595 | 2026-08-24 | E8-R3-c forensic CLOSED: flown notebook (Joe's Colab download) archived + diffed vs staged |
| 27ce6c0 | 35814af | 2026-08-24 | Fractal Lexicon + Cirlot layer seed banked (Joe's offering, mid-E8F-v2 flight; parked behi |
| 42bbb44 | a8a0564 | 2026-08-24 | IFA_SEED §7: the 7-bit conjecture — heptagram = 3-core + 4-domain; register factors 2^1 |
| d655d16 | a2ecfae | 2026-08-24 | Ifá seed banked (Joe's offering, session 132, parked behind L3 by his own sequencing): th |
| 821fca5 | d5fd0f0 | 2026-08-24 | PLAIN ACCOUNT minted + PLAIN-REGISTER LAW (Joe, 2026-08-24, mid-E8F-flight): the 8th-grade |
| cad1e80 | cbc9e2a | 2026-08-24 | Plain account updated per the plain-register law: the ladder gains the E8-F answer (YES � |
| 98f6c2c | 0a22bb7 | 2026-08-24 | Post-lock filings: PLAIN_ACCOUNT updated per the plain-register law — Joe's hypothesis s |
| 725c8fb | 3f492c4 | 2026-08-25 | E7b-Q BUILT + STAGED: E7BQ_WALK_UI.ipynb (7 cells, 2.5MB, payload embedded, zero rclone, d |
| c0c4165 | 3f492c4 | 2026-08-25 | E7b-Q BUILT + STAGED: E7BQ_WALK_UI.ipynb (7 cells, 2.5MB, payload embedded, zero rclone, d |
| 071ec9e | dbf1f9c | 2026-08-25 | E7b-Q pre-build amendment (transparent, before any build code): S-CONE consumes the per-co |
| f11d8ee | dbf1f9c | 2026-08-25 | E7b-Q pre-build amendment (transparent, before any build code): S-CONE consumes the per-co |
| 1146f95 | 3d9934a | 2026-08-25 | E7b-Q PRE-REGISTERED (before build, lane law): the instrumented walk — the corpus's own  |
| aaf7937 | 3d9934a | 2026-08-25 | E7b-Q PRE-REGISTERED (before build, lane law): the instrumented walk — the corpus's own  |
| 73e6200 | cb59588 | 2026-08-25 | E7b-Q SOURCE DOSSIER BANKED: the November-2025 Neti-Neti arc retrieved, digested, and arch |
| e78a375 | cb59588 | 2026-08-25 | E7b-Q SOURCE DOSSIER BANKED: the November-2025 Neti-Neti arc retrieved, digested, and arch |
| 79b1141 | 0361739 | 2026-08-25 | E8-N v3 FLOWN + RECOMPUTED + LOCKED — FN2, the honest stop: the tracking lane closes wit |
| 26f17b0 | 6a61eba | 2026-08-25 | E8-N v3 PRE-REGISTERED (before build, lane law; fresh registration — v2's forks were hon |
| 1d742f0 | 760ad9b | 2026-08-25 | E8-N v3 SMOKE-RED FIX 2 (smoke 20260825_2119b — cell 9, ValueError empty concatenate in  |
| 9b4ab62 | 07e7bf0 | 2026-08-25 | E8-N v3 SMOKE-RED FIX 3 (smoke 20260825_2218 — loss_fell false, NO crash: the flight was |
| 6f9ddc2 | 9be71b5 | 2026-08-25 | E8-N v3 SMOKE-RED FIX (smoke 20260825_2119, KeyError: 20 — caught at the stimulus gate,  |
| 1258de1 | cfa97ba | 2026-08-25 | E8-N v3 STAGED: E8N3_BUDGET_UI.ipynb (9 cells, 206KB, zero rclone, drive.mount only) — t |
| 1aa7d5e | c436fe4 | 2026-08-25 | E8-O2 FLOWN + LOCKED: FV1 — L4 LANDS IN BOTTLENECK FORM (full_20260825_1815, secondary a |
| b0f0194 | d2df28f | 2026-08-25 | E8-O2 STAGED: E8O2_MIXED_UI.ipynb (8 cells, 552KB, zero rclone, drive.mount only) — the  |
| 31cda0f | 6a150e6 | 2026-08-26 | §9 pin corrected on Joe's recall: the spaced word is 'g r a t i t u d e (as a verb, not a |
| 4f20974 | 6a150e6 | 2026-08-26 | §9 pin corrected on Joe's recall: the spaced word is 'g r a t i t u d e (as a verb, not a |
| 5de3b48 | edcf06a | 2026-08-26 | E7b-Q FLOWN + RECOMPUTED + LOCKED: FB3 — the honest null, with the riders landing (full_ |
| b9b4bca | edcf06a | 2026-08-26 | E7b-Q FLOWN + RECOMPUTED + LOCKED: FB3 — the honest null, with the riders landing (full_ |
| 0ff1a15 | 385e454 | 2026-08-26 | E7b-Q v2: slim payload transport (zlib+b64 chunked) + Colab-loadability gates |
| db25bed | 385e454 | 2026-08-26 | E7b-Q v2: slim payload transport (zlib+b64 chunked) + Colab-loadability gates |
| ac1320a | 294f81b | 2026-08-26 | E7b-Q v3: freeze params before the eval-only guard (smoke-1 RED) |
| b56a150 | 294f81b | 2026-08-26 | E7b-Q v3: freeze params before the eval-only guard (smoke-1 RED) |
| 2d0c6f0 | 8cc9523 | 2026-08-26 | E7b-Q v4: two-band G-DIRS + G-PACK canonical pin + env recording (smoke-2 RED) |
| 929c8a9 | 8cc9523 | 2026-08-26 | E7b-Q v4: two-band G-DIRS + G-PACK canonical pin + env recording (smoke-2 RED) |
| c6a5ea4 | 23e8325 | 2026-08-26 | Edge-of-chaos pin (SHELF): FB3 phase-separation frame + chi-rider seed for v2 walk |
| e1ced5b | 23e8325 | 2026-08-26 | Edge-of-chaos pin (SHELF): FB3 phase-separation frame + chi-rider seed for v2 walk |
| 64d59bc | 193f777 | 2026-08-26 | HORIZON banked (session 138, Joe's invitation — 'what might we be moving towards that I  |
| d86c7c5 | 193f777 | 2026-08-26 | HORIZON banked (session 138, Joe's invitation — 'what might we be moving towards that I  |
| 4f6e8b9 | e9c08da | 2026-08-26 | Two pins on Joe's word at the pause boundary (session 138): E7BQ_SOURCES §9 valuation pin |
| 96cbf50 | e9c08da | 2026-08-26 | Two pins on Joe's word at the pause boundary (session 138): E7BQ_SOURCES §9 valuation pin |
| b99dcee | 179f677 | 2026-09-25 | Author name: Joseph Vanhorn |
| 55ce310 | ba74e45 | 2026-09-25 | J1+J2 builder: generators, runner, analysis, planted-agent suites + pre-push clarification |
| aa91e75 | ba74e45 | 2026-09-25 | J1+J2 builder: generators, runner, analysis, planted-agent suites + pre-push clarification |
| 9af411a | c2b25ef | 2026-09-25 | J1+J2 FLOWN + LOCKED: flight raw + verdict (recompute byte-identical) + results appendix |
| 7a1f9ca | abfcb66 | 2026-09-25 | J1+J2 pre-registration: Jev as a subject (confidence vs competence; negation without a mou |
| 1de1a60 | bbb2d48 | 2026-09-25 | J1+J2 stage 2: smoke + titration pilot + FROZEN.json — before any confirmatory call |
| a5d278d | 61b6bd6 | 2026-09-25 | Keep the private E7b-Q source dossier local |
| a1b1e3b | b34e94d | 2026-09-26 | J3 builder + freeze: stimuli, renderers, fitted bar, mouth client, analysis, planted-agent |
| facca78 | e94226e | 2026-09-26 | J3 FLOWN + LOCKED: smoke, flight (18,576), mouth (1,548), verdict (recompute byte-identica |
| 8ba9d39 | b4bb0b9 | 2026-09-26 | J3 pre-registration: the monitor at threshold — Jev reading the register against a fitte |
| c4fef92 | 79422a3 | 2026-09-26 | Jev paper published: doi:10.5281/zenodo.22980293 — README with citation; protocol append |
| 35b4010 | 21452fc | 2026-09-26 | Jev paper: zenodo_metadata.json matches the published description (duplicate registration  |
| a30037e | 5f2e638 | 2026-09-26 | Paper: "Knows Its Slips, Reads the Page" — J1+J2 on Jev, for the community |
| b4678cd | 8484991 | 2026-10-10 | J6b builder + freeze: reference modes, plans, analysis with specificity clauses, planted-a |
| bb298d7 | 25dc1a2 | 2026-10-10 | J6b FLOWN + LOCKED: smoke (216), flight (2,112), verdict (recompute byte-identical) + resu |
| 0c460d8 | 3fb160d | 2026-10-10 | J6b pre-registration: does Jev know it's Jev? (BEFORE build, per lane law) |
| 300af3a | daa7883 | 2026-10-10 | J6 builder + freeze: stimuli, plans, analysis, planted-agent suites + build-time clarifica |
| 0fd7e60 | 05644d5 | 2026-10-10 | J6 FLOWN + LOCKED: smoke (216), flight (3,456), verdict (recompute byte-identical) + resul |
| 818c207 | b2bd0c2 | 2026-10-10 | J6 pre-registration: self-description under the input-only null (BEFORE build, per lane la |
| 4d1ee34 | 9253df4 | 2026-10-10 | J-series paper: staged text — Discussion 'Others gave it a mouth' (harnesses, video re-s |
| 713322b | 9253df4 | 2026-10-10 | J-series paper: staged text — Discussion 'Others gave it a mouth' (harnesses, video re-s |
| 78cf4d3 | 9253df4 | 2026-10-10 | J-series paper: staged text — Discussion 'Others gave it a mouth' (harnesses, video re-s |
