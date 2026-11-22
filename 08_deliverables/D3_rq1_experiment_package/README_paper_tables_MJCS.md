# Reproducing the result tables of the MJCS manuscript

Manuscript: *Cost-Aware Requirements Triage from Vietnamese App Reviews: Comparing Fine-Tuned Models and LLM Routing*,
revision v2.1 (21 November 2026), prepared for the Malaysian Journal of Computer Science. Revision v2.1 changes only the
reference list and the sentences that cite it; every table and figure is the same as in v2.0.

This guide regenerates every result table of the manuscript (Tables 4, 6–16 and A2–A4) and Figures 3 and 4 from the predictions
saved in this repository. No model is retrained and no API is called.

## 1. Inputs

| Input | Location | In the repository |
|---|---|---|
| Saved predictions and class probabilities | `05_experiments/rq1_classification/gd10_pred_*.csv`, `paper_pred_tfidf_*.csv`, `paper_pred_neural_*.csv`, `paper_flat_llm_*.csv`, `paper_tfidf_val_predictions.csv` | yes |
| LLM call logs (token counts, system fingerprints, raw outputs; no review text) | `05_experiments/rq1_classification/*_calls.jsonl` | yes |
| Frozen configurations and cost assumptions | `03_configs/frozen/` (including `paper_tfidf_config.yaml`, `paper_neural_config.yaml`, `paper_cost_model.yaml`) | yes |
| Selections made on validation data, saved with SHA-256 before test labels were read | `05_experiments/rq1_classification/paper_step4_selection.json` (Round 3), `paper_step5_selection.json` (Round 4) | yes |
| Prompts and few-shot examples | `04_prompts/classification_fewshot/` (system prompts verbatim; examples as review IDs, labels and SHA-256 of the text); hashes in `04_prompts/prompts_registry.md` | yes, without example texts |
| Results of earlier rounds used by the consistency checks | `05_experiments/rq1_classification/paper_step2_*`, `paper_step4_*`, `paper_step5_*`, `paper_cost_*` | yes |
| Split identifiers | `01_data/manifests/splits_manifest_T0_2026-10-06.json`, `validation_rebalance_v1.8_T0_2026-10-06.json` | yes |
| Labelled split files with review texts (names and e-mail addresses masked) | `01_data/processed/{shopee_train, shopee_validation, shopee_holdout, shopee_focused_feature_ux, lazada_external_test}.csv` | yes (private repository) |
| Annotation sheets, adjudicated labels and label provenance | `01_data/gold/`, `07_human_eval/` | yes (private repository) |
| Crawled raw data (shop replies still contain unmasked names) | `01_data/raw/`, `01_data/interim/` | yes (private repository); not needed to regenerate the tables |

Environment: Python 3.13, `pip install -r requirements.txt`, plus `pip install jupyter matplotlib` for running the notebooks and
drawing the figures (SciPy is installed with scikit-learn). A laptop CPU is enough; the table notebook runs in a few minutes.

## 2. Steps

1. Check that the five labelled split files are in `01_data/processed/` (they are part of the private repository).
2. Optional: re-run `notebooks/25_paper_fair_controls.ipynb` to regenerate the Round 4 files `paper_step5_*`; otherwise the saved
   files are used. The notebook verifies the SHA-256 of `paper_step5_selection.json` before it reads any test label.
3. Run the table notebook from the `notebooks/` folder:

   ```bash
   cd notebooks
   jupyter nbconvert --to notebook --execute --inplace 26_paper_tables.ipynb
   ```

4. Open `05_experiments/rq1_classification/paper_tables_gates.csv`. Each of the 42 checks compares regenerated values (macro-F1,
   confidence intervals, paired differences, McNemar p-values, break-even volumes and cost decisions) with the files of the earlier
   rounds; the notebook stops if any deviation exceeds 1e-9.

Outputs, all in `05_experiments/rq1_classification/`:

- `paper_tables_*.csv`: 41 tables with English column names and percentages rounded to one decimal;
- `paper_tables_columns.csv`: the definition of every column;
- `paper_tables_manifest.json`: the source files of every table, the commit and the run date. The run of 21/11/2026
  recorded commit `b436fad` with uncommitted changes; the table code (`paper_tables.py`, notebook 26) was committed
  unchanged in `d6b9628` on 22/11/2026;
- `paper_tables_gates.csv`: the consistency checks;
- `paper_tables_fig_routing_en.png` and `paper_tables_fig_cost_tco_en.png`: Figures 3 and 4.

## 3. From manuscript to file

| Manuscript | File (`05_experiments/rq1_classification/`) |
|---|---|
| Table 4 | `paper_tables_m1_data_splits.csv` (Cohen's κ from the annotation records) |
| Table 6 | `paper_tables_m4_configurations.csv` |
| Table 7 | `paper_tables_rq3_parameters.csv` |
| Table 8 | `paper_tables_rq1_tier1.csv`, `paper_tables_rq1_tier2.csv` |
| Table 9 | `paper_tables_rq1_e2e.csv` |
| Table 10 | `paper_tables_rq1_pairs.csv`; all 378 pairs per test set in `paper_tables_app_pairs_all.csv` |
| Table 11 | `paper_tables_rq1_error_coding.csv`, `paper_tables_rq1_error_contribution.csv` |
| Table 12 | `paper_tables_rq2_architecture.csv`, with `paper_tables_rq2_b1_pairs.csv`, `paper_tables_rq2_b1_thresholds.csv`, `paper_tables_rq2_b1_configurations.csv` |
| Table 13 | `paper_tables_rq2_routing.csv`, with `paper_tables_rq2_routing_pairs.csv`, `paper_tables_app_routing_other_small.csv` |
| Table 14 | `paper_tables_rq3_cost_drivers.csv` |
| Table 15 | `paper_tables_rq3_break_even.csv` |
| Table 16 | `paper_tables_rq3_cheapest.csv`, with `paper_tables_rq3_target_sensitivity.csv` |
| Cost scenarios in Section 4.4 | `paper_tables_rq3_existing_labels.csv`, `paper_tables_rq3_llm_price.csv`, `paper_tables_rq3_routing_rates.csv`, `paper_tables_rq3_observed_scale.csv` |
| Prompt variant and calibration (Sections 4.2 and 4.3) | `paper_tables_rq1_prompt_variant*.csv`, `paper_tables_app_calibration*.csv` |
| Tables A2, A3, A4 | `paper_tables_app_per_seed.csv`, `paper_tables_rq1_dedup.csv`, `paper_tables_app_errors.csv` |
| Figure 1 | Workflow diagram drawn by the authors; it is part of the manuscript and is not generated from result files |
| Figure 2 | `paper_step2_pareto_en.png` (notebook 17) |
| Figures 3 and 4 | `paper_tables_fig_routing_en.png`, `paper_tables_fig_cost_tco_en.png` |

Tables 1, 2, 3, 5 and A1 contain no computed results. The Word manuscript was filled from these CSV files, and each cell can be checked
against them directly.

## 4. Statistical conventions

- Primary estimate: the canonical run of each configuration, that is, seed 42 for trained models, the single run for the LLM and the
  canonical runs of the parts for hybrids and routers. The three-seed mean ± SD appears only in Table A2.
- 95% confidence intervals: paired percentile bootstrap with 10,000 resamples of the test reviews (seed 42); every configuration uses
  the same resamples.
- Exact McNemar tests on per-review correctness. Holm adjustment within the registered families: 378 end-to-end pairs per test set,
  10 pairs per tier, 24 routing pairs per test set (Round 3), 4 calibration pairs, 8 pairs for the decision-rule control and 7 for the
  routing policies (Round 4).
- Random routing: 1,000 draws with `numpy.random.default_rng(20261121)`.
- Near duplicates: cosine similarity of at least 0.90 between character 3–5-gram TF-IDF vectors of normalised text (Unicode NFC, lower
  case, punctuation replaced by spaces, whitespace collapsed), exact duplicates excluded; sensitivity threshold 0.80.

## 5. Provenance of the saved predictions

| Predictions or analysis | Notebook | Amendment | Evaluation round |
|---|---|---|---|
| ViSoBERT, Bi-LSTM and LLM tiers; flat ViSoBERT | `10_final_neural_kaggle`, `11_final_llm`, `12_holdout_scoring` | v2.1 | Round 1, 26 Oct 2026 |
| TF-IDF | `14_tfidf` | v3.2 | Round 2, 15 Nov 2026 |
| Flat LLM | `15_flat_llm` | v3.3 | Round 2 |
| PhoBERT (all tiers) and flat Bi-LSTM | `16_phobert_flat_bilstm_kaggle` (Kaggle GPU) | v3.4, v3.5 | Round 2 |
| Joint scoring of the 28 configurations | `17_joint_scoring` | v3.2–v3.5 | Round 2 |
| App-neutral prompt; manual error coding | `18_neutral_prompt`, `19_lazada_error_analysis` | v3.6, v3.7 | Round 2 |
| Cost model | `20_cost_model` | v3.8 | no test labels |
| Confidence routing; temperature scaling | `23_paper_routing_calibration` | v4.1 | Round 3, 16 Nov 2026 |
| Thresholded flat models, routing policies, near duplicates, cost scenarios | `25_paper_fair_controls` | v4.2 | Round 4, 21 Nov 2026 |
| Unified tables | `26_paper_tables` | v4.3 | Round 4, re-scoring only |

Every reading of test labels is logged in `01_data/HOLDOUT_LOCK.md`, and the dated amendments are in
`00_protocol/protocol_amendments.md`. Notebook numbers follow the renumbering of 16 November 2026 (`notebooks/README.md`).

## 6. Known limits

- The crawled raw data (`01_data/raw/`, `01_data/interim/`) are in the private repository for completeness. Shop replies in
  these files still contain unmasked names, so they must not be redistributed; the tables are regenerated from the masked,
  labelled split files and do not need the raw data.
- The few-shot example files store review IDs, labels and SHA-256 hashes, not texts. All 14 example texts are in the
  training file, and one of them is also quoted as an illustration in the annotation guideline
  (`00_protocol/annotation_guideline_v1.md`). The complete few-shot prompts can therefore be rebuilt from the training file.
- GPT-4o-mini was called at temperature 0 with a fixed seed, but the provider served several system fingerprints, so repeated API
  calls may not return the same labels; reproduction relies on the saved outputs.
