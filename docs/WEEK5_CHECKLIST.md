# Week 5 delivery checklist

Custom application: historical disaster report classification and review.
This adapts the support-ticket router handout; it is not a claim that its example
notebook has already been completed.

- [x] Separate local project folder and source-data snapshot.
- [x] Bounded product scope and source manifest.
- [x] Bounded source audit reviewed; ambiguous/duplicate cases treated explicitly.
- [x] Labelled train/development/final data, grouped to avoid episode/duplicate leakage.
- [x] Pinned local model and separate reproducible environment.
- [x] Verified completion/label boundaries and untuned baseline.
- [x] Actual LoRA training, configuration, saved adapter and loss evidence.
- [x] Adapter merge and five predeclared obvious smoke tests.
- [x] Matched merged-versus-base evaluation, including per-class metrics and confusion matrix.
- [x] Review app: classify, confirm/correct, save, reopen and export.
- [x] Evaluation of the model actually served in that workflow.
- [ ] Dataset attribution, code/data licensing scope and reproducibility instructions.
- [ ] New GitHub repository published with appropriate assets and no secrets/base weights.
- [ ] Loom demonstration and submission link recorded.

Deadline: November 8, 2026 at 11:59 PM Pacific.

Original handout and linked notebook reviewed November 6, 2026. See
`HANDOUT_ALIGNMENT.md` for the local MLX/custom-data adaptation versus the
example Colab/LLaMA Board workflow. One local LoRA/loss/merge/smoke/matched
development cycle is now evidenced in
`LORA_RUN.md`. App workflow and actual served development evaluation are now
recorded in `REVIEW_APP.md`; final packaging/licensing, publication and
Loom/submission requirements remain pending.
