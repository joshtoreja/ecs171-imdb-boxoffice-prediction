Team Meeting Log
Week 1 – Kickoff

Date: Week 1 (Project Kickoff)
Attendees: Andrew, Logan, Alex

Agenda:

Finalize project topic

Define prediction task

Choose dataset

Assign preliminary team roles

Outline repository structure

Discussion:

The team discussed multiple project ideas and ultimately agreed to focus on predicting movie profitability using structured metadata from the TMDB dataset. We agreed that instead of predicting raw revenue, we would define profitability using ROI (Revenue / Budget).

We discussed the importance of using pre-release features to make the prediction realistic for real-world deployment.

We explored:

Available features in TMDB

Data quality issues (missing budgets/revenue)

Class imbalance concerns

Binary vs regression framing

We created a GitHub repository and set up:

/data

/notebooks

/src

/reports

meetingLogs.md

Decisions:

Use TMDB dataset.

Frame problem as binary classification.

Define ROI-based profitability.

Initial threshold discussed (later refined).

Use stratified train/test splits.

Andrew = Modeling Lead

Alex = EDA & Preprocessing Lead

Logan = Literature & Report Lead

Assignments:

Andrew

Build baseline logistic regression model

Define ROI calculation

Create initial train/test split

Alex

Perform initial EDA

Visualize budget vs revenue

Explore class distribution

Identify missing or zero budget issues

Logan

Begin literature review

Research prior work on movie success prediction

Draft motivation section

Week 2 – Problem Refinement & Baseline Modeling

Date: Week 2
Attendees: Andrew, Logan, Alex

Agenda:

Refine ROI threshold

Improve preprocessing

Compare baseline models

Evaluate initial results

Discussion:

We refined the definition of profitability and experimented with multiple ROI thresholds (2.0, 2.5, 3.0). We discussed how stricter ROI thresholds make prediction harder due to class imbalance.

Andrew implemented:

Logistic Regression

Random Forest

Initial Gradient Boosting

We observed:

Overly high performance when certain features were included.

Potential data leakage from post-release variables such as vote_count, vote_average, and popularity.

Alex cleaned:

Zero budgets

Unrealistic financial records

Log transformation of budget

Logan expanded literature review and aligned it toward ROI classification instead of revenue regression.

Decisions:

Move toward ROI ≥ 2.5 as primary target (more realistic profitability threshold).

Remove post-release influenced features from final model.

Use cross-validation instead of single split.

Focus evaluation on ROC-AUC rather than just accuracy.

Assignments:

Andrew

Implement 5-fold stratified cross-validation

Remove leakage features

Compare model performance without popularity

Alex

Document preprocessing pipeline

Perform feature importance analysis

Logan

Expand literature review to emphasize ROI-based prediction

Draft dataset description section

Week 3 – Robustness, Sensitivity, and Failure Analysis

(Aligned with formal Week 3 plan 

week 3 plan - Google Docs

)

Date: Week 3
Attendees: Andrew, Logan, Alex

Agenda:

Cross-track comparison

Threshold sensitivity analysis

Feature ablation

Pre-release vs post-release comparison

Failure analysis

Discussion:

Andrew conducted:

Cross-track model comparison table

Threshold sensitivity tests (ROI ≥ 2.0 and ROI ≥ 3.0)

False positive / false negative analysis

Interpretation of misclassified films

We observed:

Stricter ROI threshold reduced performance.

Larger dataset improved stability.

Removing popularity decreased ROC-AUC slightly but eliminated leakage risk.

Alex:

Performed feature ablation tests (removed log_budget and popularity_x_votes).

Conducted pre-release-only model test.

Compared performance drop.

Logan:

Added precision-recall curve.

Performed probability calibration histogram.

Refined literature alignment.

Decisions:

Main model will exclude popularity to avoid leakage.

Gradient Boosting outperformed Logistic and Random Forest.

ROC-AUC chosen as primary metric.

F1 used for threshold tuning.

Keep pre-release-only model as defensible final version.

Assignments:

Andrew

Lock hyperparameters for Gradient Boosting

Finalize modeling notebook

Alex

Prepare feature documentation

Prepare deployment input structure

Logan

Structure Experimental Results section

Insert placeholder figures

Week 4 – Finalization & Deployment

(Aligned with final plan 

week 4 plan - Google Docs

)

Date: Week 4
Attendees: Andrew, Logan, Alex

Agenda:

Final model locking

Report finalization

Flask deployment

Presentation prep

Discussion:

We finalized:

Dataset

TMDB cleaned dataset

ROI ≥ 2.5

Movies with valid budget & revenue

Final Model

Gradient Boosting Classifier

5-fold Stratified CV

Primary metric: ROC-AUC

Secondary: F1, Precision, Recall, Accuracy

Andrew:

Finalized modeling notebook

Generated ROC curve

Generated PR curve

Generated confusion matrix

Saved final_model.pkl

Documented feature order

Verified no data leakage

Logan:

Completed full IEEE-style report

Integrated figures

Wrote abstract, introduction, EDA, results, discussion

Added GitHub link

Alex:

Built Flask frontend

Created input form

Integrated model.predict_proba

Added probability output

Styled interface

Tested edge cases

Decisions:

No further model experimentation.

Lock ROI ≥ 2.5.

Lock Gradient Boosting as final model.

Demo will show live prediction on unseen example.

Assignments:

Andrew

Provide model + preprocessing logic to Alex

Provide final metrics to Logan

Logan

Submit final report

Polish interpretation sections

Alex

Final demo testing

Prepare deployment walkthrough