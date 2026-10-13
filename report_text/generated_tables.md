### Table 1. Corpus construction summary

| Metric | Value |
| --- | --- |
| Broad corpus articles | 40,232 |
| Strict corpus articles | 3,120 |
| Strict share of broad corpus | 7.76% |
| Months covered | 60 |
| Technology/Business section articles | 1,457 |
| Articles in other sections | 1,663 |


### Table 2. Manual audit summary

| Audit group | Labeled articles | Core AI articles | Core AI share |
| --- | --- | --- | --- |
| Broad but not strict | 100 | 1 | 1.0% |
| Strict corpus | 100 | 80 | 80.0% |


### Table 3. Classification model performance on the test set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.764 | 0.742 | 0.759 | 0.750 | 0.847 |
| Logistic regression | 0.745 | 0.717 | 0.749 | 0.733 | 0.838 |
| Random forest | 0.737 | 0.728 | 0.698 | 0.712 | 0.837 |
| Majority baseline | 0.534 | 0.000 | 0.000 | 0.000 | 0.500 |