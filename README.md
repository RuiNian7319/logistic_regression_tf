# Models using Logistic Regression - Added some new stuff

### Time-series logistic regression:

**Definitions**
Threshold: Amount to multiply or divide the data's median to locate the limit for "anomalous" data

Dec 10th (using threshold = 3): 
- Time series with x(t - 1) data <br>
  - Precision was bad
- Time series with x(t - 2), x(t - 1)
  - Precision was extremely bad
- Time series with x(t - 1), x(t)
  - Precision was still bad
- Confirmed my data stacking method works, so no longer have to make massive data set for many time stamps

<br>

Dec 11th (using threshold = 15):
- Time series with x(t) data <br>
  **Activation = 0.70**
  - Precision: 15% | Recall = 98% | Did not catch: 89117

  **Activation = 0.95**
  - Precision: 30% | Recall = 79% | Did not catch: 83719, 89117, 159718, 230650, 246616, 345358, 372290, 394527, 459983, 521204

To increase recall, add more negative examples

<br>

Dec 11th (using threshold = 15, 5:1 negative:positive examples)
- Time series with x(t) data <br>
  **Activation = 0.70**
  - Precision: 54% | Recall = 96% | Did not catch: 89117, 309584

  **Activation = 0.95**
  - Precision: 41% | Recall = 38% | Did not catch: Too many...

  **Activation = 0.82**
  - Precision: 56% | Recall = 93% | Did not catch: 89117, 309584, 441921

<br>

Dec 11th (using threshold = 15, 10:1 negative:positive examples)
- Time series with x(t) data <br>
  **Activation = 0.70**
  - Precision: 58% | Recall = 98% | Did not catch: 89117

  **Activation = 0.83**
  - Precision: 57% | Recall = 92% | Did not catch: 89117, 201631, 309584, 430043

<br>

Dec 11th (using threshold = 15, 8:1 negative:positive examples)
- Time series with x(t), x(t - 1) data <br>
  **Activation = 0.70**
  - Precision: 40% | Recall = 96% | Did not catch: 89117, 201631

  **Activation = 0.83**
  - Precision: 42% | Recall = 92% | Did not catch: 89117, 201631, 208134, 430043

<br>

Dec 11th (using threshold = 15, 12:1 negative:positive examples)
- Time series with x(t), x(t - 1) data <br>
  **Activation = 0.70**
  - Precision: 53% | Recall = 96% | Did not catch: 89117, 309584

  **Activation = 0.83**
  - Precision: 57% | Recall = 96% | Did not catch: 89117, 309584

<br>

Dec 11th (using threshold = 15, 12:1 negative:positive examples, used L2 Regularization with lambda = 0.01)
- Time series with x(t), x(t - 1) data <br>
  **Activation = 0.70**
  - Precision: 54% | Recall = 94% | Did not catch: 89117, 102656, 309584

  **Activation = 0.83**
  - Precision: 62% | Recall = 51% | Did not catch: Too many

<br>

Dec 11th (using threshold = 15, 12:1 negative:positive examples, used L2 Regularization with lambda = 0.001)
- Time series with x(t), x(t - 1) data <br>
  **Activation = 0.70**
  - Precision: 69% | Recall = 96% | Did not catch: 89117, 309584

  **Activation = 0.83**
  - Precision: 75% | Recall = 96% | Did not catch: 89117, 309584

  **Activation = 0.83**
  - Precision: 81% | Recall = 92% | Did not catch: 89117, 159718, 201631, 309584
