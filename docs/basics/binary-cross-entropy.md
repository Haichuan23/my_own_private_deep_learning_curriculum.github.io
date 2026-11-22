---
sidebar_position: 2
title: Binary Cross-Entropy Loss
description: How binary cross-entropy scores predicted probabilities for binary labels.
---

# Binary Cross-Entropy Loss

Binary cross-entropy (BCE) measures how well a predicted probability matches a binary label.

Let `p` be the model's predicted probability that the label is `y = 1`, where `y` is either 0 or 1. For one example:

```text
L(p, y) = -y log(p) - (1 - y) log(1 - p)
```

Because the label is binary, only one term contributes:

| True label | Loss | The loss is minimized when |
| --- | --- | --- |
| `y = 1` | `-log(p)` | `p` approaches 1 |
| `y = 0` | `-log(1 - p)` | `p` approaches 0 |

## Understanding `-log(p)`

When the true label is 1, `p` is the probability assigned to the correct class. The negative logarithm gives a small penalty to confident correct predictions and a large penalty to confident wrong predictions.

![Plot of negative log p against predicted probability p](/img/basics/negative-log-loss.svg)

The curve shows three useful facts:

- The loss is 0 when `p = 1`.
- The loss grows as the model assigns less probability to the correct class.
- The loss approaches infinity as `p` approaches 0, heavily penalizing confident mistakes.

For example, when the true label is 1:

- Predicting `p = 0.9` gives a loss of about `0.105`.
- Predicting `p = 0.1` gives a loss of about `2.303`.

When the true label is 0, BCE uses `-log(1 - p)`, which is the same idea with the curve reflected horizontally.

## From one example to a dataset

Training usually minimizes the mean BCE across a batch of `N` examples:

```text
BCE = -(1/N) Σ [yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ)]
```

A lower BCE means the predicted probabilities align more closely with the true labels. In practice, deep-learning libraries often compute BCE directly from logits for better numerical stability.
