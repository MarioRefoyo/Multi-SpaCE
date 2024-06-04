import numpy as np


def fitness_function_final(ms, predicted_probs, desired_class, outlier_scores,
                           invalid_penalization, alpha, beta, eta, gamma, sparsity_balancer):
    # Sparsity calculator
    ones_pct = ms.sum(axis=(1, 2)) / (ms.shape[1] * ms.shape[2])
    # subsequences = np.count_nonzero(np.diff(ms, prepend=0) == 1, axis=1)
    # Working in Sub-SpaCE for univariate time series: subsequences = np.count_nonzero(np.diff(ms, prepend=0, axis=1) == 1, axis=(1, 2))
    subsequences = np.count_nonzero(np.diff(ms, prepend=0, axis=1) == 1, axis=(1, 2))
    subsequences_pct = subsequences / ((ms.shape[1] // 2) * ms.shape[2])
    sparsity_term = sparsity_balancer * ones_pct + (1 - sparsity_balancer) * subsequences_pct ** gamma

    # Penalization for not prob satisfied
    desired_class_probs = predicted_probs[:, desired_class]
    predicted_classes = np.argmax(predicted_probs, axis=1)
    penalization = (predicted_classes != desired_class).astype(int)

    # Clip outlier scores
    outlier_scores[outlier_scores < 0] = 0

    # Calculate fitness
    fit = alpha * desired_class_probs - beta * sparsity_term - eta * outlier_scores - penalization * invalid_penalization

    return fit, desired_class_probs