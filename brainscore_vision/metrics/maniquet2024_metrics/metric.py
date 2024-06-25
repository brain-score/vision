#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 00:44:19 2024

@author: costantino_ai
"""
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from brainscore_core.metrics import Metric, Score
from brainio.assemblies import BehavioralAssembly


class ConfusionSimilarity(Metric):
    """
    A metric to compute the similarity between model-generated confusion matrices and human confusion data.

    Methods:
        _extract_subjects(assembly): Extracts and sorts unique subject identifiers from the assembly.
        _rollout_matrix(matrix, remove_diagonal=True): Flattens a matrix into a vector, optionally removing diagonal elements.
        _label_from_probability(probabilities): Derives predicted labels from probabilities.
        _accuracy(y_true, y_pred): Calculates the accuracy of predictions.
        _ceiling(assembly, precomputed=True): Computes the ceiling performance by assessing the highest correlation across subjects.
        __call__(probabilities, human_assembly): Computes the correlation between model and human confusion matrices normalized by the ceiling.
    """

    def _extract_subjects(self, assembly):
        """
        Extracts and sorts unique subject identifiers from the assembly.

        Args:
            assembly (xarray.Dataset): The data assembly containing subject IDs.

        Returns:
            list: Sorted list of unique subject IDs.
        """
        return list(sorted(set(assembly["subject_id"].values)))

    def _rollout_matrix(self, matrix, remove_diagonal=True):
        """
        Flattens a matrix into a vector. Optionally removes diagonal elements to ignore self-comparison.

        Args:
            matrix (np.array): A square matrix.
            remove_diagonal (bool): Whether to remove the diagonal elements of the matrix.

        Returns:
            np.array: The flattened matrix as a vector.
        """
        if remove_diagonal:
            # Create a mask to remove diagonal elements from the matrix.
            mask = np.eye(matrix.shape[0], dtype=bool)
            return matrix[~mask].ravel()
        else:
            return matrix.ravel()

    def _label_from_probability(self, probabilities):
        """
        Derives predicted labels from probabilities by selecting the class with the highest probability.

        Args:
            probabilities (xarray.Dataset): Dataset containing class probabilities.

        Returns:
            tuple: Arrays of true labels and predicted labels.
        """
        # Extract the class with the highest probability for each instance.
        classes = probabilities.choice.values
        indices = np.argmax(probabilities.values, axis=1)
        y_pred = classes[indices]
        y_true = probabilities.image_label.values
        return y_true, y_pred

    def _accuracy(self, y_true, y_pred):
        """
        Calculates the accuracy of predictions.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns:
            float: The accuracy of the predictions.
        """
        return sum(y_true == y_pred) / len(y_pred)

    def _ceiling(self, assembly, precomputed=True):
        """
        Compute the noise ceiling of a confusion matrix using split-half correlations.

        Args:
            assembly: (Human) Assembly with expected columns 'predicted'and 'image_label'.
            precomputed (Bool): If true, use precomputed ceiling measure to save time.

        Returns:
            score (float): Noise ceiling average.
        """
        if precomputed:
            # This is to save quite a lot of time. It was precomputed on the Maniquet2024
            # human data assembly, which includes 218 participants tested on the
            # Maniquet2024 stimulus set.
            return 0.54007

        # Get labels and subjects lists
        labels = list(set(assembly.image_label.values))
        subjects = self._extract_subjects(assembly)

        # Start recording correlation scores
        correlation_scores = []
        for subject in subjects:

            # Select data from a single subject
            subj_df = assembly.sel(subject_id=subject)

            # Split it in two randomly
            n_rows = int(np.round(len(subj_df) / 2))
            half = np.random.randint(0, len(subj_df), size=n_rows)
            part_one, part_two = subj_df[half], subj_df[~half]

            # Compute confusion matrix for each half
            cm_one = confusion_matrix(
                y_true=part_one["image_label"],
                y_pred=part_one["prediction"],
                labels=labels,
            )
            cm_two = confusion_matrix(
                y_true=part_two["image_label"],
                y_pred=part_two["prediction"],
                labels=labels,
            )

            # Compute Pearson correlation between the two confusion matrices.
            correlation_score = pearsonr(
                self._rollout_matrix(cm_one),
                self._rollout_matrix(cm_two),
            )[0]
            correlation_scores.append(correlation_score)

        # Average correlations as a measure of reliability
        ceiling = np.mean(correlation_scores)

        return ceiling

    def __call__(
        self, probabilities: BehavioralAssembly, human_assembly: BehavioralAssembly
    ) -> Score:
        """
        Computes the correlation between model and human confusion matrices normalized by the ceiling.

        Args:
            probabilities (BehavioralAssembly): Model's predicted probabilities.
            human_assembly (BehavioralAssembly): Human baseline responses.

        Returns:
            Score: The normalized correlation score as a performance metric.
        """
        assert sorted(set(probabilities.choice.values)) == sorted(
            set(human_assembly.image_label.values)
        )

        # Extract labels from the model probabilities.
        y_true, y_pred = self._label_from_probability(probabilities)

        # Calculate the model's confusion matrix.
        dnn_confmat = confusion_matrix(
            y_true=y_true, y_pred=y_pred, labels=probabilities.choice.values
        )

        # Calculate the human confusion matrix.
        human_confmat = confusion_matrix(
            y_true=human_assembly["image_label"],
            y_pred=human_assembly["prediction"],
            labels=probabilities.choice.values,
        )

        # Compute the Pearson correlation between the model and human confusion matrices.
        correlation_score = pearsonr(
            self._rollout_matrix(human_confmat), self._rollout_matrix(dnn_confmat)
        )[0]
        ceiling = self._ceiling(human_assembly, precomputed=True)

        # Normalize by ceiling
        score = Score(correlation_score / ceiling)
        score.attrs["raw"] = correlation_score
        score.attrs["ceiling"] = ceiling

        return score


class TasksConsistency(Metric):
    """
    A metric to compute the consistency between model and human accuracy profiles across different tasks.

    Methods:
        _extract_subjects(assembly): Extracts and sorts unique subject identifiers from the assembly.
        _extract_tasks(assembly): Extracts and sorts unique task identifiers from the assembly.
        _rollout_matrix(matrix, remove_diagonal=True): Flattens a matrix into a vector, optionally removing diagonal elements.
        _label_from_probability(probabilities): Derives predicted labels from probabilities.
        _accuracy(y_true, y_pred): Calculates the accuracy of predictions.
        _ceiling(assembly, precomputed=True): Computes the ceiling performance by assessing the highest correlation across subjects.
        _map_human_to_dnn_categories(human_task): Maps a human task name to the corresponding DNN categories of 'manipulation' and 'manipulation_details'.
        __call__(probabilities, human_assembly): Computes the correlation between model and human confusion matrices normalized by the ceiling.
    """

    def _extract_subjects(self, assembly):
        """
        Extracts and sorts unique subject identifiers from the assembly.

        Args:
            assembly (xarray.Dataset): The data assembly containing subject IDs.

        Returns:
            list: Sorted list of unique subject IDs.
        """
        return list(sorted(set(assembly["subject_id"].values)))

    def _extract_tasks(self, assembly):
        """
        Extracts and sorts unique task identifiers from the assembly.

        Args:
            assembly (xarray.Dataset): The data assembly containing task IDs.

        Returns:
            list: Sorted list of unique task IDs.
        """
        return list(sorted(set(assembly["task"].values)))

    def _rollout_matrix(self, matrix, remove_diagonal=True):
        """
        Flattens a matrix into a vector. Optionally removes diagonal elements to ignore self-comparison.

        Args:
            matrix (np.array): A square matrix.
            remove_diagonal (bool): Whether to remove the diagonal elements of the matrix.

        Returns:
            np.array: The flattened matrix as a vector.
        """
        if remove_diagonal:
            # Create a mask to remove diagonal elements from the matrix.
            mask = np.eye(matrix.shape[0], dtype=bool)
            return matrix[~mask].ravel()
        else:
            return matrix.ravel()

    def _label_from_probability(self, probabilities):
        """
        Derives predicted labels from probabilities by selecting the class with the highest probability.

        Args:
            probabilities (xarray.Dataset): Dataset containing class probabilities.

        Returns:
            tuple: Arrays of true labels and predicted labels.
        """
        # Extract the class with the highest probability for each instance.
        classes = probabilities.choice.values
        indices = np.argmax(probabilities.values, axis=1)
        y_pred = classes[indices]
        y_true = probabilities.image_label.values
        return y_true, y_pred

    def _accuracy(self, y_true, y_pred):
        """
        Calculates the accuracy of predictions.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns:
            float: The accuracy of the predictions.
        """
        return sum(y_true == y_pred) / len(y_pred)

    def _ceiling(self, assembly, precomputed=True):
        """
        Computes the ceiling performance by assessing the average split-half correlation across subjects.

        Args:
            assembly (xarray.Dataset): The data assembly containing subject data.
            precomputed (bool): Whether to use precomputed ceiling value.

        Returns:
            Score: The average correlation score across all subject pairs.
        """
        if precomputed:
            # This precomputed value is based on the Maniquet2024 human data assembly,
            # which includes 218 participants tested on the Maniquet2024 stimulus set.
            return 0.99810

        # Initialize an empty list to store correlations for each iteration
        iter_task_correlations = []

        # Perform 50 iterations for split-half correlation
        for i in range(50):

            # Randomly split the data assembly into two halves
            n_rows = int(np.round(len(assembly) / 2))
            half = np.random.randint(0, len(assembly), size=n_rows)
            part_one, part_two = assembly[half], assembly[~half]

            # Extract performance vectors for each half across all tasks
            perf_vec_one = [
                float(np.mean(part_one[part_one["task"] == task]))
                for task in self.human_tasks
            ]
            perf_vec_two = [
                float(np.mean(part_two.loc[part_two["task"] == task]))
                for task in self.human_tasks
            ]

            # Calculate the Pearson correlation between the performance vectors of the two halves
            corr_perf = pearsonr(perf_vec_one, perf_vec_two)[0]

            # Append the correlation result to the list for this iteration
            iter_task_correlations.append(corr_perf)

        return np.mean(iter_task_correlations)

    def _map_human_to_dnn_categories(self, human_task):
        """
        Maps a human task name to the corresponding DNN categories of 'manipulation' and 'manipulation_details'.

        Args:
            human_task (str): A task name from the human tasks list.

        Returns:
            tuple: A tuple where the first element is the 'manipulation' and the second is 'manipulation_details'.
        """
        # Mapping based on the provided details
        manipulation_mapping = {
            "clutter": "clutter",
            "control": "control",
            "occlusion": "occluder",
            "scrambling": "phasescrambling",
        }

        detail_mapping = {
            "heavy": "heavy",
            "light": "light",
            "highpass": "highpass",
            "lowpass": "lowpass",
            "few_large_blobs_high": "fewlarge-high",
            "few_large_blobs_low": "fewlarge-low",
            "few_large_deletion_high": "fewlarge-high",
            "few_large_deletion_low": "fewlarge-low",
            "many_small_blobs_high": "manysmall-high",
            "many_small_blobs_low": "manysmall-low",
            "many_small_deletion_high": "manysmall-high",
            "many_small_deletion_low": "manysmall-low",
            "few_large_partial_viewing_high": "fewlarge-high",
            "few_large_partial_viewing_low": "fewlarge-low",
            "many_small_partial_viewing_high": "manysmall-high",
            "many_small_partial_viewing_low": "manysmall-low",
        }

        parts = human_task.split("_")
        if "control" in parts:
            # Handle control separately as it doesn't fit other patterns
            return ("control", "control")

        # Determine manipulation by first relevant keyword
        manipulation = next(
            (manipulation_mapping[key] for key in manipulation_mapping if key in parts),
            None,
        )

        # Construct a detail key from remaining parts excluding known manipulation keys
        detail_parts = [part for part in parts if part not in manipulation_mapping]
        detail_key = "_".join(detail_parts)

        # Find the matching manipulation detail
        manipulation_detail = detail_mapping.get(
            detail_key, "control"
        )  # Default to control if no match found

        return (manipulation, manipulation_detail)

    def __call__(
        self, probabilities: BehavioralAssembly, human_assembly: BehavioralAssembly
    ) -> Score:
        """
        Computes the correlation between model and human accuracy profiles across tasks, normalized by the ceiling.

        Args:
            probabilities (BehavioralAssembly): Model's predicted probabilities.
            human_assembly (BehavioralAssembly): Human baseline responses.

        Returns:
            Score: The normalized correlation score as a performance metric.
        """
        assert sorted(set(probabilities.choice.values)) == sorted(
            set(human_assembly.image_label.values)
        )

        # Get list of tasks
        self.human_tasks = self._extract_tasks(human_assembly)

        # Store accuracies
        dnn_accs = []
        human_accs = []

        # Calculate the model's accuracy across tasks.
        for human_task in self.human_tasks:
            # Convert the human task into DNN lingo
            manipulation, manipulation_details = self._map_human_to_dnn_categories(
                human_task
            )

            # Extract labels from the model probabilities.
            probabilities_filtered = probabilities[
                (probabilities["manipulation"] == manipulation)
                & (probabilities["manipulation_details"] == manipulation_details)
            ]

            dnn_y_true, dnn_y_pred = self._label_from_probability(probabilities_filtered)
            dnn_acc = self._accuracy(dnn_y_true, dnn_y_pred)
            dnn_accs.append(dnn_acc)

            # Extract labels from the human responses.
            human_responses_filtered = human_assembly[
                human_assembly["task"] == human_task
            ]
            human_acc = self._accuracy(
                human_responses_filtered["image_label"],
                human_responses_filtered["prediction"],
            )
            human_accs.append(human_acc)

        # Compute the Pearson correlation between the model and human accuracy profiles.
        correlation_score = pearsonr(dnn_accs, human_accs)[0]
        ceiling = self._ceiling(human_assembly, precomputed=True)

        # Normalize by ceiling
        score = Score(correlation_score / ceiling)
        score.attrs["raw"] = correlation_score
        score.attrs["ceiling"] = ceiling

        return score
