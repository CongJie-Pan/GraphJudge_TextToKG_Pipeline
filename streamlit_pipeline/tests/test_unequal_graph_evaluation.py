"""
Test cases for unequal graph size evaluation.

This module tests the enhanced evaluation system's ability to handle
variable triple counts in generated vs. reference graphs, addressing
the core issue identified in the GraphJudge paper evaluation limitations.
"""

import pytest
import logging
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from streamlit_pipeline.core.models import Triple
from streamlit_pipeline.eval.graph_evaluator import GraphEvaluator
from streamlit_pipeline.eval.metrics.exact_matching import (
    get_triple_match_f1,
    get_graph_match_accuracy,
    analyze_triple_differences,
    get_evaluation_diagnostics
)
from streamlit_pipeline.eval.metrics.text_similarity import get_bleu_rouge_scores
from streamlit_pipeline.eval.metrics.semantic_similarity import get_bert_score


class TestUnequalGraphEvaluation:
    """Test evaluation with unequal graph sizes."""

    @pytest.fixture
    def sample_triples_small(self) -> List[Triple]:
        """Small reference graph with 3 triples."""
        return [
            Triple(subject="Alice", predicate="knows", object="Bob"),
            Triple(subject="Bob", predicate="likes", object="Pizza"),
            Triple(subject="Alice", predicate="lives_in", object="NYC")
        ]

    @pytest.fixture
    def sample_triples_large(self) -> List[Triple]:
        """Large predicted graph with 7 triples."""
        return [
            Triple(subject="Alice", predicate="knows", object="Bob"),  # Correct
            Triple(subject="Bob", predicate="likes", object="Pizza"),   # Correct
            Triple(subject="Alice", predicate="lives_in", object="NYC"), # Correct
            Triple(subject="Alice", predicate="works_at", object="Tech_Corp"), # Extra
            Triple(subject="Bob", predicate="owns", object="Car"),       # Extra
            Triple(subject="Charlie", predicate="visits", object="Alice"), # Extra
            Triple(subject="Alice", predicate="enjoys", object="Coffee")   # Extra
        ]

    @pytest.fixture
    def sample_triples_medium(self) -> List[Triple]:
        """Medium predicted graph with 5 triples (partial match)."""
        return [
            Triple(subject="Alice", predicate="knows", object="Bob"),    # Correct
            Triple(subject="Bob", predicate="likes", object="Pasta"),    # Incorrect (Pizza -> Pasta)
            Triple(subject="Alice", predicate="works_at", object="Tech_Corp"), # Extra
            Triple(subject="David", predicate="meets", object="Alice"),  # Extra
            Triple(subject="Bob", predicate="drives", object="BMW")      # Extra
        ]

    def test_exact_matching_over_generation(self, sample_triples_small, sample_triples_large):
        """Test exact matching when predicted graph is larger than reference."""
        ref_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_small]]
        pred_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_large]]

        # Should not fail and should provide meaningful F1 score
        f1_score = get_triple_match_f1(ref_graphs, pred_graphs)

        assert f1_score > 0.0, "F1 score should be > 0 for partial matches"
        assert f1_score <= 1.0, "F1 score should be <= 1.0"

        # Expected: 3 correct out of 7 predicted = precision = 3/7 ≈ 0.43
        # Expected: 3 correct out of 3 reference = recall = 3/3 = 1.0
        # Expected: F1 = 2 * (0.43 * 1.0) / (0.43 + 1.0) ≈ 0.60
        assert 0.55 <= f1_score <= 0.65, f"F1 score {f1_score} should be around 0.60"

    def test_exact_matching_under_generation(self, sample_triples_small, sample_triples_medium):
        """Test exact matching when predicted graph is similar size but with different content."""
        ref_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_small]]
        pred_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_medium]]

        f1_score = get_triple_match_f1(ref_graphs, pred_graphs)

        assert f1_score > 0.0, "F1 score should be > 0 for partial matches"

        # Expected: 1 correct (Alice knows Bob) out of 5 predicted = precision = 1/5 = 0.2
        # Expected: 1 correct out of 3 reference = recall = 1/3 ≈ 0.33
        # Expected: F1 = 2 * (0.2 * 0.33) / (0.2 + 0.33) ≈ 0.25
        assert 0.2 <= f1_score <= 0.3, f"F1 score {f1_score} should be around 0.25"

    def test_text_similarity_unequal_counts(self, sample_triples_small, sample_triples_large):
        """Test text similarity metrics with unequal graph counts."""
        ref_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_small]]
        pred_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_large]]

        scores = get_bleu_rouge_scores(ref_graphs, pred_graphs)

        # Should return valid scores, not empty
        assert "bleu" in scores
        assert "rouge" in scores
        assert scores["bleu"]["precision"] >= 0.0
        assert scores["bleu"]["recall"] >= 0.0
        assert scores["bleu"]["f1"] >= 0.0
        assert scores["rouge"]["precision"] >= 0.0
        assert scores["rouge"]["recall"] >= 0.0
        assert scores["rouge"]["f1"] >= 0.0

    def test_semantic_similarity_unequal_counts(self, sample_triples_small, sample_triples_large):
        """Test semantic similarity metrics with unequal graph counts."""
        ref_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_small]]
        pred_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_large]]

        # Mock BertScore to avoid dependency issues in tests
        with patch('streamlit_pipeline.eval.metrics.semantic_similarity.BERT_SCORE_AVAILABLE', False):
            scores = get_bert_score(ref_graphs, pred_graphs)

        # Should return valid scores using fallback method
        assert "precision" in scores
        assert "recall" in scores
        assert "f1" in scores
        assert scores["precision"] >= 0.0
        assert scores["recall"] >= 0.0
        assert scores["f1"] >= 0.0

    def test_graph_evaluator_with_size_differences(self, sample_triples_small, sample_triples_large):
        """Test GraphEvaluator with different sized graphs."""
        evaluator = GraphEvaluator(enable_bert_score=False, enable_ged=False)

        result = evaluator.evaluate_graph(sample_triples_large, sample_triples_small)

        assert result.success, f"Evaluation should succeed: {result.error}"
        assert result.metrics.triple_match_f1 > 0.0, "F1 score should be meaningful"

        # Check metadata contains size information
        assert "predicted_graph_size" in result.metadata
        assert "reference_graph_size" in result.metadata
        assert "size_analysis" in result.metadata

        assert result.metadata["predicted_graph_size"] == 7
        assert result.metadata["reference_graph_size"] == 3
        assert result.metadata["size_analysis"]["over_generated"] == True
        assert result.metadata["size_analysis"]["generation_behavior"] == "over_generation"

    def test_analyze_triple_differences_comprehensive(self, sample_triples_small, sample_triples_large):
        """Test comprehensive triple difference analysis."""
        ref_graph = [[t.subject, t.predicate, t.object] for t in sample_triples_small]
        pred_graph = [[t.subject, t.predicate, t.object] for t in sample_triples_large]

        analysis = analyze_triple_differences(ref_graph, pred_graph)

        # Basic counts
        assert analysis["total_reference"] == 3
        assert analysis["total_predicted"] == 7
        assert analysis["correct_matches"] == 3
        assert analysis["missing_triples"] == 0  # All reference triples found
        assert analysis["extra_triples"] == 4    # 4 additional triples

        # Metrics
        assert analysis["precision"] == 3/7  # 3 correct out of 7 predicted
        assert analysis["recall"] == 1.0     # All 3 reference triples found
        assert analysis["coverage"] == 1.0   # Complete coverage of reference
        assert analysis["over_generation_ratio"] == 4/3  # 4 extra / 3 reference

    def test_evaluation_diagnostics(self, sample_triples_small, sample_triples_large):
        """Test evaluation diagnostics functionality."""
        ref_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_small]]
        pred_graphs = [[[t.subject, t.predicate, t.object] for t in sample_triples_large]]

        diagnostics = get_evaluation_diagnostics(ref_graphs, pred_graphs)

        assert diagnostics["count_mismatch"] == True
        assert diagnostics["mismatch_type"] == "over_generation"
        assert diagnostics["graph_counts"]["reference"] == 1
        assert diagnostics["graph_counts"]["predicted"] == 1
        assert diagnostics["excess_predictions"] == 0  # At graph level, not triple level

    def test_empty_graph_handling(self):
        """Test evaluation with empty graphs."""
        empty_graph = []
        non_empty_graph = [["Alice", "knows", "Bob"]]

        # Empty predicted, non-empty reference
        f1_empty_pred = get_triple_match_f1([non_empty_graph], [empty_graph])
        assert f1_empty_pred == 0.0, "F1 should be 0 when predicted is empty"

        # Non-empty predicted, empty reference
        f1_empty_ref = get_triple_match_f1([empty_graph], [non_empty_graph])
        assert f1_empty_ref == 0.0, "F1 should be 0 when reference is empty"

        # Both empty
        f1_both_empty = get_triple_match_f1([empty_graph], [empty_graph])
        assert f1_both_empty == 0.0, "F1 should be 0 when both are empty"

    def test_graph_evaluator_validation_allows_size_differences(self):
        """Test that GraphEvaluator validation allows different graph sizes."""
        evaluator = GraphEvaluator(enable_bert_score=False, enable_ged=False)

        small_graph = [Triple(subject="A", predicate="knows", object="B")]
        large_graph = [
            Triple(subject="A", predicate="knows", object="B"),
            Triple(subject="B", predicate="likes", object="C"),
            Triple(subject="C", predicate="visits", object="A")
        ]

        # Should not raise validation errors
        result = evaluator.evaluate_graph(large_graph, small_graph)
        assert result.success, "Evaluation should succeed with different sizes"

        # Should provide meaningful metadata
        assert result.metadata["size_analysis"]["over_generated"] == True
        assert result.metadata["evaluation_coverage"]["coverage_ratio"] < 1.0

    def test_mathematical_soundness_precision_recall(self):
        """Test mathematical soundness of precision/recall calculations with different sizes."""
        # Create controlled test case
        reference = [["A", "rel1", "B"], ["B", "rel2", "C"], ["C", "rel3", "D"]]  # 3 triples
        predicted = [
            ["A", "rel1", "B"],  # Correct
            ["B", "rel2", "C"],  # Correct
            ["X", "rel4", "Y"],  # Extra
            ["Y", "rel5", "Z"]   # Extra
        ]  # 4 triples, 2 correct

        analysis = analyze_triple_differences(reference, predicted)

        # Mathematical verification
        expected_precision = 2 / 4  # 2 correct out of 4 predicted
        expected_recall = 2 / 3     # 2 correct out of 3 reference
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)

        assert analysis["precision"] == expected_precision, f"Precision should be {expected_precision}"
        assert analysis["recall"] == expected_recall, f"Recall should be {expected_recall}"
        assert abs(analysis["f1"] - expected_f1) < 1e-6, f"F1 should be {expected_f1}"

    def test_real_world_scenario_variable_generation(self):
        """Test real-world scenario with highly variable generation."""
        # Simulate realistic scenario: reference has 5 triples, generation varies wildly
        reference = [
            ["user", "has", "account"],
            ["account", "belongs_to", "service"],
            ["service", "provides", "feature"],
            ["feature", "costs", "price"],
            ["price", "currency", "USD"]
        ]

        # Scenario 1: Under-generation (LLM generates only 2 triples)
        under_generated = [
            ["user", "has", "account"],     # Correct
            ["account", "type", "premium"]  # Extra/incorrect
        ]

        # Scenario 2: Over-generation (LLM generates 10 triples)
        over_generated = reference + [
            ["user", "age", "25"],
            ["user", "location", "NYC"],
            ["account", "created", "2023"],
            ["service", "rating", "5_stars"],
            ["feature", "category", "premium"]
        ]

        # Test under-generation
        f1_under = get_triple_match_f1([reference], [under_generated])
        assert 0.0 < f1_under < 1.0, "Under-generation should give partial F1 score"

        # Test over-generation
        f1_over = get_triple_match_f1([reference], [over_generated])
        assert f1_over > f1_under, "Over-generation should have higher F1 than under-generation"
        # Over-generation: 5 correct out of 10 predicted = precision = 0.5
        # Over-generation: 5 correct out of 5 reference = recall = 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2/3 ≈ 0.67
        assert 0.6 <= f1_over <= 0.7, f"Over-generation F1 {f1_over} should be around 0.67"


class TestEvaluationSystemIntegration:
    """Integration tests for the complete evaluation system."""

    def test_full_pipeline_with_variable_counts(self):
        """Test the full evaluation pipeline with variable triple counts."""
        evaluator = GraphEvaluator(enable_bert_score=False, enable_ged=False)

        # Simulate GraphJudge output: variable triple counts
        scenarios = [
            # (predicted_count, reference_count, expected_behavior)
            (10, 5, "over_generation"),
            (3, 8, "under_generation"),
            (6, 6, "appropriate_size"),
            (1, 15, "under_generation"),
            (20, 3, "over_generation")
        ]

        for pred_count, ref_count, expected_behavior in scenarios:
            # Generate synthetic graphs
            pred_graph = [
                Triple(subject=f"entity_{i}", predicate=f"rel_{i%3}", object=f"target_{i}")
                for i in range(pred_count)
            ]
            ref_graph = [
                Triple(subject=f"entity_{i}", predicate=f"rel_{i%3}", object=f"target_{i}")
                for i in range(ref_count)
            ]

            result = evaluator.evaluate_graph(pred_graph, ref_graph)

            # Should succeed regardless of size differences
            assert result.success, f"Evaluation should succeed for {pred_count}vs{ref_count}"

            # Should provide correct behavior classification
            assert result.metadata["size_analysis"]["generation_behavior"] == expected_behavior

            # Should provide meaningful metrics
            if pred_count > 0 and ref_count > 0:
                assert result.metrics.triple_match_f1 >= 0.0
                assert result.metadata["quality_insights"]["meaningful_evaluation"] == True

    def test_error_logging_and_warnings(self, caplog):
        """Test that appropriate warnings are logged for size mismatches."""
        evaluator = GraphEvaluator(enable_bert_score=False, enable_ged=False)

        large_graph = [Triple(subject=f"s{i}", predicate="p", object=f"o{i}") for i in range(10)]
        small_graph = [Triple(subject="s1", predicate="p", object="o1")]

        with caplog.at_level(logging.INFO):
            result = evaluator.evaluate_graph(large_graph, small_graph)

        # Check that size difference is logged
        assert "Graph size difference" in caplog.text
        assert "predicted=10, reference=1" in caplog.text

    def test_metadata_completeness(self):
        """Test that all expected metadata is provided."""
        evaluator = GraphEvaluator(enable_bert_score=False, enable_ged=False)

        pred_graph = [Triple(subject="A", predicate="p1", object="B")]
        ref_graph = [
            Triple(subject="A", predicate="p1", object="B"),
            Triple(subject="C", predicate="p2", object="D")
        ]

        result = evaluator.evaluate_graph(pred_graph, ref_graph)

        # Check all expected metadata keys are present
        expected_keys = [
            "predicted_graph_size", "reference_graph_size", "size_difference",
            "size_analysis", "evaluation_coverage", "evaluation_settings",
            "performance", "quality_insights"
        ]

        for key in expected_keys:
            assert key in result.metadata, f"Missing metadata key: {key}"

        # Check nested structure
        assert "generation_behavior" in result.metadata["size_analysis"]
        assert "coverage_ratio" in result.metadata["evaluation_coverage"]
        assert "evaluation_mode" in result.metadata["evaluation_settings"]
        assert result.metadata["evaluation_settings"]["evaluation_mode"] == "count_agnostic"


class TestMalformedInputHandling:
    """Test handling of malformed and edge case inputs."""

    def test_empty_triple_handling(self):
        """Test evaluation with empty triples."""
        from streamlit_pipeline.core.models import Triple
        from streamlit_pipeline.eval.graph_evaluator import GraphEvaluator

        evaluator = GraphEvaluator(enable_bert_score=False, enable_ged=False)

        # Test with empty strings
        malformed_graph = [
            Triple(subject="", predicate="", object=""),
            Triple(subject="A", predicate="rel", object="B")  # One valid triple
        ]
        valid_graph = [
            Triple(subject="A", predicate="rel", object="B")
        ]

        result = evaluator.evaluate_graph(malformed_graph, valid_graph)

        # Should succeed but with warnings logged
        assert result.success
        # Should still compute meaningful metrics for the valid triple
        assert result.metrics.triple_match_f1 >= 0.0

    def test_malformed_triple_structures_in_metrics(self):
        """Test metrics handling of malformed triple structures."""
        from streamlit_pipeline.eval.metrics.exact_matching import get_triple_match_f1
        from streamlit_pipeline.eval.metrics.text_similarity import get_bleu_rouge_scores
        from streamlit_pipeline.eval.metrics.semantic_similarity import get_bert_score

        # Test with various malformed structures
        malformed_ref = [
            [],  # Empty triple
            ["A"],  # Incomplete triple
            ["B", "rel"],  # Missing object
            ["C", "rel", "D"],  # Valid triple
            [1, 2, 3],  # Non-string elements (will be converted)
        ]

        malformed_pred = [
            ["C", "rel", "D"],  # Valid match
            ["E", "rel2"],  # Incomplete
            [None, "rel3", "F"],  # None value
        ]

        # Should not crash and return meaningful scores
        f1_score = get_triple_match_f1([malformed_ref], [malformed_pred])
        assert isinstance(f1_score, float)
        assert 0.0 <= f1_score <= 1.0

        # Text similarity should handle malformed triples
        text_scores = get_bleu_rouge_scores([malformed_ref], [malformed_pred])
        assert "bleu" in text_scores
        assert "rouge" in text_scores
        assert all(0.0 <= text_scores[metric][key] <= 1.0
                  for metric in text_scores
                  for key in text_scores[metric])

        # Semantic similarity should handle malformed triples (with fallback)
        semantic_scores = get_bert_score([malformed_ref], [malformed_pred])
        assert "precision" in semantic_scores
        assert "recall" in semantic_scores
        assert "f1" in semantic_scores

    def test_none_and_null_inputs(self):
        """Test handling of None and null inputs."""
        from streamlit_pipeline.eval.metrics.exact_matching import (
            get_triple_match_f1, get_graph_match_accuracy
        )

        # Test with None inputs
        f1_none_ref = get_triple_match_f1(None, [[["A", "rel", "B"]]])
        assert f1_none_ref == 0.0

        f1_none_pred = get_triple_match_f1([[["A", "rel", "B"]]], None)
        assert f1_none_pred == 0.0

        # Test with empty lists
        f1_empty = get_triple_match_f1([], [])
        assert f1_empty == 0.0

        # Test graph accuracy with None inputs
        acc_none = get_graph_match_accuracy(None, [[["A", "rel", "B"]]])
        assert acc_none == 0.0

    def test_large_graph_performance(self):
        """Test performance with large graphs to check for efficiency issues."""
        from streamlit_pipeline.core.models import Triple
        from streamlit_pipeline.eval.graph_evaluator import GraphEvaluator
        import time

        evaluator = GraphEvaluator(enable_bert_score=False, enable_ged=False)

        # Create large graphs (1000 triples each)
        large_pred = [
            Triple(subject=f"entity_{i}", predicate=f"rel_{i%10}", object=f"target_{i}")
            for i in range(1000)
        ]
        large_ref = [
            Triple(subject=f"entity_{i}", predicate=f"rel_{i%10}", object=f"target_{i}")
            for i in range(500)  # Different size
        ]

        start_time = time.time()
        result = evaluator.evaluate_graph(large_pred, large_ref)
        end_time = time.time()

        # Should complete within reasonable time (< 10 seconds)
        assert (end_time - start_time) < 10.0
        assert result.success
        # Should handle size difference gracefully
        assert result.metadata["size_analysis"]["over_generated"] == True

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in triples."""
        from streamlit_pipeline.core.models import Triple
        from streamlit_pipeline.eval.graph_evaluator import GraphEvaluator

        evaluator = GraphEvaluator(enable_bert_score=False, enable_ged=False)

        unicode_graph = [
            Triple(subject="用户", predicate="喜欢", object="音乐"),  # Chinese
            Triple(subject="café", predicate="serves", object="espresso"),  # Accents
            Triple(subject="user@domain.com", predicate="email_type", object="work"),  # Special chars
            Triple(subject="price", predicate="equals", object="$19.99"),  # Currency
        ]

        english_graph = [
            Triple(subject="user", predicate="likes", object="music"),
            Triple(subject="cafe", predicate="serves", object="coffee"),
        ]

        result = evaluator.evaluate_graph(unicode_graph, english_graph)

        # Should handle Unicode gracefully without crashing
        assert result.success
        assert result.metrics.triple_match_f1 >= 0.0

    def test_comprehensive_unicode_handling(self):
        """Test comprehensive Unicode handling across all metric types."""
        from streamlit_pipeline.eval.metrics.exact_matching import get_triple_match_f1, analyze_triple_differences
        from streamlit_pipeline.eval.metrics.text_similarity import get_bleu_rouge_scores
        from streamlit_pipeline.eval.metrics.semantic_similarity import get_bert_score

        # Test various Unicode character sets
        unicode_test_data = [
            # Chinese characters
            [["用户", "喜欢", "音乐"], ["咖啡馆", "提供", "咖啡"]],
            # Mix of languages and special characters
            [["café", "serves", "coffee"], ["user@domain.com", "email", "work"]],
            # Emojis and symbols
            [["user", "feels", "😊"], ["price", "equals", "€19.99"]],
        ]

        for unicode_graph in unicode_test_data:
            # Test exact matching with Unicode
            f1_score = get_triple_match_f1([unicode_graph], [unicode_graph])
            assert f1_score == 1.0  # Should match perfectly

            # Test text similarity with Unicode
            text_scores = get_bleu_rouge_scores([unicode_graph], [unicode_graph])
            assert text_scores["bleu"]["f1"] >= 0.0
            assert text_scores["rouge"]["f1"] >= 0.0

            # Test semantic similarity with Unicode (fallback mode)
            semantic_scores = get_bert_score([unicode_graph], [unicode_graph])
            assert semantic_scores["f1"] >= 0.0

            # Test diagnostic analysis with Unicode
            analysis = analyze_triple_differences(unicode_graph, unicode_graph)
            assert analysis["f1"] == 1.0
            assert isinstance(analysis["generation_behavior"], str)

    def test_diagnostic_functions_with_malformed_data(self):
        """Test diagnostic functions handle malformed data."""
        from streamlit_pipeline.eval.metrics.exact_matching import (
            analyze_triple_differences, get_evaluation_diagnostics
        )

        malformed_ref = [
            [],  # Empty triple
            ["A", "rel", "B"],  # Valid
            [None, "rel2"],  # Incomplete with None
        ]

        malformed_pred = [
            ["A", "rel", "B"],  # Match
            ["C"],  # Incomplete
            [1, 2, 3, 4, 5],  # Too many elements
        ]

        # Should not crash
        analysis = analyze_triple_differences(malformed_ref, malformed_pred)

        assert isinstance(analysis, dict)
        assert "precision" in analysis
        assert "recall" in analysis
        assert "f1" in analysis
        assert "generation_behavior" in analysis
        assert isinstance(analysis["generation_behavior"], str)

        # Test evaluation diagnostics
        diagnostics = get_evaluation_diagnostics([malformed_ref], [malformed_pred])
        assert isinstance(diagnostics, dict)
        assert "count_mismatch" in diagnostics
        assert "graph_counts" in diagnostics