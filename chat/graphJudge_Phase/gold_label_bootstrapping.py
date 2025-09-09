"""
Gold label bootstrapping functionality for the GraphJudge system.

This module implements the two-stage gold label bootstrapping process:
1. Stage 1: RapidFuzz string similarity matching
2. Stage 2: LLM semantic evaluation for uncertain cases
"""

import os
import asyncio
import json
import csv
import random
import re
from typing import List, Optional
from .data_structures import TripleData, BootstrapResult, BootstrapStatistics
from .config import GOLD_BOOTSTRAP_CONFIG
from .graph_judge_core import PerplexityGraphJudge


class GoldLabelBootstrapper:
    """
    Handles the two-stage gold label bootstrapping process.
    
    This class implements an automated approach to assign gold labels to
    knowledge graph triples by combining string similarity matching with
    semantic evaluation using LLM reasoning.
    """
    
    def __init__(self, graph_judge: PerplexityGraphJudge):
        """
        Initialize the gold label bootstrapper.
        
        Args:
            graph_judge (PerplexityGraphJudge): Graph judge instance for LLM evaluation
        """
        self.graph_judge = graph_judge
        self.rapidfuzz_available = self._check_rapidfuzz_availability()
    
    def _check_rapidfuzz_availability(self) -> bool:
        """
        Check if RapidFuzz is available for string similarity matching.
        
        Returns:
            bool: True if RapidFuzz is available, False otherwise
        """
        return self._import_rapidfuzz()
    
    def _import_rapidfuzz(self) -> bool:
        """
        Helper method to import rapidfuzz. Separated for easier testing.
        
        Returns:
            bool: True if import successful, False otherwise
        """
        try:
            from rapidfuzz import fuzz
            print("✓ RapidFuzz imported successfully for gold label bootstrapping")
            return True
        except ImportError:
            print("⚠️ RapidFuzz not available. Install with: pip install rapidfuzz")
            return False
    
    def load_triples_from_file(self, triples_file: str) -> List[TripleData]:
        """
        Load and parse triples from the generated graphs file.
        從生成的圖文件中加載和解析三元組
        
        Args:
            triples_file (str): Path to the triples file
            
        Returns:
            List[TripleData]: Parsed triples with metadata
        """
        triples = []
        
        if not os.path.exists(triples_file):
            print(f"⚠️ Triples file not found: {triples_file}")
            return triples
        
        try:
            with open(triples_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse different formats of triple representation
                    # Format 1: JSON-like lists in text
                    if '[' in line and ']' in line:
                        try:
                            # Find all potential JSON arrays in the line
                            # Pattern matches individual triple arrays: ["subject", "predicate", "object"]
                            triple_pattern = r'\[\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\]'
                            matches = re.findall(triple_pattern, line)
                            
                            for match in matches:
                                if len(match) == 3:
                                    subject, predicate, obj = match
                                    triple = TripleData(
                                        subject=subject.strip(),
                                        predicate=predicate.strip(),
                                        object=obj.strip(),
                                        source_line=line,
                                        line_number=line_num
                                    )
                                    triples.append(triple)
                                    
                        except Exception as e:
                            print(f"⚠️ Error processing line {line_num}: {e}")
                    
                    # Format 2: Simple "S P O" format
                    elif ' ' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            triple = TripleData(
                                subject=parts[0].strip(),
                                predicate=parts[1].strip(),
                                object=' '.join(parts[2:]).strip(),
                                source_line=line,
                                line_number=line_num
                            )
                            triples.append(triple)
                    
            print(f"✓ Loaded {len(triples)} triples from {triples_file}")
            return triples
            
        except Exception as e:
            print(f"✗ Error loading triples from {triples_file}: {e}")
            return []
    
    def load_source_text(self, source_file: str) -> List[str]:
        """
        Load source text lines for comparison.
        加載源文本行用於比較
        
        Args:
            source_file (str): Path to the source text file
            
        Returns:
            List[str]: List of source text lines
        """
        if not os.path.exists(source_file):
            print(f"⚠️ Source file not found: {source_file}")
            return []
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Limit source lines to prevent memory issues
            max_lines = GOLD_BOOTSTRAP_CONFIG['max_source_lines']
            if len(lines) > max_lines:
                print(f"⚠️ Source file has {len(lines)} lines, limiting to {max_lines}")
                lines = lines[:max_lines]
            
            print(f"✓ Loaded {len(lines)} source text lines from {source_file}")
            return lines
            
        except Exception as e:
            print(f"✗ Error loading source text from {source_file}: {e}")
            return []
    
    def stage1_rapidfuzz_matching(self, triples: List[TripleData], source_lines: List[str]) -> List[BootstrapResult]:
        """
        Stage 1: RapidFuzz string similarity matching.
        第一階段：RapidFuzz 字符串相似度匹配
        
        Args:
            triples (List[TripleData]): List of triples to evaluate
            source_lines (List[str]): Source text lines for comparison
            
        Returns:
            List[BootstrapResult]: Initial bootstrap results with fuzzy scores
        """
        if self.rapidfuzz_available:
            from rapidfuzz import fuzz
        else:
            print("⚠️ RapidFuzz not available, using mock scoring")
            # Mock fuzz for testing
            class MockFuzz:
                @staticmethod
                def partial_ratio(a, b):
                    return 50.0 if a != b else 100.0
            fuzz = MockFuzz()
        
        results = []
        threshold = GOLD_BOOTSTRAP_CONFIG['fuzzy_threshold']
        
        print(f"🔍 Stage 1: Running RapidFuzz matching with threshold {threshold}")
        
        for triple in triples:
            # Create triple string representation
            triple_str = f"{triple.subject} {triple.predicate} {triple.object}"
            
            best_score = 0.0
            best_source_idx = -1
            
            # Compare against all source lines
            for idx, source_line in enumerate(source_lines):
                if not source_line:
                    continue
                
                score = fuzz.partial_ratio(triple_str, source_line) / 100.0
                if score > best_score:
                    best_score = score
                    best_source_idx = idx
            
            # Determine auto_expected based on threshold
            if best_score >= threshold:
                auto_expected = True
                note = f"High similarity (≥{threshold}) with source"
            else:
                auto_expected = None  # Uncertain, needs Stage 2 evaluation
                note = f"Low similarity (<{threshold}), requires semantic evaluation"
            
            result = BootstrapResult(
                triple=triple,
                source_idx=best_source_idx,
                fuzzy_score=best_score,
                auto_expected=auto_expected,
                llm_evaluation=None,
                expected=auto_expected,  # Will be updated in Stage 2 if needed
                note=note
            )
            results.append(result)
        
        confirmed_count = sum(1 for r in results if r.auto_expected == True)
        uncertain_count = sum(1 for r in results if r.auto_expected is None)
        
        print(f"📊 Stage 1 Results: {confirmed_count} confirmed, {uncertain_count} uncertain (need Stage 2)")
        
        return results
    
    async def stage2_llm_semantic_evaluation(self, uncertain_results: List[BootstrapResult], 
                                           source_lines: List[str]) -> List[BootstrapResult]:
        """
        Stage 2: LLM semantic evaluation for uncertain cases.
        
        Args:
            uncertain_results (List[BootstrapResult]): Results that need semantic evaluation
            source_lines (List[str]): Source text lines for context
            
        Returns:
            List[BootstrapResult]: Updated results with LLM evaluations
        """
        if not uncertain_results:
            print("📊 Stage 2: No uncertain cases to evaluate")
            return uncertain_results
        
        print(f"🧠 Stage 2: LLM semantic evaluation for {len(uncertain_results)} uncertain cases")
        
        updated_results = []
        batch_size = GOLD_BOOTSTRAP_CONFIG['llm_batch_size']
        
        for i in range(0, len(uncertain_results), batch_size):
            batch = uncertain_results[i:i + batch_size]
            
            for result in batch:
                try:
                    triple = result.triple
                    source_context = ""
                    
                    # Get source context around the best matching line
                    if 0 <= result.source_idx < len(source_lines):
                        # Include context lines for better evaluation
                        start_idx = max(0, result.source_idx - 2)
                        end_idx = min(len(source_lines), result.source_idx + 3)
                        source_context = '\n'.join(source_lines[start_idx:end_idx])
                    
                    # Create semantic evaluation prompt
                    evaluation_prompt = f"""
請判斷以下三元組是否可以從給定的源文本中語義推導出來：

三元組：{triple.subject} {triple.predicate} {triple.object}

源文本上下文：
{source_context}

判斷規則：
- 如果源文本明確提到或可以合理推導出此三元組關係，回答 "Yes"
- 如果源文本與此三元組矛盾或完全無關，回答 "No"
- 只需回答 "Yes" 或 "No"，無需解釋

判斷結果："""

                    # Use the existing Perplexity system for evaluation
                    if not self.graph_judge.is_mock:
                        llm_evaluation = await self.graph_judge.judge_graph_triple(evaluation_prompt)
                    else:
                        # Mock evaluation for testing
                        llm_evaluation = "Yes" if "創作" in triple.predicate or "喜歡" in triple.predicate else "No"
                    
                    # Update result based on LLM evaluation
                    auto_expected = True if llm_evaluation == "Yes" else False
                    note = f"LLM semantic evaluation: {llm_evaluation}"
                    
                    updated_result = result._replace(
                        auto_expected=auto_expected,
                        llm_evaluation=llm_evaluation,
                        expected=auto_expected,
                        note=note
                    )
                    updated_results.append(updated_result)
                    
                    print(f"🔄 Evaluated: {triple.subject} {triple.predicate} {triple.object} → {llm_evaluation}")
                    
                    # Small delay to respect API limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error evaluating triple {triple.subject} {triple.predicate} {triple.object}: {e}")
                    # Keep original result but mark as error
                    error_result = result._replace(
                        note=f"LLM evaluation error: {str(e)}"
                    )
                    updated_results.append(error_result)
        
        confirmed_by_llm = sum(1 for r in updated_results if r.auto_expected == True)
        rejected_by_llm = sum(1 for r in updated_results if r.auto_expected == False)
        
        print(f"📊 Stage 2 Results: {confirmed_by_llm} confirmed by LLM, {rejected_by_llm} rejected by LLM")
        
        return updated_results
    
    def sample_uncertain_cases(self, results: List[BootstrapResult]) -> List[BootstrapResult]:
        """
        Sample uncertain cases for manual review.
        
        Args:
            results (List[BootstrapResult]): All bootstrap results
            
        Returns:
            List[BootstrapResult]: Results with sampled cases marked for manual review
        """
        sample_rate = GOLD_BOOTSTRAP_CONFIG['sample_rate']
        random.seed(GOLD_BOOTSTRAP_CONFIG['random_seed'])
        
        # Find cases that were initially uncertain (went through Stage 2)
        stage2_cases = [r for r in results if r.llm_evaluation is not None]
        
        if not stage2_cases:
            print("📊 No Stage 2 cases found for sampling")
            return results
        
        # Sample for manual review
        sample_size = max(1, int(len(stage2_cases) * sample_rate))
        sampled_indices = set(random.sample(range(len(stage2_cases)), 
                                          min(sample_size, len(stage2_cases))))
        
        print(f"📝 Sampling {sample_size} cases from {len(stage2_cases)} Stage 2 cases for manual review")
        
        updated_results = []
        stage2_idx = 0
        
        for result in results:
            if result.llm_evaluation is not None:  # This is a Stage 2 case
                if stage2_idx in sampled_indices:
                    # Mark for manual review
                    updated_result = result._replace(
                        expected=None,  # Clear auto-assignment, needs manual review
                        note=result.note + " | SAMPLED FOR MANUAL REVIEW"
                    )
                    updated_results.append(updated_result)
                else:
                    updated_results.append(result)
                stage2_idx += 1
            else:
                updated_results.append(result)
        
        manual_review_count = sum(1 for r in updated_results if r.expected is None and "MANUAL REVIEW" in r.note)
        print(f"📝 {manual_review_count} cases marked for manual review")
        
        return updated_results
    
    def save_bootstrap_results(self, results: List[BootstrapResult], output_file: str) -> bool:
        """
        Save bootstrap results to CSV file.
        
        Args:
            results (List[BootstrapResult]): Bootstrap results to save
            output_file (str): Output CSV file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['subject', 'predicate', 'object', 'source_idx', 'fuzzy_score', 
                            'auto_expected', 'expected', 'note']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    writer.writerow({
                        'subject': result.triple.subject,
                        'predicate': result.triple.predicate,
                        'object': result.triple.object,
                        'source_idx': result.source_idx,
                        'fuzzy_score': f"{result.fuzzy_score:.3f}",
                        'auto_expected': result.auto_expected if result.auto_expected is not None else '',
                        'expected': result.expected if result.expected is not None else '',
                        'note': result.note
                    })
            
            print(f"✅ Bootstrap results saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving bootstrap results: {e}")
            return False
    
    def calculate_statistics(self, results: List[BootstrapResult]) -> BootstrapStatistics:
        """
        Calculate statistics from bootstrap results.
        
        Args:
            results (List[BootstrapResult]): Bootstrap results
            
        Returns:
            BootstrapStatistics: Calculated statistics
        """
        total_triples = len(results)
        auto_confirmed = sum(1 for r in results if r.expected == True)
        auto_rejected = sum(1 for r in results if r.expected == False)
        manual_review = sum(1 for r in results if r.expected is None)
        coverage_percentage = (auto_confirmed + auto_rejected) / total_triples * 100 if total_triples > 0 else 0
        
        return BootstrapStatistics(
            total_triples=total_triples,
            auto_confirmed=auto_confirmed,
            auto_rejected=auto_rejected,
            manual_review=manual_review,
            coverage_percentage=coverage_percentage
        )
    
    async def bootstrap_gold_labels(self, triples_file: str, source_file: str, output_file: str) -> bool:
        """
        Main gold label bootstrapping method.
        
        This method implements the two-stage gold label bootstrapping process:
        1. Stage 1: RapidFuzz string similarity matching
        2. Stage 2: LLM semantic evaluation for uncertain cases
        
        Args:
            triples_file (str): Path to the triples file
            source_file (str): Path to the source text file  
            output_file (str): Path to the output CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("=" * 70)
        print("🎯 Gold Label Bootstrapping - Automatic Gold Label Assignment")
        print("=" * 70)
        
        try:
            # Load input data
            print("📚 Loading input data...")
            triples = self.load_triples_from_file(triples_file)
            if not triples:
                print("❌ No triples loaded, aborting")
                return False
            
            source_lines = self.load_source_text(source_file)
            if not source_lines:
                print("❌ No source text loaded, aborting")
                return False
            
            # Stage 1: RapidFuzz matching
            print("\n" + "="*50)
            print("🔍 Stage 1: RapidFuzz String Similarity Matching")
            print("="*50)
            stage1_results = self.stage1_rapidfuzz_matching(triples, source_lines)
            
            # Stage 2: LLM semantic evaluation for uncertain cases
            print("\n" + "="*50)
            print("🧠 Stage 2: LLM Semantic Evaluation")
            print("="*50)
            uncertain_results = [r for r in stage1_results if r.auto_expected is None]
            
            if uncertain_results:
                stage2_results = await self.stage2_llm_semantic_evaluation(uncertain_results, source_lines)
                
                # Merge results
                final_results = []
                stage2_dict = {(r.triple.subject, r.triple.predicate, r.triple.object): r 
                             for r in stage2_results}
                
                for result in stage1_results:
                    key = (result.triple.subject, result.triple.predicate, result.triple.object)
                    if key in stage2_dict:
                        final_results.append(stage2_dict[key])
                    else:
                        final_results.append(result)
            else:
                final_results = stage1_results
            
            # Sample uncertain cases for manual review
            print("\n" + "="*50)
            print("📝 Sampling for Manual Review")
            print("="*50)
            final_results = self.sample_uncertain_cases(final_results)
            
            # Save results
            print("\n" + "="*50)
            print("💾 Saving Results")
            print("="*50)
            success = self.save_bootstrap_results(final_results, output_file)
            
            if success:
                # Print final statistics
                stats = self.calculate_statistics(final_results)
                
                print(f"\n📊 Final Bootstrap Statistics:")
                print(f"   - Total triples processed: {stats.total_triples}")
                print(f"   - Auto-confirmed (True): {stats.auto_confirmed} ({stats.auto_confirmed/stats.total_triples*100:.1f}%)")
                print(f"   - Auto-rejected (False): {stats.auto_rejected} ({stats.auto_rejected/stats.total_triples*100:.1f}%)")
                print(f"   - Manual review needed: {stats.manual_review} ({stats.manual_review/stats.total_triples*100:.1f}%)")
                print(f"   - Coverage (auto-labeled): {stats.coverage_percentage:.1f}%")
                
                print(f"\n🎉 Gold label bootstrapping completed successfully!")
                print(f"📂 Results saved to: {output_file}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error during gold label bootstrapping: {e}")
            return False
