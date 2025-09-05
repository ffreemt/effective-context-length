"""
Results Analyzer for Effective Context Length Testing

This module provides statistical analysis and effective context length determination
functionality for test results from the EffectiveContextLengthTester.
"""

import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import math

try:
    import numpy as np
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    np = None
    stats = None
    SCIPY_AVAILABLE = False

from .tester import TestSession, ContextLengthResults, TestResult
from .http_client import ErrorType
from .utils import format_number, format_duration


@dataclass
class ConfidenceInterval:
    """Confidence interval for a statistical measure."""
    lower: float
    upper: float
    confidence_level: float
    method: str = "normal"


@dataclass
class EffectiveContextAnalysis:
    """Analysis results for effective context length determination."""
    effective_context_length: Optional[int]
    confidence_interval: Optional[ConfidenceInterval]
    error_threshold_used: float
    success_rate_at_effective_length: float
    recommended_safe_length: Optional[int]
    analysis_method: str
    quality_score: float  # 0-1 score indicating confidence in the result
    warnings: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics from test results."""
    average_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    requests_per_second: float
    total_test_duration: float


@dataclass
class ErrorAnalysis:
    """Analysis of errors encountered during testing."""
    total_errors: int
    error_rate: float
    error_types: Dict[ErrorType, int]
    error_rate_by_context_length: Dict[int, float]
    context_length_error_threshold: Optional[int]  # First length where errors exceed threshold


@dataclass
class DataQualityAssessment:
    """Assessment of data quality and reliability."""
    total_data_points: int
    context_lengths_tested: int
    samples_per_length_stats: Dict[str, float]  # min, max, mean, median
    coverage_score: float  # 0-1 score for how well the range was covered
    consistency_score: float  # 0-1 score for result consistency
    reliability_warnings: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveAnalysis:
    """Complete analysis results."""
    effective_context: EffectiveContextAnalysis
    performance: PerformanceMetrics
    errors: ErrorAnalysis
    data_quality: DataQualityAssessment
    summary: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)


class ResultsAnalyzer:
    """
    Analyzes test results to determine effective context lengths and provide
    comprehensive statistical analysis.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the analyzer.
        
        Args:
            confidence_level: Confidence level for statistical calculations
        """
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
        
        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy not available. Using basic statistical methods.")
    
    def analyze_session(
        self,
        session: TestSession,
        error_threshold: float = 0.1,
        min_samples: int = 3
    ) -> ComprehensiveAnalysis:
        """
        Perform comprehensive analysis of a test session.
        
        Args:
            session: Test session to analyze
            error_threshold: Error rate threshold for effective length determination
            min_samples: Minimum samples required per context length
            
        Returns:
            ComprehensiveAnalysis with all results
        """
        self.logger.info("Starting comprehensive analysis of test session")
        
        # Validate session data
        if not session.context_results:
            raise ValueError("No test results to analyze")
        
        # Perform individual analyses
        effective_context = self.determine_effective_context_length(
            session.context_results, error_threshold, min_samples
        )
        
        performance = self.analyze_performance_metrics(session)
        errors = self.analyze_errors(session.context_results)
        data_quality = self.assess_data_quality(session, min_samples)
        
        # Generate summary
        summary = self._generate_summary(session, effective_context, performance, errors)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            effective_context, performance, errors, data_quality
        )
        
        analysis = ComprehensiveAnalysis(
            effective_context=effective_context,
            performance=performance,
            errors=errors,
            data_quality=data_quality,
            summary=summary,
            recommendations=recommendations
        )
        
        self.logger.info(f"Analysis complete. Effective context length: "
                        f"{effective_context.effective_context_length or 'Not determined'}")
        
        return analysis
    
    def determine_effective_context_length(
        self,
        context_results: Dict[int, ContextLengthResults],
        error_threshold: float = 0.1,
        min_samples: int = 3
    ) -> EffectiveContextAnalysis:
        """
        Determine the effective context length based on error rates.
        
        Args:
            context_results: Results by context length
            error_threshold: Maximum acceptable error rate
            min_samples: Minimum samples required for reliable results
            
        Returns:
            EffectiveContextAnalysis with determination results
        """
        if not context_results:
            return EffectiveContextAnalysis(
                effective_context_length=None,
                confidence_interval=None,
                error_threshold_used=error_threshold,
                success_rate_at_effective_length=0.0,
                recommended_safe_length=None,
                analysis_method="insufficient_data",
                quality_score=0.0,
                warnings=["No test results available"]
            )
        
        # Filter results with sufficient samples
        reliable_results = {
            length: result for length, result in context_results.items()
            if len(result.results) >= min_samples
        }
        
        warnings = []
        if len(reliable_results) < len(context_results):
            filtered_count = len(context_results) - len(reliable_results)
            warnings.append(f"Filtered out {filtered_count} context lengths with insufficient samples")
        
        if not reliable_results:
            return EffectiveContextAnalysis(
                effective_context_length=None,
                confidence_interval=None,
                error_threshold_used=error_threshold,
                success_rate_at_effective_length=0.0,
                recommended_safe_length=None,
                analysis_method="insufficient_samples",
                quality_score=0.0,
                warnings=warnings + ["No context lengths have sufficient samples"]
            )
        
        # Sort by context length
        sorted_lengths = sorted(reliable_results.keys())
        
        # Find effective length using multiple methods
        methods_results = {}
        
        # Method 1: Simple threshold method
        methods_results['threshold'] = self._find_effective_length_threshold(
            reliable_results, error_threshold
        )
        
        # Method 2: Statistical method (if scipy available)
        if SCIPY_AVAILABLE:
            methods_results['statistical'] = self._find_effective_length_statistical(
                reliable_results, error_threshold
            )
        
        # Method 3: Conservative method (stricter threshold)
        methods_results['conservative'] = self._find_effective_length_threshold(
            reliable_results, error_threshold * 0.5
        )
        
        # Choose best method result
        effective_length, method_used, quality_score = self._select_best_method_result(
            methods_results, reliable_results
        )
        
        # Calculate confidence interval
        confidence_interval = None
        if effective_length and SCIPY_AVAILABLE:
            confidence_interval = self._calculate_confidence_interval(
                reliable_results, effective_length
            )
        
        # Determine success rate at effective length
        success_rate = 0.0
        if effective_length and effective_length in reliable_results:
            success_rate = reliable_results[effective_length].success_rate
        
        # Calculate recommended safe length (10% buffer)
        recommended_safe_length = None
        if effective_length:
            recommended_safe_length = int(effective_length * 0.9)
        
        # Additional quality checks
        if quality_score < 0.7:
            warnings.append("Low confidence in effective context length determination")
        
        if len(sorted_lengths) < 5:
            warnings.append("Limited context length range tested")
            quality_score *= 0.8
        
        return EffectiveContextAnalysis(
            effective_context_length=effective_length,
            confidence_interval=confidence_interval,
            error_threshold_used=error_threshold,
            success_rate_at_effective_length=success_rate,
            recommended_safe_length=recommended_safe_length,
            analysis_method=method_used,
            quality_score=quality_score,
            warnings=warnings
        )
    
    def _find_effective_length_threshold(
        self,
        results: Dict[int, ContextLengthResults],
        threshold: float
    ) -> Optional[int]:
        """Find effective length using simple threshold method."""
        sorted_lengths = sorted(results.keys(), reverse=True)
        
        for length in sorted_lengths:
            error_rate = 1 - results[length].success_rate
            if error_rate <= threshold:
                return length
        
        return None
    
    def _find_effective_length_statistical(
        self,
        results: Dict[int, ContextLengthResults],
        threshold: float
    ) -> Optional[int]:
        """Find effective length using statistical method with confidence intervals."""
        if not SCIPY_AVAILABLE:
            return None
        
        sorted_lengths = sorted(results.keys(), reverse=True)
        
        for length in sorted_lengths:
            result = results[length]
            n_samples = len(result.results)
            n_successes = result.success_count
            
            # Calculate confidence interval for success rate
            if n_samples > 0:
                success_rate = n_successes / n_samples
                
                # Wilson score interval (more accurate for small samples)
                z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
                denominator = 1 + z**2 / n_samples
                centre_adjusted_probability = (success_rate + z**2 / (2 * n_samples)) / denominator
                adjusted_standard_deviation = math.sqrt(
                    (success_rate * (1 - success_rate) + z**2 / (4 * n_samples)) / n_samples
                ) / denominator
                
                lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
                
                # Check if lower bound of success rate is above threshold
                if lower_bound >= (1 - threshold):
                    return length
        
        return None
    
    def _select_best_method_result(
        self,
        methods_results: Dict[str, Optional[int]],
        results: Dict[int, ContextLengthResults]
    ) -> Tuple[Optional[int], str, float]:
        """Select the best result from multiple methods."""
        
        # Remove None results
        valid_results = {method: length for method, length in methods_results.items() if length is not None}
        
        if not valid_results:
            return None, "no_valid_results", 0.0
        
        # If only one method has results, use it
        if len(valid_results) == 1:
            method, length = list(valid_results.items())[0]
            quality_score = 0.6  # Medium confidence for single method
            return length, method, quality_score
        
        # If multiple methods agree (within 10%), high confidence
        lengths = list(valid_results.values())
        if len(set(lengths)) == 1:
            # All methods agree
            return lengths[0], "consensus", 0.95
        
        # Check if results are close
        min_length, max_length = min(lengths), max(lengths)
        if (max_length - min_length) / min_length <= 0.1:  # Within 10%
            # Use conservative estimate
            conservative_length = min(lengths)
            return conservative_length, "conservative_consensus", 0.85
        
        # Methods disagree significantly, use most conservative
        conservative_length = min(lengths)
        return conservative_length, "conservative_fallback", 0.5
    
    def _calculate_confidence_interval(
        self,
        results: Dict[int, ContextLengthResults],
        effective_length: int
    ) -> Optional[ConfidenceInterval]:
        """Calculate confidence interval for effective context length."""
        if not SCIPY_AVAILABLE or effective_length not in results:
            return None
        
        result = results[effective_length]
        n_samples = len(result.results)
        n_successes = result.success_count
        
        if n_samples == 0:
            return None
        
        # Use Wilson score interval
        success_rate = n_successes / n_samples
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        denominator = 1 + z**2 / n_samples
        centre_adjusted_probability = (success_rate + z**2 / (2 * n_samples)) / denominator
        adjusted_standard_deviation = math.sqrt(
            (success_rate * (1 - success_rate) + z**2 / (4 * n_samples)) / n_samples
        ) / denominator
        
        lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
        upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
        
        # Convert success rate bounds to context length bounds (approximate)
        # This is a simplified approach - in practice, you'd want more sophisticated modeling
        length_margin = effective_length * 0.05  # 5% margin
        
        return ConfidenceInterval(
            lower=max(0, effective_length - length_margin),
            upper=effective_length + length_margin,
            confidence_level=self.confidence_level,
            method="wilson_score"
        )
    
    def analyze_performance_metrics(self, session: TestSession) -> PerformanceMetrics:
        """Analyze performance metrics from test results."""
        all_response_times = []
        successful_response_times = []
        
        for context_result in session.context_results.values():
            for result in context_result.results:
                all_response_times.append(result.response_time_ms)
                if result.success:
                    successful_response_times.append(result.response_time_ms)
        
        if not successful_response_times:
            # No successful requests
            return PerformanceMetrics(
                average_latency_ms=0,
                median_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                requests_per_second=0,
                total_test_duration=0
            )
        
        # Calculate percentiles
        if SCIPY_AVAILABLE and len(successful_response_times) > 1:
            p95 = np.percentile(successful_response_times, 95)
            p99 = np.percentile(successful_response_times, 99)
        else:
            # Fallback calculation
            sorted_times = sorted(successful_response_times)
            n = len(sorted_times)
            p95 = sorted_times[int(0.95 * n)] if n > 0 else 0
            p99 = sorted_times[int(0.99 * n)] if n > 0 else 0
        
        # Calculate duration
        duration = 0
        if session.end_time and session.start_time:
            duration = (session.end_time - session.start_time).total_seconds()
        
        # Calculate requests per second
        rps = 0
        if duration > 0:
            rps = session.total_requests / duration
        
        return PerformanceMetrics(
            average_latency_ms=statistics.mean(successful_response_times),
            median_latency_ms=statistics.median(successful_response_times),
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            min_latency_ms=min(successful_response_times),
            max_latency_ms=max(successful_response_times),
            requests_per_second=rps,
            total_test_duration=duration
        )
    
    def analyze_errors(self, context_results: Dict[int, ContextLengthResults]) -> ErrorAnalysis:
        """Analyze error patterns from test results."""
        total_errors = 0
        total_requests = 0
        error_types = {}
        error_rate_by_length = {}
        context_length_error_threshold = None
        
        for length, result in context_results.items():
            length_errors = result.failure_count
            length_requests = len(result.results)
            
            total_errors += length_errors
            total_requests += length_requests
            
            # Error rate for this context length
            error_rate_by_length[length] = length_errors / length_requests if length_requests > 0 else 0
            
            # Aggregate error types
            for error_type, count in result.error_types.items():
                error_types[error_type] = error_types.get(error_type, 0) + count
        
        # Find first context length where errors exceed 10%
        sorted_lengths = sorted(context_results.keys())
        for length in sorted_lengths:
            if error_rate_by_length.get(length, 0) > 0.1:
                context_length_error_threshold = length
                break
        
        overall_error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        return ErrorAnalysis(
            total_errors=total_errors,
            error_rate=overall_error_rate,
            error_types=error_types,
            error_rate_by_context_length=error_rate_by_length,
            context_length_error_threshold=context_length_error_threshold
        )
    
    def assess_data_quality(
        self,
        session: TestSession,
        min_samples: int = 3
    ) -> DataQualityAssessment:
        """Assess the quality and reliability of test data."""
        context_results = session.context_results
        
        if not context_results:
            return DataQualityAssessment(
                total_data_points=0,
                context_lengths_tested=0,
                samples_per_length_stats={},
                coverage_score=0.0,
                consistency_score=0.0,
                reliability_warnings=["No test data available"]
            )
        
        # Basic statistics
        total_data_points = sum(len(result.results) for result in context_results.values())
        context_lengths_tested = len(context_results)
        
        # Samples per length statistics
        samples_per_length = [len(result.results) for result in context_results.values()]
        samples_stats = {
            'min': min(samples_per_length),
            'max': max(samples_per_length),
            'mean': statistics.mean(samples_per_length),
            'median': statistics.median(samples_per_length)
        }
        
        # Coverage score (how well the range was covered)
        coverage_score = self._calculate_coverage_score(context_results, session.configuration)
        
        # Consistency score (how consistent the results are)
        consistency_score = self._calculate_consistency_score(context_results)
        
        # Generate reliability warnings
        warnings = []
        
        if samples_stats['min'] < min_samples:
            warnings.append(f"Some context lengths have fewer than {min_samples} samples")
        
        if coverage_score < 0.5:
            warnings.append("Poor coverage of the context length range")
        
        if consistency_score < 0.7:
            warnings.append("Inconsistent results detected - consider more samples")
        
        if context_lengths_tested < 5:
            warnings.append("Limited number of context lengths tested")
        
        return DataQualityAssessment(
            total_data_points=total_data_points,
            context_lengths_tested=context_lengths_tested,
            samples_per_length_stats=samples_stats,
            coverage_score=coverage_score,
            consistency_score=consistency_score,
            reliability_warnings=warnings
        )
    
    def _calculate_coverage_score(
        self,
        context_results: Dict[int, ContextLengthResults],
        config: Dict[str, Any]
    ) -> float:
        """Calculate how well the context length range was covered."""
        if not context_results:
            return 0.0
        
        min_tested = min(context_results.keys())
        max_tested = max(context_results.keys())
        
        min_target = config.get('min_tokens', 1000)
        max_target = config.get('max_tokens', 200000)
        
        # Calculate coverage as percentage of target range covered
        target_range = max_target - min_target
        tested_range = max_tested - min_tested
        
        if target_range <= 0:
            return 1.0
        
        coverage = min(1.0, tested_range / target_range)
        
        # Bonus for testing near the boundaries
        boundary_bonus = 0.0
        if abs(min_tested - min_target) / min_target < 0.1:  # Within 10% of min
            boundary_bonus += 0.1
        if abs(max_tested - max_target) / max_target < 0.1:  # Within 10% of max
            boundary_bonus += 0.1
        
        return min(1.0, coverage + boundary_bonus)
    
    def _calculate_consistency_score(self, context_results: Dict[int, ContextLengthResults]) -> float:
        """Calculate consistency score based on result variability."""
        if len(context_results) < 2:
            return 1.0
        
        # Look for smooth transitions in success rates
        sorted_lengths = sorted(context_results.keys())
        success_rates = [context_results[length].success_rate for length in sorted_lengths]
        
        # Calculate how "smooth" the success rate curve is
        # Penalize large jumps in success rate between adjacent context lengths
        consistency_penalties = []
        
        for i in range(1, len(success_rates)):
            rate_change = abs(success_rates[i] - success_rates[i-1])
            # Expect success rates to generally decrease with context length
            # Large increases are suspicious
            if success_rates[i] > success_rates[i-1]:
                consistency_penalties.append(rate_change * 2)  # Double penalty for increases
            else:
                consistency_penalties.append(rate_change)
        
        if not consistency_penalties:
            return 1.0
        
        avg_penalty = statistics.mean(consistency_penalties)
        consistency_score = max(0.0, 1.0 - avg_penalty * 2)  # Scale penalty
        
        return consistency_score
    
    def _generate_summary(
        self,
        session: TestSession,
        effective_context: EffectiveContextAnalysis,
        performance: PerformanceMetrics,
        errors: ErrorAnalysis
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'test_strategy': session.strategy.value,
            'total_requests': session.total_requests,
            'successful_requests': session.successful_requests,
            'failed_requests': session.failed_requests,
            'overall_success_rate': session.successful_requests / session.total_requests if session.total_requests > 0 else 0,
            'effective_context_length': effective_context.effective_context_length,
            'recommended_safe_length': effective_context.recommended_safe_length,
            'average_response_time_ms': performance.average_latency_ms,
            'total_test_duration_seconds': performance.total_test_duration,
            'context_lengths_tested': len(session.context_results),
            'error_rate': errors.error_rate,
            'primary_error_type': max(errors.error_types.items(), key=lambda x: x[1])[0].value if errors.error_types else None
        }
    
    def _generate_recommendations(
        self,
        effective_context: EffectiveContextAnalysis,
        performance: PerformanceMetrics,
        errors: ErrorAnalysis,
        data_quality: DataQualityAssessment
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Context length recommendations
        if effective_context.effective_context_length:
            recommendations.append(
                f"Effective context length determined: {format_number(effective_context.effective_context_length)} tokens"
            )
            if effective_context.recommended_safe_length:
                recommendations.append(
                    f"Recommended safe operating limit: {format_number(effective_context.recommended_safe_length)} tokens "
                    f"(10% buffer below effective limit)"
                )
        else:
            recommendations.append("Could not determine effective context length - consider testing with more samples or different range")
        
        # Quality recommendations
        if effective_context.quality_score < 0.7:
            recommendations.append("Low confidence in results - consider running additional tests with more samples")
        
        # Performance recommendations
        if performance.average_latency_ms > 10000:  # 10 seconds
            recommendations.append("High average response times detected - consider implementing timeout handling")
        
        if performance.p95_latency_ms > performance.average_latency_ms * 3:
            recommendations.append("High latency variability detected - implement retry logic for timeouts")
        
        # Error recommendations
        if errors.error_rate > 0.05:  # 5% error rate
            recommendations.append(f"High overall error rate ({errors.error_rate:.1%}) - investigate error causes")
        
        if ErrorType.CONTEXT_LENGTH_EXCEEDED in errors.error_types:
            context_errors = errors.error_types[ErrorType.CONTEXT_LENGTH_EXCEEDED]
            recommendations.append(f"Context length errors detected ({context_errors} occurrences) - effective limit reached")
        
        if ErrorType.RATE_LIMIT in errors.error_types:
            recommendations.append("Rate limiting detected - consider reducing request rate or implementing backoff")
        
        # Data quality recommendations
        if data_quality.coverage_score < 0.7:
            recommendations.append("Limited range coverage - consider testing broader context length range")
        
        if data_quality.consistency_score < 0.7:
            recommendations.append("Inconsistent results detected - run additional samples for better reliability")
        
        if data_quality.samples_per_length_stats.get('min', 0) < 3:
            recommendations.append("Some context lengths have insufficient samples - increase samples per size")
        
        return recommendations
    
    def export_detailed_results(
        self,
        analysis: ComprehensiveAnalysis,
        format_type: str = "dict"
    ) -> Dict[str, Any]:
        """Export detailed analysis results in structured format."""
        
        def serialize_dataclass(obj):
            """Helper to serialize dataclasses to dict."""
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for field_name, field_value in obj.__dict__.items():
                    if isinstance(field_value, dict):
                        # Handle dict with enum keys
                        serialized_dict = {}
                        for k, v in field_value.items():
                            key = k.value if hasattr(k, 'value') else str(k)
                            serialized_dict[key] = serialize_dataclass(v) if hasattr(v, '__dataclass_fields__') else v
                        result[field_name] = serialized_dict
                    elif isinstance(field_value, list):
                        result[field_name] = [serialize_dataclass(item) if hasattr(item, '__dataclass_fields__') else item for item in field_value]
                    elif hasattr(field_value, '__dataclass_fields__'):
                        result[field_name] = serialize_dataclass(field_value)
                    elif hasattr(field_value, 'value'):  # Enum
                        result[field_name] = field_value.value
                    else:
                        result[field_name] = field_value
                return result
            return obj
        
        return serialize_dataclass(analysis)