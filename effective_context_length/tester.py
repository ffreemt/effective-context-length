"""
Effective Context Length Tester

This module provides the main testing logic for determining effective context lengths
of OpenAI-compatible LLM endpoints using various testing strategies.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import statistics
from datetime import datetime

from .http_client import HTTPClient, APIResponse, ErrorType
from .context_generator import ContextGenerator, ContextStrategy
from .utils import ProgressReporter, Timer, format_duration, format_number


class TestStrategy(Enum):
    """Available testing strategies."""
    HYBRID = "hybrid"
    GRADUAL = "gradual"
    BINARY_SEARCH = "binary-search"
    PREDEFINED = "predefined"


@dataclass
class TestResult:
    """
    Individual test result with comprehensive metrics.
    
    Represents a single API request attempt with detailed timing,
    token usage, and error information for context length testing.
    
    Attributes:
        context_length: Number of tokens sent in the request
        success: Whether the request completed successfully
        response_time_ms: Response time in milliseconds
        tokens_sent: Number of tokens in the request payload
        tokens_received: Number of tokens in the response
        error_type: Categorized error type if request failed
        error_message: Detailed error message if request failed
        timestamp: When the test was performed
        attempt_number: Which attempt this was (for retry logic)
    """
    context_length: int
    success: bool
    response_time_ms: float
    tokens_sent: Optional[int] = None
    tokens_received: Optional[int] = None
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    attempt_number: int = 1


@dataclass
class ContextLengthResults:
    """
    Aggregated results for tests at a specific context length.
    
    Contains all test results for a particular context length along
    with calculated statistics including success rate, average response time,
    and error type distribution.
    
    Attributes:
        context_length: The context length these results represent
        results: List of individual test results
        success_count: Number of successful requests
        failure_count: Number of failed requests
        success_rate: Calculated success rate (0.0 to 1.0)
        average_response_time: Average response time in milliseconds
        error_types: Dictionary mapping error types to occurrence counts
    """
    context_length: int
    results: List[TestResult] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    error_types: Dict[ErrorType, int] = field(default_factory=dict)


@dataclass
class TestSession:
    """
    Complete test session with all results and metadata.
    
    Represents a full testing session including configuration, results,
    timing information, and calculated statistics. This is the primary
    data structure returned by testing operations.
    
    Attributes:
        base_url: API endpoint that was tested
        model: Model name that was tested
        strategy: Testing strategy used
        start_time: When the test session started
        end_time: When the test session completed (None if in progress)
        results: List of aggregated results by context length
        context_results: Dictionary mapping context lengths to results (for compatibility)
        total_requests: Total number of API requests made
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        configuration: Copy of the configuration used
        metadata: Additional metadata about the test session
    """
    base_url: str
    model: str
    strategy: TestStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[ContextLengthResults] = field(default_factory=list)
    context_results: Dict[int, ContextLengthResults] = field(default_factory=dict)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestStrategyBase(ABC):
    """Base class for testing strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def generate_test_plan(self) -> List[int]:
        """Generate list of context lengths to test."""
        pass
    
    @abstractmethod
    def should_continue_testing(self, results: Dict[int, ContextLengthResults]) -> bool:
        """Determine if testing should continue based on current results."""
        pass


class GradualStrategy(TestStrategyBase):
    """Gradual testing strategy - incremental increases with configurable steps."""
    
    async def generate_test_plan(self) -> List[int]:
        """Generate incremental test plan."""
        min_tokens = self.config.get('min_tokens', 1000)
        max_tokens = self.config.get('max_tokens', 200000)
        step_size = self.config.get('step_size', 2000)
        
        test_lengths = []
        current = min_tokens
        
        while current <= max_tokens:
            test_lengths.append(current)
            current += step_size
        
        self.logger.info(f"Generated gradual test plan: {len(test_lengths)} context lengths "
                        f"from {min_tokens:,} to {max_tokens:,} tokens")
        return test_lengths
    
    def should_continue_testing(self, results: Dict[int, ContextLengthResults]) -> bool:
        """Continue until error threshold is exceeded or max tokens reached."""
        error_threshold = self.config.get('error_threshold', 0.1)
        
        # Get the latest results
        if not results:
            return True
        
        latest_length = max(results.keys())
        latest_result = results[latest_length]
        
        # Stop if error rate exceeds threshold
        if latest_result.success_rate < (1 - error_threshold):
            self.logger.info(f"Stopping gradual test: error rate {1 - latest_result.success_rate:.2%} "
                           f"exceeds threshold {error_threshold:.2%} at {latest_length:,} tokens")
            return False
        
        return True


class BinarySearchStrategy(TestStrategyBase):
    """Binary search strategy - efficient logarithmic approach."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.search_bounds = [config.get('min_tokens', 1000), config.get('max_tokens', 200000)]
        self.tested_lengths = set()
    
    async def generate_test_plan(self) -> List[int]:
        """Generate binary search test plan."""
        # Start with initial bounds test
        min_tokens, max_tokens = self.search_bounds
        initial_tests = [min_tokens, max_tokens]
        
        self.logger.info(f"Starting binary search between {min_tokens:,} and {max_tokens:,} tokens")
        return initial_tests
    
    def get_next_test_point(self, results: Dict[int, ContextLengthResults]) -> Optional[int]:
        """Get next test point based on binary search logic."""
        if not results:
            return None
        
        # Find current working bounds
        working_lengths = [length for length, result in results.items() 
                          if result.success_rate >= (1 - self.config.get('error_threshold', 0.1))]
        failing_lengths = [length for length, result in results.items() 
                          if result.success_rate < (1 - self.config.get('error_threshold', 0.1))]
        
        if not working_lengths and not failing_lengths:
            return None
        
        # Determine search bounds
        if working_lengths:
            lower_bound = max(working_lengths)
        else:
            lower_bound = self.search_bounds[0]
        
        if failing_lengths:
            upper_bound = min(failing_lengths)
        else:
            upper_bound = self.search_bounds[1]
        
        # Check if search is complete
        if upper_bound - lower_bound <= self.config.get('step_size', 2000):
            return None
        
        # Find midpoint
        midpoint = (lower_bound + upper_bound) // 2
        
        # Round to nearest step size for cleaner numbers
        step_size = self.config.get('step_size', 2000)
        midpoint = round(midpoint / step_size) * step_size
        
        # Avoid retesting
        if midpoint in self.tested_lengths:
            return None
        
        return midpoint
    
    def should_continue_testing(self, results: Dict[int, ContextLengthResults]) -> bool:
        """Continue until binary search converges."""
        next_point = self.get_next_test_point(results)
        if next_point:
            self.tested_lengths.add(next_point)
        return next_point is not None


class PredefinedStrategy(TestStrategyBase):
    """Predefined strategy - test user-specified context lengths."""
    
    async def generate_test_plan(self) -> List[int]:
        """Use predefined context lengths."""
        predefined_sizes = self.config.get('predefined_sizes', [])
        if not predefined_sizes:
            raise ValueError("Predefined strategy requires predefined_sizes configuration")
        
        # Sort and deduplicate
        test_lengths = sorted(set(predefined_sizes))
        
        self.logger.info(f"Using predefined test lengths: {test_lengths}")
        return test_lengths
    
    def should_continue_testing(self, results: Dict[int, ContextLengthResults]) -> bool:
        """Test all predefined lengths."""
        predefined_sizes = set(self.config.get('predefined_sizes', []))
        tested_sizes = set(results.keys())
        return not predefined_sizes.issubset(tested_sizes)


class HybridStrategy(TestStrategyBase):
    """Hybrid strategy - combines binary search, fine-tuning, and validation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.phase = "binary_search"  # binary_search -> fine_tuning -> validation
        self.binary_strategy = BinarySearchStrategy(config)
        self.effective_length = None
    
    async def generate_test_plan(self) -> List[int]:
        """Generate initial binary search plan."""
        return await self.binary_strategy.generate_test_plan()
    
    def should_continue_testing(self, results: Dict[int, ContextLengthResults]) -> bool:
        """Multi-phase testing logic."""
        error_threshold = self.config.get('error_threshold', 0.1)
        
        if self.phase == "binary_search":
            # Continue binary search until convergence
            if self.binary_strategy.should_continue_testing(results):
                return True
            else:
                # Move to fine-tuning phase
                self.phase = "fine_tuning"
                self.effective_length = self._estimate_effective_length(results)
                self.logger.info(f"Binary search complete. Moving to fine-tuning around {self.effective_length:,} tokens")
                return True
        
        elif self.phase == "fine_tuning":
            # Fine-tune around the estimated effective length
            if not self._fine_tuning_complete(results):
                return True
            else:
                # Move to validation phase
                self.phase = "validation"
                self.logger.info("Fine-tuning complete. Moving to validation phase")
                return True
        
        elif self.phase == "validation":
            # Validation phase - ensure we have enough samples at critical points
            return not self._validation_complete(results)
        
        return False
    
    def get_next_test_points(self, results: Dict[int, ContextLengthResults]) -> List[int]:
        """Get next test points based on current phase."""
        if self.phase == "binary_search":
            next_point = self.binary_strategy.get_next_test_point(results)
            return [next_point] if next_point else []
        
        elif self.phase == "fine_tuning":
            return self._get_fine_tuning_points(results)
        
        elif self.phase == "validation":
            return self._get_validation_points(results)
        
        return []
    
    def _estimate_effective_length(self, results: Dict[int, ContextLengthResults]) -> int:
        """Estimate effective length from binary search results."""
        error_threshold = self.config.get('error_threshold', 0.1)
        
        working_lengths = [length for length, result in results.items() 
                          if result.success_rate >= (1 - error_threshold)]
        
        return max(working_lengths) if working_lengths else self.config.get('min_tokens', 1000)
    
    def _get_fine_tuning_points(self, results: Dict[int, ContextLengthResults]) -> List[int]:
        """Get fine-tuning test points around effective length."""
        if not self.effective_length:
            return []
        
        step_size = self.config.get('step_size', 2000) // 2  # Smaller steps for fine-tuning
        points = []
        
        # Test points around the effective length
        for offset in [-2 * step_size, -step_size, step_size, 2 * step_size]:
            point = self.effective_length + offset
            if point > 0 and point not in results:
                points.append(point)
        
        return points[:2]  # Limit to 2 points at a time
    
    def _get_validation_points(self, results: Dict[int, ContextLengthResults]) -> List[int]:
        """Get validation test points for critical context lengths."""
        min_samples = self.config.get('min_samples', 3)
        
        # Find context lengths that need more samples
        for length, result in results.items():
            if len(result.results) < min_samples and result.success_rate > 0:
                return [length]
        
        return []
    
    def _fine_tuning_complete(self, results: Dict[int, ContextLengthResults]) -> bool:
        """Check if fine-tuning phase is complete."""
        if not self.effective_length:
            return True
        
        step_size = self.config.get('step_size', 2000) // 2
        required_points = [
            self.effective_length - step_size,
            self.effective_length + step_size
        ]
        
        tested_points = set(results.keys())
        return all(point in tested_points or point <= 0 for point in required_points)
    
    def _validation_complete(self, results: Dict[int, ContextLengthResults]) -> bool:
        """Check if validation phase is complete."""
        min_samples = self.config.get('min_samples', 3)
        
        # Check if all important context lengths have enough samples
        for length, result in results.items():
            if result.success_rate > 0 and len(result.results) < min_samples:
                return False
        
        return True


class EffectiveContextLengthTester:
    """
    Main tester class that orchestrates the testing process using different strategies.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        strategy: TestStrategy = TestStrategy.HYBRID,
        **config
    ):
        """
        Initialize the tester.
        
        Args:
            base_url: API endpoint base URL
            api_key: API authentication key
            model: Model name to test
            strategy: Testing strategy to use
            **config: Additional configuration parameters
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.strategy = strategy
        self.config = {
            'min_tokens': 1000,
            'max_tokens': 200000,
            'step_size': 2000,
            'samples_per_size': 5,
            'error_threshold': 0.1,
            'timeout': 300,
            'rate_limit': 1.0,
            'max_retries': 3,
            'temperature': 0.1,
            'min_samples': 3,
            **config
        }
        
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # Initialize components
        self.http_client = None
        self.context_generator = None
        self.test_strategy = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize HTTP client and other components."""
        self.http_client = HTTPClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.config['timeout'],
            max_retries=self.config['max_retries'],
            rate_limit=self.config['rate_limit'],
            model=self.model
        )
        
        self.context_generator = ContextGenerator(
            model=self.model,
            temperature=self.config['temperature']
        )
        
        # Initialize strategy
        if self.strategy == TestStrategy.GRADUAL:
            self.test_strategy = GradualStrategy(self.config)
        elif self.strategy == TestStrategy.BINARY_SEARCH:
            self.test_strategy = BinarySearchStrategy(self.config)
        elif self.strategy == TestStrategy.PREDEFINED:
            self.test_strategy = PredefinedStrategy(self.config)
        elif self.strategy == TestStrategy.HYBRID:
            self.test_strategy = HybridStrategy(self.config)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.logger.info(f"Initialized tester with {self.strategy.value} strategy")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.http_client:
            await self.http_client.close()
    
    async def run_test(self, progress_callback=None) -> TestSession:
        """
        Run the complete context length test.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            TestSession with complete results
        """
        self.logger.info(f"Starting context length test with {self.strategy.value} strategy")
        
        # Initialize session
        self.session = TestSession(
            base_url=self.base_url,
            model=self.model,
            strategy=self.strategy,
            start_time=datetime.now(),
            configuration=self.config.copy(),
            metadata={
                'base_url': self.base_url,
                'model': self.model,
                'tool_version': '1.0.0'
            }
        )
        
        try:
            # Health check
            if not await self._perform_health_check():
                raise RuntimeError("API health check failed")
            
            # Generate initial test plan
            test_lengths = await self.test_strategy.generate_test_plan()
            
            # Execute tests
            await self._execute_test_plan(test_lengths, progress_callback)
            
            # Continue with strategy-specific logic
            while self.test_strategy.should_continue_testing(self.session.context_results):
                next_points = self._get_next_test_points()
                if not next_points:
                    break
                
                await self._execute_test_plan(next_points, progress_callback)
            
            self.session.end_time = datetime.now()
            self._finalize_session_stats()
            
            duration = (self.session.end_time - self.session.start_time).total_seconds()
            self.logger.info(f"Test completed in {format_duration(duration)}")
            
            return self.session
            
        except Exception as e:
            self.logger.error(f"Test failed: {str(e)}")
            if self.session:
                self.session.end_time = datetime.now()
            raise
    
    async def _perform_health_check(self) -> bool:
        """Perform API health check."""
        self.logger.info("Performing API health check...")
        
        try:
            health_ok = await self.http_client.health_check()
            if health_ok:
                self.logger.info("✅ API health check passed")
                return True
            else:
                self.logger.error("❌ API health check failed")
                return False
        except Exception as e:
            self.logger.error(f"❌ API health check error: {str(e)}")
            return False
    
    async def _execute_test_plan(self, test_lengths: List[int], progress_callback=None):
        """Execute tests for the given context lengths."""
        total_tests = len(test_lengths) * self.config['samples_per_size']
        
        if progress_callback:
            progress_callback(0, total_tests, "Starting tests...")
        
        completed_tests = 0
        
        for context_length in test_lengths:
            self.logger.info(f"Testing context length: {context_length:,} tokens")
            
            # Initialize results for this context length
            if context_length not in self.session.context_results:
                new_result = ContextLengthResults(context_length=context_length)
                self.session.context_results[context_length] = new_result
                self.session.results.append(new_result)
            
            context_result = self.session.context_results[context_length]
            
            # Run multiple samples for this context length
            for sample in range(self.config['samples_per_size']):
                try:
                    result = await self._run_single_test(context_length, sample + 1)
                    context_result.results.append(result)
                    
                    if result.success:
                        context_result.success_count += 1
                        self.session.successful_requests += 1
                    else:
                        context_result.failure_count += 1
                        self.session.failed_requests += 1
                        
                        # Track error types
                        if result.error_type:
                            context_result.error_types[result.error_type] = \
                                context_result.error_types.get(result.error_type, 0) + 1
                    
                    self.session.total_requests += 1
                    completed_tests += 1
                    
                    if progress_callback:
                        progress_callback(completed_tests, total_tests,
                                        f"Context: {context_length:,} tokens, Sample: {sample + 1}")
                    
                except Exception as e:
                    self.logger.error(f"Test failed for {context_length:,} tokens, sample {sample + 1}: {str(e)}")
                    
                    # Create failed result
                    failed_result = TestResult(
                        context_length=context_length,
                        success=False,
                        response_time_ms=0,
                        error_type=ErrorType.UNKNOWN,
                        error_message=str(e),
                        attempt_number=sample + 1
                    )
                    
                    context_result.results.append(failed_result)
                    context_result.failure_count += 1
                    self.session.failed_requests += 1
                    self.session.total_requests += 1
                    completed_tests += 1
            
            # Calculate statistics for this context length
            self._calculate_context_stats(context_result)
            
            self.logger.info(f"Context {context_length:,} tokens: "
                           f"{context_result.success_rate:.1%} success rate, "
                           f"{context_result.average_response_time:.0f}ms avg response time")
    
    async def _run_single_test(self, context_length: int, attempt_number: int) -> TestResult:
        """Run a single test for the specified context length."""
        
        # Generate test payload
        payload = self.context_generator.generate_test_payload(
            target_tokens=context_length,
            strategy=ContextStrategy.PADDING,
            max_response_tokens=50
        )
        
        # Get actual token count
        token_count = self.context_generator.get_payload_token_count(payload)
        
        # Make API request
        with Timer(f"Request for {context_length:,} tokens") as timer:
            api_response = await self.http_client.make_chat_completion_request(
                payload=payload,
                tokens_sent=token_count.total_tokens
            )
        
        # Create test result
        result = TestResult(
            context_length=context_length,
            success=api_response.success,
            response_time_ms=api_response.response_time_ms,
            tokens_sent=api_response.tokens_sent,
            tokens_received=api_response.tokens_received,
            error_type=api_response.error_type,
            error_message=api_response.error_message,
            attempt_number=attempt_number
        )
        
        if result.success:
            self.logger.debug(f"✅ Success: {context_length:,} tokens in {result.response_time_ms:.0f}ms")
        else:
            self.logger.debug(f"❌ Failed: {context_length:,} tokens - {result.error_type.value if result.error_type else 'Unknown'}")
        
        return result
    
    def _get_next_test_points(self) -> List[int]:
        """Get next test points based on strategy."""
        if hasattr(self.test_strategy, 'get_next_test_points'):
            return self.test_strategy.get_next_test_points(self.session.context_results)
        elif hasattr(self.test_strategy, 'get_next_test_point'):
            next_point = self.test_strategy.get_next_test_point(self.session.context_results)
            return [next_point] if next_point else []
        else:
            return []
    
    def _calculate_context_stats(self, context_result: ContextLengthResults):
        """Calculate statistics for a context length result."""
        if not context_result.results:
            return
        
        total_results = len(context_result.results)
        context_result.success_rate = context_result.success_count / total_results
        
        # Calculate average response time for successful requests
        successful_times = [r.response_time_ms for r in context_result.results if r.success]
        if successful_times:
            context_result.average_response_time = statistics.mean(successful_times)
        else:
            context_result.average_response_time = 0.0
    
    def _finalize_session_stats(self):
        """Calculate final session statistics."""
        if not self.session:
            return
        
        # Overall success rate
        if self.session.total_requests > 0:
            self.session.metadata['overall_success_rate'] = \
                self.session.successful_requests / self.session.total_requests
        
        # Duration
        if self.session.end_time:
            duration = (self.session.end_time - self.session.start_time).total_seconds()
            self.session.metadata['duration_seconds'] = duration
        
        # Context length statistics
        tested_lengths = list(self.session.context_results.keys())
        if tested_lengths:
            self.session.metadata['min_context_length'] = min(tested_lengths)
            self.session.metadata['max_context_length'] = max(tested_lengths)
            self.session.metadata['total_context_lengths_tested'] = len(tested_lengths)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the test session."""
        if not self.session:
            return {}
        
        return {
            'strategy': self.session.strategy.value,
            'total_requests': self.session.total_requests,
            'successful_requests': self.session.successful_requests,
            'failed_requests': self.session.failed_requests,
            'success_rate': self.session.metadata.get('overall_success_rate', 0),
            'duration': self.session.metadata.get('duration_seconds', 0),
            'context_lengths_tested': len(self.session.context_results),
            'configuration': self.session.configuration
        }