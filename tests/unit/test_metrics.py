"""Tests for OTel Metrics (MetricsCollector)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent_governor.observability.metrics import OTEL_METRICS_AVAILABLE, MetricsCollector


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_init_noop_when_otel_not_available(self):
        """MetricsCollector should not crash when OTel is unavailable."""
        with patch("agent_governor.observability.metrics.OTEL_METRICS_AVAILABLE", False):
            mc = MetricsCollector()
            mc.record_call("gpt-4", "success")  # Should not raise
            mc.record_error("gpt-4", "timeout")  # Should not raise
            mc.record_latency("gpt-4", 100.0)  # Should not raise
            mc.record_tokens("gpt-4", 500)  # Should not raise
            mc.record_budget_exceeded()  # Should not raise
            mc.shutdown()  # Should not raise

    def test_init_with_empty_endpoint(self):
        """MetricsCollector should handle empty endpoint."""
        mc = MetricsCollector(otlp_endpoint="")
        assert mc is not None
        mc.shutdown()

    def test_shutdown_noop_when_not_initialized(self):
        """Shutdown without init should not raise."""
        MetricsCollector().shutdown()

    def test_record_call_creates_counter(self):
        """record_call should use a Counter instrument."""
        mock_counter = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        with (
            patch("agent_governor.observability.metrics.OTEL_METRICS_AVAILABLE", True),
            patch.object(MetricsCollector, "_init"),
        ):
            mc = MetricsCollector()
            mc._meter = mock_meter
            mc.record_call("gpt-4", "success")

        mock_meter.create_counter.assert_called_once()
        mock_counter.add.assert_called_once_with(1, {"model": "gpt-4", "status": "success"})

    def test_record_latency_creates_histogram(self):
        """record_latency should use a Histogram instrument."""
        mock_histogram = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_histogram.return_value = mock_histogram

        with (
            patch("agent_governor.observability.metrics.OTEL_METRICS_AVAILABLE", True),
            patch.object(MetricsCollector, "_init"),
        ):
            mc = MetricsCollector()
            mc._meter = mock_meter
            mc.record_latency("gpt-4", 150.5)

        mock_meter.create_histogram.assert_called_once()
        mock_histogram.record.assert_called_once_with(150.5, {"model": "gpt-4"})

    def test_records_are_idempotent(self):
        """Calling record methods multiple times should work."""
        with patch("agent_governor.observability.metrics.OTEL_METRICS_AVAILABLE", False):
            mc = MetricsCollector()
            for _ in range(3):
                mc.record_call("gpt-4", "success")  # No error
                mc.record_error("gpt-4", "timeout")  # No error


class TestOTELMetricsAvailability:
    """Tests for OTEL_METRICS_AVAILABLE flag."""

    def test_otel_metrics_availability_flag_exists(self):
        """OTEL_METRICS_AVAILABLE should be defined as bool."""
        assert isinstance(OTEL_METRICS_AVAILABLE, bool)
