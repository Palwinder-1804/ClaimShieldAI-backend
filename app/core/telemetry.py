import logging
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

logger = logging.getLogger(__name__)

# Configure tracing resource
resource = Resource(attributes={
    SERVICE_NAME: "claimshield-backend"
})

# Initialize TracerProvider
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Always add a Console exporter in development/logs
try:
    console_processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(console_processor)
except Exception as e:
    logger.warning(f"Could not initialize console span exporter: {e}")

# We can also add OTLP Exporters (e.g., to Jaeger, collector) if needed
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from app.core.config import settings
    # Points to Prometheus/Grafana agent, or otel collector default port 4317
    otlp_exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
    otlp_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(otlp_processor)
    logger.info(f"OTLP Trace exporter registered successfully to endpoint: {settings.OTEL_EXPORTER_OTLP_ENDPOINT}")
except ImportError:
    logger.info("OTLP Exporter package not installed, skipping GRPC trace exporter.")
except Exception as e:
    logger.warning(f"Could not initialize OTLP trace exporter: {e}")

# Global tracer instance
claim_tracer = trace.get_tracer("claimshield-tracer")

def setup_telemetry(app: FastAPI) -> None:
    """
    Instruments the FastAPI application for OpenTelemetry.
    """
    try:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("FastAPI successfully instrumented with OpenTelemetry.")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI application: {e}")
