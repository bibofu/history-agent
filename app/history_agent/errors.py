class HistoryAgentError(Exception):
    """Base class for expected project errors."""


class ConfigurationError(HistoryAgentError):
    """Raised when required configuration is missing or invalid."""


class CatalogError(HistoryAgentError):
    """Raised when the corpus catalog is invalid."""


class CorpusScanError(HistoryAgentError):
    """Raised when a corpus scan cannot be completed safely."""


class ExtractionError(HistoryAgentError):
    """Raised when a PDF page cannot be processed."""


class IndexBuildError(HistoryAgentError):
    """Raised when a retrieval index cannot be built safely."""


class RetrievalError(HistoryAgentError):
    """Raised when a retrieval request is invalid or cannot be executed."""


class ResearchDataError(HistoryAgentError):
    """Raised when structured research data violates an auditable constraint."""
