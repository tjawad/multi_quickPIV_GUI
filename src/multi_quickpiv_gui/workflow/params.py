"""Parameter models for GUI and processing workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

CorrAlg = str
BackgroundFilterLevel = str
SizeND = tuple[int, ...]

BACKGROUND_FILTER_LEVELS = {
    "Off",
    "Low",
    "Medium",
    "High",
    "Very High",
}

@dataclass(slots=True)
class PIVRunParams:
    """Parameters that directly control the PIV computation."""

    inter_size: SizeND = (64, 64)
    search_margin: SizeND = (128, 128)
    step: SizeND = (32, 32)
    compute_sn: bool = True
    corr_alg: CorrAlg = "nsqecc"
    background_filter: BackgroundFilterLevel = "Off"
    downsample_factor: SizeND = (1, 1)

    def validate(self) -> None:
        """Validate the full run configuration."""
        self._validate_size("inter_size", self.inter_size)
        self._validate_size("search_margin", self.search_margin)
        self._validate_size("step", self.step)
        self._validate_background_filter()
        self._validate_downsample_factor()

    @staticmethod
    def _validate_size(name: str, value: SizeND) -> None:
        """Validate a 2D or 3D integer size tuple."""
        if len(value) not in {2, 3}:
            raise ValueError(f"{name} must contain exactly 2 or 3 integers.")

        if any(component <= 0 for component in value):
            raise ValueError(f"{name} values must be greater than 0.")

    def _validate_background_filter(self) -> None:
        """Validate the low-signal/background filter level."""
        if self.background_filter not in BACKGROUND_FILTER_LEVELS:
            allowed = ", ".join(sorted(BACKGROUND_FILTER_LEVELS))
            raise ValueError(
                f"Background filter must be one of: {allowed}."
            )

    def _validate_downsample_factor(self) -> None:
        """Validate the pre-PIV downsampling factors."""
        if len(self.downsample_factor) not in {2, 3}:
            raise ValueError(
                "Downsampling must contain exactly 2 or 3 integer factors."
            )

        if any(component < 1 for component in self.downsample_factor):
            raise ValueError("Downsampling factors must be at least 1.")


@dataclass(slots=True)
class MedianDespikeParams:
    """Configuration for median-despike post-processing."""

    enabled: bool = False
    ksize: int = 3
    threshold: float = 3.5

    def validate(self) -> None:
        """Validate median-despike settings."""
        if self.ksize < 1:
            raise ValueError("Median-despike window size must be at least 1.")
        if self.ksize % 2 == 0:
            raise ValueError("Median-despike window size must be odd.")
        if self.threshold <= 0:
            raise ValueError("Median-despike threshold must be greater than 0.")


@dataclass(slots=True)
class SNFilterParams:
    """Configuration for signal-to-noise post-processing."""

    enabled: bool = False
    minimum: float = 1.0

    def validate(self, *, compute_sn: bool) -> None:
        """Validate signal-to-noise filtering settings."""
        if self.enabled and not compute_sn:
            raise ValueError("SN filtering requires compute_sn=True.")
        if self.enabled and self.minimum <= 0:
            raise ValueError("SN minimum threshold must be greater than 0.")


@dataclass(slots=True)
class SpatioTemporalAverageParams:
    """Configuration for backend spatio-temporal averaging."""

    enabled: bool = False
    spatial_radius: int = 1
    temporal_radius: int = 0

    def validate(self) -> None:
        """Validate backend averaging settings."""
        if self.spatial_radius < 0:
            raise ValueError("Spatio-temporal averaging spatial radius must be at least 0.")
        if self.temporal_radius < 0:
            raise ValueError("Spatio-temporal averaging temporal radius must be at least 0.")


@dataclass(slots=True)
class PostProcessParams:
    """Collection of post-processing settings for a computed vector field."""

    median_despike: MedianDespikeParams = field(
        default_factory=MedianDespikeParams
    )
    sn_filter: SNFilterParams = field(default_factory=SNFilterParams)
    spatiotemporal_average: SpatioTemporalAverageParams = field(
        default_factory=SpatioTemporalAverageParams
    )

    def validate(self, *, run: PIVRunParams) -> None:
        """Validate all post-processing settings."""
        self.median_despike.validate()
        self.sn_filter.validate(compute_sn=run.compute_sn)
        self.spatiotemporal_average.validate()


@dataclass(slots=True)
class WorkflowParams:
    """Complete workflow configuration for one analysis run."""

    run: PIVRunParams = field(default_factory=PIVRunParams)
    postprocess: PostProcessParams = field(default_factory=PostProcessParams)

    def validate(self) -> None:
        """Validate the complete workflow configuration."""
        self.run.validate()
        self.postprocess.validate(run=self.run)