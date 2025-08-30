"""
Settings manager.
"""

import enum

# pylint: disable=no-name-in-module
from pathlib import Path

from pydantic.dataclasses import dataclass


class Metrics(enum.Enum):
    """
    Metrics enum.
    """

    BLEU = "bleu"
    ROUGE = "rouge"
    SQUAD = "squad"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    ACCURACY = "accuracy"

    def __str__(self) -> str:
        """
        String representation of a metric.

        Returns:
             str: Name of a metric
        """
        return self.value


@dataclass
class ParametersModel:
    """
    Additional parameters of a lab.
    """

    model: str
    dataset: str
    metrics: list[Metrics]


@dataclass
class InferenceParams:
    """
    Inference parameters.
    """

    num_samples: int
    max_length: int
    batch_size: int
    predictions_path: Path
    device: str


@dataclass
class SFTParams:
    """
    Fine-tuning parameters.
    """

    max_length: int
    batch_size: int
    max_fine_tuning_steps: int
    device: str
    finetuned_model_path: Path
    learning_rate: float
    target_modules: list[str] | None = None


@dataclass
class CtlrParameters:
    """
    DTO for storing computer tools course parameters.
    """

    project_team: int | None = None


@dataclass
class CourseParameters:
    """
    DTO for storing specific parameters for courses.
    """

    ctlr: CtlrParameters | None = None
    hello_llm: ParametersModel | None = None


@dataclass
class LabSettingsModel:
    """
    DTO for storing labs settings.
    """

    target_score: int
    parameters: CourseParameters | None = None


class LabSettings:
    """
    Main model for working with settings.
    """

    # Labs settings
    _dto: LabSettingsModel

    def __init__(self, config_path: Path) -> None:
        """
        Initialize LabSettings.

        Args:
            config_path (pathlib.Path): Path to configuration
        """
        super().__init__()
        with config_path.open(encoding="utf-8") as config_file:
            # pylint: disable=no-member
            self._dto = LabSettingsModel.__pydantic_validator__.validate_json(config_file.read())

    @property
    def target_score(self) -> int:
        """
        Property for target score.

        Returns:
            int: A target score.
        """
        return self._dto.target_score

    @property
    def parameters(self) -> CourseParameters | None:
        """
        Property for additional parameters.

        Returns:
            ParametersModel | None: Parameters DTO.
        """
        return self._dto.parameters

    @property
    def team_project(self) -> int | None:
        """
        Property for project team id (Computer Tools Final Project).

        Returns:
            int | None: Project team identifier.
        """
        return self._dto.parameters.ctlr.project_team
