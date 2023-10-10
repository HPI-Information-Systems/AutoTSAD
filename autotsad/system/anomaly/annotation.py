from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

ANNOTATION_CODES: Dict[str, str] = {
    "outlier": "E",
    "compress": "C",
    "stretch": "S",
    "noise": "N",
    "smoothing": "M",
    "hmirror": "H",
    "vmirror": "V",
    "scale": "A",
    "pattern": "P",
}


@dataclass(init=True, repr=True, eq=True, order=True, frozen=True)
class AnomalyAnnotation:
    position: int
    length: int
    anomaly_type: str
    strength: float
    display_idx: int

    def __sizeof__(self) -> int:
        # 4 * 32 bytes for 3 int and float members + size of anomaly_type string + 48 bytes for the object overhead
        return 4 * 32 + self.anomaly_type.__sizeof__() + 48

    @property
    def text(self) -> str:
        """Returns the annotation text.

        Returns
        -------
        str
            Annotation text.
        """
        return f"{self.anomaly_type}({self.strength:.2f})"

    def to_annotation_code(self) -> str:
        """Converts an annotation to its code string.

        Returns
        -------
        str
            Annotation code.
        """
        name = ANNOTATION_CODES[self.anomaly_type.lower().strip()]
        return f"{self.position}[{name}{self.strength*100:03.0f}({self.length})]"

    def adjust_position(self, delta: int) -> AnomalyAnnotation:
        """Adjusts the position of the annotation.

        Parameters
        ----------
        delta : int
            Adjustment amount. If negative, the position is moved to the left.

        Returns
        -------
        AnomalyAnnotation
            Adjusted annotation.
        """
        return AnomalyAnnotation(
            self.position + delta, self.length, self.anomaly_type, self.strength, self.display_idx + delta
        )


def encode_annotations(annotations: List[AnomalyAnnotation]) -> str:
    """Encode annotations as a short string.

    Parameters
    ----------
    annotations : List[AnomalyAnnotation]
        Annotations to encode.

    Returns
    -------
    str
        Encoded annotations.
    """
    values = sorted(annotations, key=lambda x: x.position)
    return "".join([a.to_annotation_code() for a in values])
