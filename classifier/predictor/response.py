from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from typing import List

from classifier.predictor.debug_info import DebugInfo


@dataclass_json
@dataclass
class GenresResponse:
    id: int
    name: str
    espy_genres: List[str] = field(default_factory=list)
    debug_info: DebugInfo = field(default_factory=dict)
