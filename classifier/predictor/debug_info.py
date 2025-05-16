from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass_json
@dataclass
class FeaturesInfo:
    external_genres: List[str] = field(default_factory=list)
    igdb_tags: List[str] = field(default_factory=list)
    steam_tags: List[str] = field(default_factory=list)
    gog_tags: List[str] = field(default_factory=list)
    wiki_tags: List[str] = field(default_factory=list)
    description_tags: List[str] = field(default_factory=list)


@dataclass_json
@dataclass
class PredictionInfo:
    genres: Dict[str, str] = field(default_factory=dict)


@dataclass_json
@dataclass
class DebugInfo:
    features: FeaturesInfo = field(default_factory=dict)
    predictions: PredictionInfo = field(default_factory=dict)
