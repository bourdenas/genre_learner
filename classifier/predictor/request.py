from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from typing import List


@dataclass_json
@dataclass
class GenresRequest:
    id: int
    name: str

    igdb_genres: List[str] = field(default_factory=list)
    steam_genres: List[str] = field(default_factory=list)
    gog_genres: List[str] = field(default_factory=list)
    wiki_genres: List[str] = field(default_factory=list)

    igdb_tags: List[str] = field(default_factory=list)
    steam_tags: List[str] = field(default_factory=list)
    gog_tags: List[str] = field(default_factory=list)
    wiki_tags: List[str] = field(default_factory=list)

    description: str = field(default='')

    debug: bool = field(default=False)
