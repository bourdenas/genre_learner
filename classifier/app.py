from flask import Flask, request, jsonify
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List
from dataset.espy import Features, Labels

app = Flask(__name__)


@dataclass_json
@dataclass
class GenresRequest:
    id: int
    name: str
    igdb_genres: List[str] = field(default_factory=list)
    steam_tags: List[str] = field(default_factory=list)


@dataclass_json
@dataclass
class GenresResponse:
    id: int
    name: str
    espy_genres: List[str] = field(default_factory=list)


features = Features.load()
labels = Labels.load()


@app.route('/genres', methods=['POST'])
def handle_genres():
    try:
        json = request.get_json()
        req = GenresRequest(**json)

        result = predict(req)
        return jsonify(result)
    except Exception as e:
        error_message = {'error': str(e)}
        return jsonify(error_message), 500


def predict(req: GenresRequest):
    return GenresResponse(req.id, req.name, espy_genres=['CRPG'])


if __name__ == '__main__':
    app.run(debug=True)
