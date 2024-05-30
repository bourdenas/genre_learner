import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import tensorflow as tf

from argparse import ArgumentParser
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
global model


@app.route('/genres', methods=['POST'])
def handle_genres():
    try:
        json = request.get_json()
        req = GenresRequest(**json)

        X = features.build_array(
            igdb_genres=[f'IGDB_{e}' for e in req.igdb_genres],
            steam_tags=[f'STEAM_{e}' for e in req.steam_tags]
        )
        Y = model(X)
        genres = labels.labels(Y[0])

        resp = GenresResponse(req.id, req.name, espy_genres=genres)
        return jsonify(resp)
    except Exception as e:
        error_message = {'error': str(e)}
        return jsonify(error_message), 500


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Flask app for serving the genres model predictions.")
    parser.add_argument(
        '--model', help='Filepath to the model used for serving.')
    parser.add_argument(
        '--port', help='Port number to listen for requests. (default: 5000)', type=int, default=5000)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    model.summary()

    app.run(debug=True, port=args.port)
