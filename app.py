import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

from classifier.dataset.espy import Features, Labels
from typing import Dict, List
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from flask import Flask, request, jsonify
from argparse import ArgumentParser, BooleanOptionalAction
import tensorflow as tf


app = Flask(__name__)


@dataclass_json
@dataclass
class GenresRequest:
    id: int
    name: str
    igdb_genres: List[str] = field(default_factory=list)
    igdb_keywords: List[str] = field(default_factory=list)
    steam_genres: List[str] = field(default_factory=list)
    steam_tags: List[str] = field(default_factory=list)


@dataclass_json
@dataclass
class GenresDebugInfo:
    labels: Dict[str, str]


@dataclass_json
@dataclass
class GenresResponse:
    id: int
    name: str
    espy_genres: List[str] = field(default_factory=list)
    debug_info: GenresDebugInfo = field(default_factory=Dict)


features = Features.load()
labels = Labels.load()
model = None


def genres(filename):
    global model
    model = tf.keras.models.load_model(filename)
    return app


@app.route('/genres', methods=['POST'])
def handle_genres():
    try:
        json = request.get_json()
        req = GenresRequest(**json)

        X = features.build_array(
            igdb_genres=[f'IGDB_{e}' for e in req.igdb_genres],
            igdb_keywords=[f'KW_IGDB_{e}' for e in req.igdb_keywords],
            steam_genres=[f'STEAM_{e}' for e in req.steam_genres],
            steam_tags=[f'KW_STEAM_{e}' for e in req.steam_tags],
        )
        Y = model(X)
        genres = labels.labels(Y[0], threshold=.3333)

        resp = GenresResponse(req.id, req.name, espy_genres=genres)
        return jsonify(resp)
    except Exception as e:
        error_message = {'error': str(e)}
        return jsonify(error_message), 500


@app.route('/genres_debug', methods=['POST'])
def handle_genres_debug():
    try:
        json = request.get_json()
        req = GenresRequest(**json)

        X = features.build_array(
            igdb_genres=[f'IGDB_{e}' for e in req.igdb_genres],
            igdb_keywords=[f'KW_IGDB_{e}' for e in req.igdb_keywords],
            steam_genres=[f'STEAM_{e}' for e in req.steam_genres],
            steam_tags=[f'KW_STEAM_{e}' for e in req.steam_tags],
        )
        Y = model(X)
        genres = labels.labels(Y[0], threshold=.3333)
        decoded = labels.decode_array(Y[0])

        resp = GenresResponse(
            req.id, req.name, espy_genres=genres, debug_info=GenresDebugInfo(labels=decoded))
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
        '--port', help='Port number to listen for requests. (default: 8080)', type=int, default=8080)
    parser.add_argument(
        '--debug', help='Run the flask app in debug mode. (default: True)', default=True, action=BooleanOptionalAction)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    if (args.debug):
        model.summary()

    app.run(debug=args.debug, port=args.port)
