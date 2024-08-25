import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

from classifier.dataset.genres import Genres
from classifier.dataset.features import Features
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
    gog_genres: List[str] = field(default_factory=list)
    gog_tags: List[str] = field(default_factory=list)
    description: str = field(default='')


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
    debug_info: GenresDebugInfo = field(default_factory=dict)


features = Features.load()
genres = Genres.load()
model = None


def predict_genres(filename):
    global model
    model = tf.keras.models.load_model(filename)
    return app


@app.route('/genres', methods=['POST'])
def handle_genres():
    try:
        json = request.get_json()
        req = GenresRequest(**json)

        X = features.build_array(
            igdb_genres=req.igdb_genres,
            steam_genres=req.steam_genres,
            gog_genres=req.gog_genres,
            igdb_keywords=req.igdb_keywords,
            steam_tags=req.steam_tags,
            gog_tags=req.gog_tags,
            description=req.description,
        )
        Y = model(X)
        espy_genres = genres.labels(Y[0], threshold=.2)

        resp = GenresResponse(req.id, req.name, espy_genres=espy_genres)
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
            igdb_genres=req.igdb_genres,
            steam_genres=req.steam_genres,
            gog_genres=req.gog_genres,
            igdb_keywords=req.igdb_keywords,
            steam_tags=req.steam_tags,
            gog_tags=req.gog_tags,
            description=req.description
        )
        Y = model(X)
        espy_genres = genres.labels(Y[0], threshold=.2)
        decoded = genres.decode_array(Y[0])

        resp = GenresResponse(
            req.id, req.name, espy_genres=espy_genres, debug_info=GenresDebugInfo(labels=decoded))
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
