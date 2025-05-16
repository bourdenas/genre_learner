import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import tensorflow as tf

from argparse import ArgumentParser, BooleanOptionalAction
from flask import Flask, request, jsonify

from classifier.dataset.features import Features
from classifier.dataset.genres import Genres
from classifier.predictor.debug_info import DebugInfo, PredictionInfo
from classifier.predictor.request import GenresRequest
from classifier.predictor.response import GenresResponse

app = Flask(__name__)


features = Features.load()
genres = Genres.load()
model = None
prediction_threshold = .5


def predict_genres(filename: str, threshold: float = .5):
    global model
    global prediction_threshold

    model = tf.keras.models.load_model(filename)
    prediction_threshold = threshold
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
            wiki_genres=req.wiki_genres,
            igdb_tags=req.igdb_tags,
            steam_tags=req.steam_tags,
            gog_tags=req.gog_tags,
            wiki_tags=req.wiki_tags,
            description=req.description,
        )
        Y = model(X)
        espy_genres = genres.labels(Y[0], threshold=prediction_threshold)
        if req.debug:
            debug_info = DebugInfo(
                features=features.debug(X), predictions=genres.debug(Y[0]))
        else:
            debug_info = {}

        resp = GenresResponse(
            req.id, req.name, espy_genres=espy_genres, debug_info=debug_info)
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
        '--threshold', help='Prediction threshold for genre label. (default: 0.5)', type=float, default=.5)
    parser.add_argument(
        '--port', help='Port number to listen for requests. (default: 8080)', type=int, default=8080)
    parser.add_argument(
        '--debug', help='Run the flask app in debug mode. (default: True)', default=True, action=BooleanOptionalAction)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    prediction_threshold = args.threshold

    if (args.debug):
        model.summary()

    app.run(debug=args.debug, port=args.port)
