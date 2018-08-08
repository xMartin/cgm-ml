from __future__ import absolute_import

# For importing stuff.
import sys
sys.path.append("..")

import utils
from keras.models import load_model
import numpy as np
import pyntcloud
import pandas as pd
from flask import Flask, request, Response
import jsonpickle
import tensorflow as tf
from keras import backend as K


class CGMRegressor(object):

    def __init__(self):
        model_path = utils.get_latest_model("..", "voxnet")
        print(model_path)

        self.model = load_model(model_path)
        self.model.summary()
        self.graph = tf.get_default_graph()

        self.voxel_size_meters = 0.1


    def predict(self, X, features_names):

        # Reshape first.
        points = np.reshape(X, (-1, 4))

        # Turn it into a voxelgrid.
        dataframe = pd.DataFrame(points, columns=["x", "y", "z", "c"])
        point_cloud = pyntcloud.PyntCloud(dataframe)
        voxelgrid_id = point_cloud.add_structure("voxelgrid", size_x=self.voxel_size_meters, size_y=self.voxel_size_meters, size_z=self.voxel_size_meters)
        voxelgrid = point_cloud.structures[voxelgrid_id].get_feature_vector(mode="density")

        # Pad the voxelgrid.
        voxelgrid = utils.ensure_voxelgrid_shape(voxelgrid, (32, 32, 32))
        voxelgrid = np.expand_dims(voxelgrid, axis=0)

        with self.graph.as_default():
            return self.model.predict(voxelgrid)


# When using flask.




# Note: Just for testing. Hopefully seldon will do something similar.
if __name__ == "__main__":

    regressor = CGMRegressor()
    app = Flask(__name__)

    @app.route("/cgm/regressor", methods=["POST"])
    def flask_post():

        r = request
        points = np.fromstring(r.data, np.float32)
        print("Received points", points.shape)

        # Predict.
        prediction = regressor.predict(points, [])[0]

        # Build the response.
        response = {
            "message": "success",
            "height": str(prediction[0]),
            "weight": str(prediction[1])
        }
        response = jsonpickle.encode(response)

        # Respond.
        return Response(response=response, status=200, mimetype="application/json")

    app.run(host="0.0.0.0", port=5000)

    #pointcloud = np.random.random(30000 * 4)
    #prediction = regressor.predict(pointcloud, features_names=["height", "weight"])
    #print("Prediction:", prediction)
