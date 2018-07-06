import os
import numpy as np
import glob2
import json
import random
import keras.preprocessing.image as image_preprocessing
import progressbar


class DataGenerator(object):

    def __init__(self, dataset_path, input_type, output_targets):

        # Preconditions.
        assert os.path.exists(dataset_path), "dataset_path must exist: " + str(dataset_path)
        assert isinstance(input_type, str), "input_type must be string: " + str(input_type)
        assert isinstance(output_targets, list), "output_targets must be list: " + str(output_targets)

        # Assign the instance-variables.
        self.dataset_path = dataset_path
        self.input_type = input_type
        self.output_targets = output_targets

        # Get all the paths.
        self._get_paths()

        # Find all QR-codes.
        self._find_qrcodes()
        assert self.qrcodes != [], "No QR-codes found!"

        # Create the QR-codes dictionary.
        self._create_qrcodes_dictionary()


    def _get_paths(self):

        # Getting the paths for images.
        glob_search_path = os.path.join(self.dataset_path, "storage/person", "**/*.jpg")
        self.jpg_paths = glob2.glob(glob_search_path)

        # Getting the paths for point clouds.
        glob_search_path = os.path.join(self.dataset_path, "storage/person", "**/*.pcd")
        self.pcd_paths = glob2.glob(glob_search_path)

        # Getting the paths for personal and measurement.
        glob_search_path = os.path.join(self.dataset_path, "**/*.json")
        json_paths = glob2.glob(glob_search_path)
        self.json_paths_personal = [json_path for json_path in json_paths if "measures" not in json_path]
        self.json_paths_measures = [json_path for json_path in json_paths if "measures" in json_path]
        del json_paths


    def _find_qrcodes(self):

        qrcodes = []
        for json_path_measure in self.json_paths_measures:
            json_data_measure = json.load(open(json_path_measure))
            qrcode = self._extract_qrcode(json_data_measure)
            qrcodes.append(qrcode)
        qrcodes = sorted(list(set(qrcodes)))

        self.qrcodes = qrcodes


    def _create_qrcodes_dictionary(self):

        qrcodes_dictionary = {}

        for json_path_measure in self.json_paths_measures:
            json_data_measure = json.load(open(json_path_measure))

            # Ensure manual data.
            if json_data_measure["type"]["value"] != "manual":
                continue

            # Extract the QR-code.
            qrcode = self._extract_qrcode(json_data_measure)

            # In the future there will be multiple manual measurements. Handle this!
            if qrcode in qrcodes_dictionary.keys():
                raise Exception("Multiple manual measurements for QR-code: " + qrcode)

            # Extract the targets.
            targets = self._extract_targets(json_data_measure)
            jpg_paths = [jpg_path for jpg_path in self.jpg_paths if qrcode in jpg_path and "measurements" in jpg_path]
            pcd_paths = [pcd_path for pcd_path in self.pcd_paths if qrcode in pcd_path and "measurements" in pcd_path]

            qrcodes_dictionary[qrcode] = targets, jpg_paths, pcd_paths

        self.qrcodes_dictionary = qrcodes_dictionary


    def _extract_targets(self, json_data_measure):
        targets = []
        for output_target in self.output_targets:
            value = json_data_measure[output_target]["value"]
            targets.append(value)
        return targets


    def _extract_qrcode(self, json_data_measure):
        person_id = json_data_measure["personId"]["value"]
        json_path_personal = [json_path for json_path in self.json_paths_personal if person_id in json_path]
        assert len(json_path_personal) == 1
        json_path_personal = json_path_personal[0]
        json_data_personal = json.load(open(json_path_personal))
        qrcode = json_data_personal["qrcode"]["value"]
        return qrcode


    def _load_image(self, image_path):
        image = image_preprocessing.load_img(image_path, target_size=(160, 90))
        image = image.rotate(-90, expand=True)
        image = np.array(image)
        return image


    def generate(self, size, qrcodes_to_use=None, verbose=False):

        if qrcodes_to_use == None:
            qrcodes_to_use = self.qrcodes

        while True:

            x_inputs = []
            y_outputs = []

            if verbose == True:
                bar = progressbar.ProgressBar(max_value=size)
            while len(x_inputs) < size:

                # Get a random QR-code.
                qrcode = random.choice(qrcodes_to_use)

                # Get targets and paths.
                if qrcode not in  self.qrcodes_dictionary.keys():
                    continue
                targets, jpg_paths, pcd_paths = self.qrcodes_dictionary[qrcode]

                # Get a random image.
                if len(jpg_paths) == 0:
                    continue
                jpg_path = random.choice(jpg_paths)
                image = self._load_image(jpg_path)

                x_input = image
                y_output = targets

                x_inputs.append(x_input)
                y_outputs.append(y_output)

                if verbose == True:
                    bar.update(len(x_inputs))

            if verbose == True:
                bar.finish()

            x_inputs = np.array(x_inputs)
            y_outputs = np.array(y_outputs)

            yield x_inputs, y_outputs


def test_generator():

    if os.path.exists("datasetpath.txt"):
        dataset_path = open("datasetpath.txt", "r").read().replace("\n", "")
    else:
        dataset_path = "../data"

    data_generator = DataGenerator(dataset_path=dataset_path, input_type="image", output_targets=["height", "weight"])

    print("jpg_paths", len(data_generator.jpg_paths))
    print("pcd_paths", len(data_generator.jpg_paths))
    print("json_paths_personal", len(data_generator.jpg_paths))
    print("json_paths_measures", len(data_generator.jpg_paths))
    print("QR-Codes:\n" + "\n".join(data_generator.qrcodes))
    #print(data_generator.qrcodes_dictionary)

    qrcodes_shuffle = list(data_generator.qrcodes)
    random.shuffle(qrcodes_shuffle)
    split_index = int(0.8 * len(qrcodes_shuffle))
    qrcodes_train = qrcodes_shuffle[:split_index]
    qrcodes_validate = qrcodes_shuffle[split_index:]

    print("Training data:")
    x_train, y_train = next(data_generator.generate(size=20, qrcodes_to_use=qrcodes_train))
    print(x_train.shape)
    print(y_train.shape)
    print("")

    print("Validation data:")
    x_validate, y_validate = next(data_generator.generate(size=20, qrcodes_to_use=qrcodes_validate))
    print(x_validate.shape)
    print(y_validate.shape)
    print("")


if __name__ == "__main__":
    test_generator()
