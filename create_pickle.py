if os.path.exists("datasetpath.txt"):
    dataset_path = open("datasetpath.txt", "r").read().replace("\n", "")
else:
    dataset_path = "/home/jovyan/work/data/"

input_type="pointcloud"

# For creating voxelgrids.
if input_type == "voxelgrid"
    dataset_parameters = {}
    dataset_parameters["input_type"] = "voxelgrid"
    dataset_parameters["output_targets"] = ["height", "weight"]    
    dataset_parameters["random_seed"] = 666
    dataset_parameters["voxelgrid_target_shape"] = (32, 32, 32)
    dataset_parameters["voxel_size_meters"] = 0.1
    dataset_parameters["voxelgrid_random_rotation"] = True
    dataset_parameters["dataset_size_train"] = 6000
    dataset_parameters["dataset_size_test"] = 1000

# For creating pointclouds.
if input_type == "pointcloud"
    dataset_parameters = {}
    dataset_parameters["input_type"] = "pointcloud"
    dataset_parameters["output_targets"] = ["height", "weight"]    
    dataset_parameters["random_seed"] = 666
    dataset_parameters["pointcloud_target_size"] = 30000
    dataset_parameters["pointcloud_random_rotation"] = True
    dataset_parameters["dataset_size_train"] = 3000
    dataset_parameters["dataset_size_test"] = 500

print("Creating data-generator...")
data_generator = DataGenerator(
    dataset_path=dataset_path, 
    input_type=dataset_parameters["input_type"], 
    output_targets=dataset_parameters["output_targets"],
    voxelgrid_target_shape=dataset_parameters.get("voxelgrid_target_shape", None),
    voxel_size_meters=dataset_parameters.get("voxel_size_meters", None),
    voxelgrid_random_rotation=dataset_parameters.get("voxelgrid_random_rotation", None),
    pointcloud_target_size=dataset_parameters.get("pointcloud_target_size", None),
    pointcloud_random_rotation=dataset_parameters.get("pointcloud_random_rotation", None)
)
data_generator.print_statistics()

do_analysis = False
#do_analysis = True

if do_analysis == True:
    data_generator.analyze_files()
    data_generator.analyze_targets()
    data_generator.analyze_pointclouds()
    data_generator.analyze_voxelgrids()
    # how much data per measure?
else:
    print("Skipped analysis.")

# Do the split.
random.seed(dataset_parameters["random_seed"])
qrcodes_shuffle = data_generator.qrcodes[:]
random.shuffle(qrcodes_shuffle)
split_index = int(0.8 * len(qrcodes_shuffle))
qrcodes_train = sorted(qrcodes_shuffle[:split_index])
qrcodes_test = sorted(qrcodes_shuffle[split_index:])
del qrcodes_shuffle
print("")

print("QR-Codes for training:", " ".join(qrcodes_train))
print("")
print("QR-Codes for testing:", " ".join(qrcodes_test))
print("")

print("Generating training data...")
dataset_train = next(data_generator.generate(size=dataset_parameters["dataset_size_train"], verbose=False))

print("Generating testing data...")
dataset_test = next(data_generator.generate(size=dataset_parameters["dataset_size_test"], verbose=False))
    
print("Done.")

datetime_string = datetime.datetime.now().strftime("%Y%m%d-%H%M")
dataset_name = datetime_string + "-" + dataset_parameters["input_type"] + "-dataset.p"
pickle.dump((dataset_train, dataset_test, dataset_parameters), open(dataset_name, "wb"))
print("Saved " + dataset_name)