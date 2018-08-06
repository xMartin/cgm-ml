README in progress, for now please refer to 
- info@childgrowthmonitor.org
- [GitHub main project](https://github.com/Welthungerhilfe/ChildGrowthMonitor/)
- [Child Growth Monitor Website](https://childgrowthmonitor.org)

# Child Growth Monitor Machine Learning

## Introduction
This project uses machine learning to identify malnutrition from 3D scans of children under 5 years of age. This [one-minute video](https://www.youtube.com/watch?v=f2doV43jdwg) explains.

## Getting started

### Requirements
Training the models realistically requires using GPU computation. Project members are currently using a variety of cloud instances (GCS, AWS, Azure) and local machines for training. A separate backend project is currently developing the DevOps infrastructure to simplify this.

You will need:
* Python 3
* TensorFlow GPU
* Keras

### Installation
These steps provide an example installation on a local Ubuntu workstation from scratch:
* Install Ubuntu Desktop 18.04.1 LTS	
* Install NVIDIA drivers  
*Please note that after rebooting, the secure boot process will prompt you to authorize the driver to use the hardware via a MOK Management screen.*
```sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt-get install nvidia-390
sudo reboot now
```
* Install [Anaconda with Python 3.6](https://www.anaconda.com/download)
```conda update conda
conda update anaconda
conda update python
conda update --all
conda create --name cgm
source activate cgm
conda install tensorflow-gpu
conda install ipykernel
conda install keras
conda install vtk progressbar2 glob2 numbs pandas
pip install --upgrade pip
pip install git+https://github.com/daavoo/pyntcloud
```

### Dataset access
Data access is provided on as-needed basis following signature of the Welthungerhilfe Data Privacy & Commitment to
Maintain Data Secrecy Agreement. If you need data access (e.g. to train your machine learning models), 
please contact [Markus Matiaschek](mailto:mmatiaschek@gmail.com) for details.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Versioning

Our [releases](https://github.com/Welthungerhilfe/cgm-ml/releases) use [semantic versioning](http://semver.org). You can find a chronologically ordered list of notable changes in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details and refer to [NOTICE](NOTICE) for additional licensing notes and uses of third-party components.
