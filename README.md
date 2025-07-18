<div align="center">
    <img src="https://raw.githubusercontent.com/CNES/picanteo/master/docs/images/logo_with_text.png" width=500>

**Picanteo: CNES change detection framework for natural disaster response**

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pypi](https://img.shields.io/pypi/v/picanteo?color=%2334D058&label=pypi)](https://pypi.org/project/picanteo/)
</div>
 
# 🌶️ Overview

**Picanteo** is a flexible and modular framework developed by CNES for building custom change detection pipelines. It provides a set of tools in the `picanteo/toolbox` module, allowing users to create tailored pipelines or integrate their own tools to extend functionality.  
Whether you're analyzing satellite imagery or other geospatial data, **Picanteo** simplifies the process of detecting changes in dynamic environments.
<div align="center">
    <img src="https://raw.githubusercontent.com/CNES/picanteo/master/docs/images/picanteo_visualize.png" width=1024>
</div>   


# ✨ Features
- **Modular Pipeline Design**: Easily construct and customize change detection pipelines using predefined or user-developed tools.
- **Extensible Toolbox**: Integrate your own tools into the `picanteo/toolbox` for enhanced flexibility.
- **Configuration-Driven**: Define pipelines using simple YAML configuration files.
- **Open Source**: Licensed under the Apache 2.0 License, encouraging collaboration and contributions.

# 🚀 Quick Start

## 🛠️ Installation

Install **Picanteo** via pip:
```sh
pip install picanteo
```
### Requirements
- Python 3.9 or higher

Dependencies are automatically installed via pip.  
See [install_requires](setup.cfg) for details.

## ⚙️ Run **Picanteo**

1) To launch **Picanteo**, you need a YAML configuration file specifying the pipeline and its parameters. 
Template configurations are available in the `/conf/pipelines` directory.  
2) Create or modify a YAML configuration file (e.g., my_conf.yaml).

3) Once this file is created, simply run the following command:
```sh
picanteo my_conf.yaml
```

If you want to try the pipelines already available in **Picanteo**, you have to download the weights from our model with this [link](https://drive.google.com/uc?id=1n1olMUY3ycx48YRZ7ZG-ME63cNjnRBtc). Then you have to move those weights in a specific folder:
```sh
mkdir docs/tutorials/data/weights && mv weights.ckpt docs/tutorials/data/weights/manet_cos1_weights.ckpt
```
Then you can run the following command to launch the bitemporal 2D/3D change detection pipeline demo:
```sh
picanteo conf/pipelines/bitemporal_2D3D_pipeline.yaml
```
And you can run the following command to try the bitemporal 2D change detection pipeline demo:
```sh
picanteo conf/pipelines/bitemporal_2D_pipeline.yaml
```
You can now see the results with the following command line:
```sh
picanteo_visualize conf/steps/visualization_conf.yaml
```
*Hint:  in case you just want to try one of the pipelines, you should edit the `visualization_conf.yaml` and remove the unused pipeline under the `pages` section.*

# 👨‍🏫 Tutorials

Learn how to use **Picanteo** with these step-by-step guides:
- [Creating Your Own Change Detection Pipeline](docs/tutorials/how_to_create_a_pipeline.ipynb): A Jupyter notebook tutorial for building custom pipelines.
- [Preparing Your Data](docs/tutorials/data_readiness.md): Instructions for formatting your data to work with **Picanteo**’s toolbox.

# ✒️ Credits
If you use **Picanteo** in your research, please cite the following paper:
```text
@INPROCEEDINGS{picanteo2024,
  author={Hümmer, Christian and Lallement, Dimitri and Youssefi, David},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Uncertainty-Aware 2d/3d Change Detection for Natural Disaster Response}, 
  year={2024},
  pages={3796-3801},
  doi={10.1109/IGARSS53475.2024.10642271}
}
```
# 📜 License

**Picanteo**  is licensed under Apache License v2.0. Please refer to the [LICENSE](LICENSE) file for more details.

# 🆘 Support

For issues, questions, or feature requests, please open an issue on our [GitHub Issues page](https://github.com/CNES/picanteo/issues) or check the documentation for additional resources.


# 🤝Contributing
We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get involved, including coding standards and submission processes.
