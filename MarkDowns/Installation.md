
To install follow the instructions.
1. (Optional) source your custom environment
2. Install the environment `pip install -e .` 
3. Install [tensorflow](https://www.tensorflow.org/install/pip)

### 4 Installing Ray
This repo uses a custom version of rllib so you need to follow the 
[instructions](https://ray.readthedocs.io/en/latest/installation.html#building-ray-from-source) on how to install the custom ray library.
1. Download the necessary [dependencies](https://ray.readthedocs.io/en/latest/installation.html#dependencies).
2. Run `ray/ci/travis/install-bazel.sh`
3. Install ray: 
```
cd ray/python
pip install -e . --verbose  # Add --user if you see a permission denied error.
```