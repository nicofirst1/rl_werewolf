
To install follow the instructions.
1. Clone this repo `git clone https://gitlab.com/nicofirst1/rl-werewolf --recursive`
2. (Optional) source your custom environment
3. Install the environment `pip install -e .` 
4. Install [tensorflow](https://www.tensorflow.org/install/pip)
5. Install [tensorflow probability](https://www.tensorflow.org/probability/install)
6. Install [Ray](https://ray.readthedocs.io/en/latest/installation.html#latest-stable-version)

###  Installing Ray [deprecated]
This repo uses a custom version of rllib so you need to follow the 
[instructions](https://ray.readthedocs.io/en/latest/installation.html#building-ray-from-source) on how to install the custom ray library.
1. Download the necessary [dependencies](https://ray.readthedocs.io/en/latest/installation.html#dependencies).
2. Run `ray/ci/travis/install-bazel.sh`
3. Install ray: 
```
cd ray/python
pip install -e . --verbose  # Add --user if you see a permission denied error.
```

If you ran into any problem for step 3 you probably need to follow [bazel installation](https://docs.bazel.build/versions/master/install-os-x.html).