from ray.rllib.models import ModelCatalog

from policies.SimpleQPolicy import MyPreprocessorClass

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)