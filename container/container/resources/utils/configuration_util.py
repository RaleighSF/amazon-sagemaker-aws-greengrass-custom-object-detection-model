import os
import json
from ast import literal_eval
import tensorflow as tf
from object_detection.utils import config_util


class Configuration:
    def __init__(self):
        # DEFAULT PARAMS
        self.object_detection_folder = '/opt/ml/code/tensorflow-models/research/object_detection/'
        self.training_script = os.path.join(self.object_detection_folder, 'model_main.py')
        self.freezing_script = os.path.join(self.object_detection_folder, 'export_inference_graph.py')

        # SAGEMAKER PARAMS
        self.prefix = '/opt/ml/'
        self.input_path = os.path.join(self.prefix, 'input/data')
        self.output_path = os.path.join(self.prefix, 'output')
        self.model_path = os.path.join(self.prefix, 'model')
        self.pretrained_checkpoint_path = os.path.join(self.input_path, 'checkpoint/')
        print(f'EXTRACTED CHECKPOINT FILES: {os.listdir(self.pretrained_checkpoint_path)}')
        self.base_pipeline_config = None
        self.hparams_path = os.path.join(self.prefix, 'input/config/hyperparameters.json')
        self.hparams = self.load_hparams(self.hparams_path)
        self.label_map_path = self.hparams['label_map_path']
        print(f'LOADED TRAINING PARAMETERS: {self.hparams}')
        self.pipeline_config_path = self.generate_pipeline_config()
        self.num_steps = int(self.hparams['train_config.num_steps'])
        print('CONFIGURATION LOADED')

    def generate_pipeline_config(self):
        configs = config_util.get_configs_from_pipeline_file(self.base_pipeline_config)
        tf_hparams = tf.contrib.training.HParams(**self.hparams)
        config_util.merge_external_params_with_configs(configs, tf_hparams)
        pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_config, self.prefix)
        return os.path.join(self.prefix, 'pipeline.config')

    def load_hparams(self, file_path):
        with open(file_path, encoding='utf-8') as f:
            content = json.load(f)
            hparams = self.destringify_dict_values(content)
            hparams['train_input_path'] = os.path.join(self.input_path, hparams['train_input_path'])
            hparams['eval_input_path'] = os.path.join(self.input_path, hparams['eval_input_path'])
            hparams['train_input_config.label_map_path'] = os.path.join(self.input_path,
                                                                        hparams['train_input_config.label_map_path'])
            hparams['label_map_path'] = os.path.join(self.input_path, hparams['label_map_path'])
            hparams['train_config.fine_tune_checkpoint'] = os.path.join(self.input_path,
                                                                        hparams['train_config.fine_tune_checkpoint'])
            hparams.pop('_tuning_objective_metric', None) # delete HPO key if exists
            base_config_name = hparams['base_config_name']
            self.base_pipeline_config = os.path.join(self.object_detection_folder, 'samples/configs/', base_config_name)
            print(f'USING BASE CONFIG: {self.base_pipeline_config}')
            hparams.pop('base_config_name', None)  # delete base pipeline config name
        return hparams

    def destringify_dict_values(self, d):
        return {k: self.destringify(v) for k, v in d.items()}

    @staticmethod
    def destringify(s):
        if s == len(s) * "0":
            return 0
        else:
            s = s.lstrip("0")
        if isinstance(s, str):
            try:
                val = literal_eval(s)
            except ValueError:
                val = s
        else:
            val = s
        if isinstance(val, float):
            if val.is_integer():
                return int(val)
            return val
        return val
