import json
import logging
import os
import requests
import tarfile
import tensorflow as tf
import numpy as np
import torch
import os
import logging
import requests

from unsup_vvs.neural_fit.cleaned_network_builder import get_network_outputs
from unsup_vvs.neural_fit.brainscore_mask.bs_fit_utils import get_dc_model
from unsup_vvs.neural_fit.brainscore_mask.bs_fit_utils import get_la_cmc_model

from model_tools.activations.tensorflow import TensorflowSlimWrapper
from model_tools.activations.tensorflow import load_resize_image
from model_tools.activations.pytorch import load_images
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper

_logger = logging.getLogger(__name__)


class ModelBuilder:
    CKPT_PATH = {
        'resnet18-supervised': 'http://visualmaster-models.s3.amazonaws.com/supervised/seed0/checkpoint-505505.tar',
        'resnet18-la': 'http://visualmaster-models.s3.amazonaws.com/la/seed1/checkpoint-2502500.tar',
        'resnet18-ir': 'http://visualmaster-models.s3.amazonaws.com/ir/seed1/checkpoint-2502500.tar',
        'resnet18-ae': 'http://visualmaster-models.s3.amazonaws.com/ae/seed0/checkpoint-1301300.tar',
        'resnet18-cpc': 'http://visualmaster-models.s3.amazonaws.com/cpc/seed0/model.ckpt-1301300.tar',
        'resnet18-color': 'http://visualmaster-models.s3.amazonaws.com/color/seed0/model.ckpt-5605040.tar',
        'resnet18-rp': 'http://visualmaster-models.s3.amazonaws.com/rp/seed0/model.ckpt-1181162.tar',
        'resnet18-depth': 'http://visualmaster-models.s3.amazonaws.com/depth/seed0/model.ckpt-2982682.tar',
        'prednet': 'http://visualmaster-models.s3.amazonaws.com/prednet/seed0/checkpoint-90000.tar',
        'resnet18-simclr': 'http://visualmaster-models.s3.amazonaws.com/simclr/seed0/model.ckpt-311748.tar',
        'resnet18-deepcluster': 'http://visualmaster-models.s3.amazonaws.com/dc/seed0/checkpoint.pth.tar',
        'resnet18-cmc': 'http://visualmaster-models.s3.amazonaws.com/cmc/seed0/final.pth.tar',
    }
    PREP_TYPE = {
        'resnet18-color': 'color_prep',
        'resnet18-depth': 'no_prep',
        'resnet18-rp': 'only_mean',
        'prednet': 'no_prep',
    }
    CFG_KWARGS = {
        'resnet18-color': {
            'ignorebname_new': 0,
            'add_batchname': '_colorization'},
        'resnet18-depth': {
            'ignorebname_new': 0,
            'add_batchname': '_pbrnet'},
        'resnet18-rp': {
            'ignorebname_new': 0,
            'add_batchname': '_rp'},
    }
    MODEL_TYPE = {
        'prednet': 'vd_prednet:default',
        'resnet18-simclr': 'simclr_model',
    }
    BATCH_SIZE = {
        'prednet': 32,
    }
    PT_MODELS = ['resnet18-deepcluster', 'resnet18-cmc']

    def __call__(self, identifier):
        if identifier not in self.CKPT_PATH:
            raise ValueError(f"No known checkpoint for identifier {identifier}")
        load_from_ckpt = self.__get_ckpt_from_aws(identifier)
        if identifier not in self.PT_MODELS:
            return self.__get_tf_model(
                identifier=identifier, load_from_ckpt=load_from_ckpt,
                batch_size=self.BATCH_SIZE.get(identifier, 64),
                model_type=self.MODEL_TYPE.get(identifier, 'vm_model'),
                prep_type=self.PREP_TYPE.get(identifier, 'mean_std'),
                cfg_kwargs=self.CFG_KWARGS.get(identifier, {}))
        else:
            return self.__get_pt_model(
                    identifier=identifier,
                    load_from_ckpt=load_from_ckpt)

    def __get_ckpt_from_aws(self, identifier):
        framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
        weightsdir_path = os.path.join(framework_home, 'model-weights', 'unsup_vvs', identifier)
        aws_path = self.CKPT_PATH[identifier]
        if identifier not in self.PT_MODELS:
            weight_data_name = os.path.basename(aws_path)[:-3] + 'data-00000-of-00001'
        else:
            weight_data_name = os.path.basename(aws_path)
        weights_path = os.path.join(weightsdir_path, weight_data_name)
        if os.path.exists(weights_path):
            _logger.debug(f"Using cached weights at {weights_path}")
        else:
            _logger.debug(f"Downloading weights for {identifier} to {weights_path}")
            os.makedirs(weightsdir_path, exist_ok=True)
            tar_path = os.path.join(
                weightsdir_path, os.path.basename(aws_path))
            r = requests.get(aws_path, allow_redirects=True)
            with open(tar_path, 'wb') as tar_file:
                tar_file.write(r.content)
            if identifier not in self.PT_MODELS:
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=weightsdir_path)
                os.remove(tar_path)
        if identifier not in self.PT_MODELS:
            return os.path.join(weightsdir_path, os.path.basename(aws_path)[:-4])
        else:
            return os.path.join(weightsdir_path, os.path.basename(aws_path))

    def __get_prednet_var_list(self, ckpt_file):
        # Resolve a naming inconsistency between tensorflow and keras
        reader = tf.train.NewCheckpointReader(ckpt_file)
        var_shapes = reader.get_variable_to_shape_map()
        var_dict = {}
        for each_var in var_shapes:
            if ('prednet' in each_var) \
                    and ('layer' in each_var) \
                    and ('Adam' not in each_var):
                var_dict[each_var] = each_var.strip('__GPU0__/')
        load_var_list = json.dumps(var_dict)
        return load_var_list

    def __get_tf_model(self,
                       identifier, load_from_ckpt,
                       batch_size=64, model_type='vm_model', prep_type='mean_std',
                       cfg_kwargs={}):
        img_path_placeholder = tf.placeholder(
            dtype=tf.string,
            shape=[batch_size])
        ending_points = self._build_model_ending_points(
            img_paths=img_path_placeholder, prep_type=prep_type, 
            model_type=model_type, cfg_kwargs=cfg_kwargs)

        load_var_list = None
        if identifier == 'prednet':
            load_var_list = self.__get_prednet_var_list(load_from_ckpt)
        SESS = self.get_tf_sess_restore_model_weight(
                load_from_ckpt=load_from_ckpt,
                load_var_list=load_var_list)

        self.ending_points = ending_points
        self.img_path_placeholder = img_path_placeholder
        self.SESS = SESS
        self.identifier = identifier
        return self._build_activations_model(batch_size=batch_size)

    def _build_model_ending_points(self, img_paths, prep_type, model_type,
                                   setting_name='cate_res18_exp0', cfg_kwargs={}):
        imgs = self._get_imgs_from_paths(img_paths)

        ending_points, _ = get_network_outputs(
            {'images': imgs},
            prep_type=prep_type,
            model_type=model_type,
            setting_name=setting_name,
            **cfg_kwargs)
        for key in ending_points:
            if len(ending_points[key].get_shape().as_list()) == 4:
                ending_points[key] = tf.transpose(
                    ending_points[key],
                    [0, 3, 1, 2])
        return ending_points

    def _get_imgs_from_paths(self, img_paths):
        _load_func = lambda image_path: load_resize_image(
            image_path, 224)
        imgs = tf.map_fn(_load_func, img_paths, dtype=tf.float32)
        return imgs

    def get_tf_sess_restore_model_weight(self, load_var_list=None, from_scratch=False, load_from_ckpt=None):
        SESS = self.get_tf_sess()
        if load_var_list is not None:
            name_var_list = json.loads(load_var_list)
            needed_var_list = {}
            curr_vars = tf.global_variables()
            curr_names = [variable.op.name for variable in curr_vars]
            for old_name in name_var_list:
                new_name = name_var_list[old_name]
                assert new_name in curr_names, "Variable %s not found!" % new_name
                _ts = curr_vars[curr_names.index(new_name)]
                needed_var_list[old_name] = _ts
            saver = tf.train.Saver(needed_var_list)

            init_op_global = tf.global_variables_initializer()
            SESS.run(init_op_global)
            init_op_local = tf.local_variables_initializer()
            SESS.run(init_op_local)
        else:
            saver = tf.train.Saver()

        if not from_scratch:
            saver.restore(SESS, load_from_ckpt)
        else:
            init_op_global = tf.global_variables_initializer()
            SESS.run(init_op_global)
            init_op_local = tf.local_variables_initializer()
            SESS.run(init_op_local)

        assert len(SESS.run(tf.report_uninitialized_variables())) == 0, \
            (SESS.run(tf.report_uninitialized_variables()))
        return SESS

    def get_tf_sess(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        SESS = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
        ))
        return SESS

    def _build_activations_model(self, batch_size):
        self.activations_model = TensorflowSlimWrapper(
            identifier=self.identifier, labels_offset=0,
            endpoints=self.ending_points,
            inputs=self.img_path_placeholder,
            session=self.SESS,
            batch_size=batch_size)
        return self.activations_model

    def __get_pt_model(self, identifier, load_from_ckpt):
        self.identifier = identifier
        self.load_from_ckpt = load_from_ckpt
        if identifier == 'resnet18-deepcluster':
            self.__get_dc_model()
        elif identifier == 'resnet18-cmc':
            self.__get_cmc_model()
        else:
            raise NotImplementedError
        self.activations_model = PytorchWrapper(
                identifier=self.identifier,
                model=self.model,
                preprocessing=self.preprocessing,
                batch_size=64,
                )
        return self.activations_model

    def __get_dc_model(self):
        model = get_dc_model(self.load_from_ckpt)
        sobel_filter = model.sobel
        model = model.features

        def _do_sobel(images):
            device = torch.device("cpu")
            images = torch.from_numpy(images)
            if torch.cuda.is_available():
                images = images.cuda()
            images = torch.autograd.Variable(images)
            images = sobel_filter(images)
            images = images.float().to(device).numpy()
            return images

        def _hvm_load_preprocess(image_paths):
            images = load_preprocess_images(image_paths, 224)
            return _do_sobel(images)
        self.model = model
        self.preprocessing = _hvm_load_preprocess

    def __get_cmc_model(self):
        model = get_la_cmc_model(self.load_from_ckpt)
        self.model = model.module.l_to_ab

        def _do_resize_lab_normalize(images, img_size):
            from unsup_vvs.neural_fit.pt_scripts.main import tolab_normalize
            from PIL import Image
            post_images = []
            for image in images:
                image = image.resize([img_size, img_size])
                image = np.asarray(image).astype(np.float32)
                if len(image.shape) == 2:
                    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
                image = tolab_normalize(image)
                image = image[:, :, :1]
                image = np.transpose(image, [2, 0, 1])
                post_images.append(image)
            images = np.stack(post_images, axis=0)
            return images
        def _hvm_load_preprocess(image_paths):
            images = load_images(image_paths)
            return _do_resize_lab_normalize(images, 224)
        self.preprocessing = _hvm_load_preprocess
