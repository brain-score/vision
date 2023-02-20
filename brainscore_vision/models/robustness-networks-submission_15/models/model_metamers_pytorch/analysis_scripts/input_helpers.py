import os
import scipy.io.wavfile as wav
import h5py
import pickle
import random
import numpy as np
from shutil import copyfile
import glob
import scipy
try:
    from scipy.misc import imread, imresize
except:
    pass
from functools import partial
try:
    import resampy
except: 
    "Missing resampy package. This is fine if you are running only image models but audio models will not work correctly"
from PIL import Image
import json  
from analysis_scripts.default_paths import *

def generate_import_audio_functions(audio_func='psychophysicskell2018dry', preproc_scaled=1, rms_normalize=1, **kwargs):
  """
  Wrapper to choose which type of audio function to import.
  Input
  -----
  audio_func : a string determining which function will be returned to import the audio
  preproc_scaled (float) : multiplies the input audio by this value for scaling
  rms_normalize (None or float) : if not None, sets the RMS value to this float. 

  Returns
  -------
  audio_function : a function that takes in an index and returns a dictionary with (at minimum) the audio corresponding to the index along with the SR

  """
  if audio_func == 'psychophysicskell2018dry_overlap_jsinv3':
    return partial(psychophysicskell2018dry_overlap_jsinv3, preproc_scaled=preproc_scaled, rms_normalize=rms_normalize, **kwargs)
  elif audio_func == 'psychophysics_wsj400_jsintest':
    return partial(psychophysics_wsj400_jsintest, preproc_scaled=preproc_scaled, rms_normalize=rms_normalize, **kwargs)
  elif audio_func == 'load_specified_audio_path':
    return partial(use_audio_path_specified_audio, preproc_scaled=preproc_scaled, rms_normalize=rms_normalize, **kwargs)


def psychophysicskell2018dry_overlap_jsinv3(WAV_IDX, preproc_scaled=1, rms_normalize=None, SR=20000):
  """
  Loads an example from the dry psychophysics set used in kell2018 that is overlapped with the set in jsinv3
  This set contains 295 words

  Metamers from this set were used in Feather et al. 2019 (NeurIPS) 
  """

  # Contains ONLY the dry stimuli
  save_dry_path = os.path.join(ASSETS_PATH, 'behavioralKellDataset_sr20000_kellBehavioralDataset_jsinv3overlap_dry_only.pckl')

  with open(save_dry_path, 'rb') as handle:
    behavioral_dataset_kell = pickle.load(handle)

  word_to_int = dict(zip(behavioral_dataset_kell['stimuli']['word'], behavioral_dataset_kell['stimuli']['word_int']))
  int_to_word = dict(zip(behavioral_dataset_kell['stimuli']['word_int'], behavioral_dataset_kell['stimuli']['word']))
    
  SR_loaded = behavioral_dataset_kell['stimuli']['sr'][WAV_IDX]
  wav_f = behavioral_dataset_kell['stimuli']['signal'][WAV_IDX]
  if SR_loaded != SR:
    print('RESAMPLING')
    wav_f = resampy.resample(wav_f, SR_loaded, SR)
    SR_loaded = SR

  wav_f = wav_f * preproc_scaled # some of networks require us to scale the audio

  print("Loading: %s"%behavioral_dataset_kell['stimuli']['source'][WAV_IDX])
  
  if rms_normalize is not None:
    wav_f = wav_f - np.mean(wav_f.ravel())
    wav_f = wav_f/(np.sqrt(np.mean(wav_f.ravel()**2)))*rms_normalize
    print(np.sqrt(np.mean(wav_f.ravel()**2)))
    rms = rms_normalize
  else:
    rms = b['stimuli']['rms'][WAV_IDX]
  audio_dict={}

  audio_dict['wav'] = wav_f
  audio_dict['SR'] = SR
  audio_dict['word_int'] = behavioral_dataset_kell['stimuli']['word_int'][WAV_IDX]
  audio_dict['word'] = behavioral_dataset_kell['stimuli']['word'][WAV_IDX]
  audio_dict['rms'] = rms
  audio_dict['filename'] = behavioral_dataset_kell['stimuli']['path'][WAV_IDX]
  audio_dict['filename_short'] = behavioral_dataset_kell['stimuli']['source'][WAV_IDX]
  audio_dict['correct_response'] = behavioral_dataset_kell['stimuli']['word'][WAV_IDX]

  return audio_dict


def psychophysics_wsj400_jsintest(WAV_IDX, preproc_scaled=1, rms_normalize=None, SR=20000):
  """
  Loads an example from a set of 400 WSJ clips pulled from the jsinv3 test set.
  Each clip is of a different word, and a unique clip from WSJ (ie no two clips that are back to back words)

  Metamers from this set were used in Feather et al. 2022
  """

  pckl_path = os.path.join(ASSETS_PATH, 'word_WSJ_validation_jsin_400words_1samplesperword_with_metadata.pckl')
  with open(pckl_path, 'rb') as handle:
    behavioral_dataset = pickle.load(handle)
    
  word = behavioral_dataset['Dataset_Word_Order'][WAV_IDX]
  word_data = behavioral_dataset[word]
  assert word_data['dataframe_metadata']['word'] == word

  SR_loaded = word_data['dataframe_metadata']['sr']
  wav_f = word_data['audio_clips'][0]

  if SR_loaded != SR:
    print('RESAMPLING')
    wav_f = resampy.resample(wav_f, SR_loaded, SR)
    SR_loaded = SR

  wav_f = wav_f * preproc_scaled # some of networks require us to scale the audio

  print("Loading: %s"%word)

  # Always mean subtract the clip in this dataset. 
  wav_f = wav_f - np.mean(wav_f.ravel())
  rms_clip = np.sqrt(np.mean(wav_f.ravel()**2))

  if rms_normalize is not None:
    wav_f = wav_f/rms_clip*rms_normalize
    rms = rms_normalize
  else:
    rms = rms_clip
    
  audio_dict={}

  old_path = word_data['dataframe_metadata']['path']
  path_without_root = old_path.split('/home/raygon/projects/user/jfeather/')[-1]
    
  audio_dict['wav'] = wav_f
  audio_dict['SR'] = SR
  audio_dict['word_int'] = word_data['dataframe_metadata']['word_int']
  audio_dict['word'] = word_data['dataframe_metadata']['word']
  audio_dict['rms'] = rms
  audio_dict['filename'] = path_without_root
  audio_dict['filename_short'] = word_data['dataframe_metadata']['source']
  audio_dict['correct_response'] =  word_data['dataframe_metadata']['word']

  return audio_dict


def use_audio_path_specified_audio(WAV_IDX, wav_path=None, wav_word=None, 
                                   preproc_scaled=1, rms_normalize=None, SR=20000):
  """
  Loads an example wav specified by wav_path
  """
  del WAV_IDX

  word_and_speaker_encodings = pickle.load( open(WORD_AND_SPEAKER_ENCODINGS_PATH, "rb" ))
  word_to_int = word_and_speaker_encodings['word_to_idx']

  print("Loading: %s"%wav_path)
  SR_loaded, wav_f = scipy.io.wavfile.read(wav_path)
  if SR_loaded != SR:
    wav_f = resampy.resample(wav_f, SR_loaded, SR)
    SR_loaded = SR

  if rms_normalize is not None:
    wav_f = wav_f - np.mean(wav_f.ravel())
    wav_f = wav_f/(np.sqrt(np.mean(wav_f.ravel()**2)))*rms_normalize
    rms = rms_normalize
  else:
    rms = np.sqrt(np.mean(wav_f.ravel()**2))

  wav_f = wav_f * preproc_scaled # some of networks require us to scale the audio

  audio_dict={}

  audio_dict['wav'] = wav_f
  audio_dict['SR'] = SR
  audio_dict['word_int'] = word_to_int[wav_word]
  audio_dict['word'] = wav_word
  audio_dict['rms'] = rms
  audio_dict['filename'] = wav_path
  audio_dict['filename_short'] = wav_path.split('/')[-1]
  audio_dict['correct_response'] = wav_word

  return audio_dict


def generate_import_image_functions(image_func='small_16_class_imagenet',
                                    im_shape=224,
                                    image_path=None,
                                    image_class=None,
                                    data_format='NHWC',
                                    pytorch_dataset_kwargs=None):
  """
  Wrapper to choose which type of image function to import. Wrapper to easily work with command line arguments. 
  Input
  -----
  image_func : a string determining which function will be returned to import the images
  im_shape : int, the shape of the images to load (square)
  image_path : string, a path to a specific image if loading a specified image
  image_class : string, the imagenet class corresponding to the image at image_path
  data_format : string, the order of the dimensions. With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels]. 
  pytorch_dataset_kwargs : None or dict. Contains the keyword arguments used 
    with the pytorch data loader.

  Returns
  -------
  image_function : a function that takes in an index and returns a dictionary with (at minimum) the image corresponding to the index along with the filename and image shape
  
  """
  if image_func == '256_16_class_imagenet':
    return partial(medium_256_16_class_imagenet, im_shape=im_shape, data_format=data_format)
  elif image_func == '400_16_class_imagenet_val':
    return partial(medium_400_16_class_imagenet_val_images, im_shape=im_shape, data_format=data_format)
  elif image_func  == 'load_specified_image_path':
    return partial(use_image_path_specified_image, image_path=image_path, image_class=image_class, im_shape=im_shape, data_format=data_format)
  elif image_func == 'pytorch_image_datasets':
    return partial(use_pytorch_datasets, data_format=data_format, im_shape=im_shape, **pytorch_dataset_kwargs)


def medium_256_16_class_imagenet(IMG_IDX, im_shape=224, data_format='NHWC'):
  """
  Set consists of the 16 image net classes described in https://arxiv.org/pdf/1808.08750.pdf, and this medium dataset consists of 16 images randomly chosen from each class.
  This set was used in Feather et al. 2019 (NeurIPS). Images are from the training dataset, as in the linked paper. 
  """
  image_locations = os.path.join(ASSETS_PATH, 'full_256_16_class_imagenet/')
  mage_name = glob.glob('%s/%d_*'%(image_locations, IMG_IDX))[0].split('/')[-1] # image_list[IMG_IDX]
  print("Loading: %s"%image_name)
  assert int(image_name.split('_')[0]) == IMG_IDX, 'Check the ordering for the images'
  img1 = scipy.misc.imread(os.path.join(image_locations, image_name), mode='RGB')
  img1 = preproc_imagenet_center_crop(img1, im_shape=im_shape)

  if data_format=='NCHW':
    img1 = np.rollaxis(np.array(img1),2,0)
  elif data_format=='NHWC':
    pass
  else:
    raise ValueError('Unsupported data_format %s'%data_format)

  image_dict = {}
  image_dict['image'] = img1
  image_dict['shape'] = im_shape
  image_dict['filename'] = os.path.join(image_locations, image_name)
  image_dict['filename_short'] = image_name
  image_dict['correct_response'] = image_name.split('_')[1]
  image_dict['max_value_image_set'] = 255
  image_dict['min_value_image_set'] = 0
  return image_dict


def medium_400_16_class_imagenet_val_images(IMG_IDX, im_shape=224, data_format='NHWC'):
  """
  Set consists of the 16 image net classes described in https://arxiv.org/pdf/1808.08750.pdf, and this medium dataset consists of 25 images randomly chosen from each class.

  Images are chosen from the validation set rather than from the training set.
  """
  image_locations = os.path.join(ASSETS_PATH, 'full_400_16_class_imagenet_val_images')
  assert len(glob.glob('%s/*.JPEG'%image_locations))==400, 'Did not find exactly 400 images in %s'%image_locations
  image_name = glob.glob('%s/%d_*'%(image_locations, IMG_IDX))[0].split('/')[-1] # image_list[IMG_IDX]
  assert int(image_name.split('_')[0]) == IMG_IDX, 'Check the ordering for the images'
  img_pil = Image.open(os.path.join(image_locations, image_name))
  width, height = img_pil.size
  smallest_dim = min((width, height))
  left = (width - smallest_dim)/2
  right = (width + smallest_dim)/2
  top = (height - smallest_dim)/2
  bottom = (height + smallest_dim)/2
  img_pil = img_pil.crop((left, top, right, bottom))
  img_pil = img_pil.resize((im_shape,im_shape))
  img_pil.load()
  img1 = np.asarray(img_pil, dtype="float32")

  if data_format=='NCHW':
    img1 = np.rollaxis(np.array(img1),2,0)
  elif data_format=='NHWC':
    pass
  else:
    raise ValueError('Unsupported data_format %s'%data_format)

  image_dict = {}
  image_dict['image'] = img1
  image_dict['shape'] = im_shape
  image_dict['filename'] = os.path.join(image_locations, image_name)
  image_dict['filename_short'] = image_name
  image_dict['correct_response_idx'] = image_name.split('_')[1]
  image_dict['correct_response'] = image_name.split('_')[2]
  image_dict['imagenet_category'] = image_name.split('_')[3]
  image_dict['imagenet_image_id'] = image_name.split('_')[4].split('.')[0]
  image_dict['max_value_image_set'] = 255
  image_dict['min_value_image_set'] = 0
  return image_dict

def preproc_imagenet_center_crop(img1, im_shape=224, pil_preproc=False):
  """
  Returns a square portion of the image net image, taken from the center and rescaled to 224. 
  """
  image_shape_hw = img1.shape[0:2]
  smallest_dim = min(image_shape_hw)
  img1 = img1[int((image_shape_hw[0]/2-smallest_dim/2)):int((image_shape_hw[0]/2+smallest_dim/2)), 
              int((image_shape_hw[1]/2-smallest_dim/2)):int((image_shape_hw[1]/2+smallest_dim/2)), :]
  img1 = imresize(img1, (im_shape, im_shape))
  return img1

def make_400_16_class_imagenet_val_data():
  """
  Gets 25 examples for each of the 16 classes, removing some images if it seemed to not fit the 16-way class (or was cropped out), hand screened by jfeather
  Set used in Feather et al. 2022. 
  """
  
  from analysis_scripts.imagenet_16_categories_validation_paths import validation_paths

  random_seed = 517
  np.random.seed(random_seed)
  image_classes = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
  image_net_path = IMAGENET_PATH
  output_image_path = os.path.join(ASSETS_PATH, 'full_400_16_class_imagenet_val_images/')
  small_dataset_idx = 0

  total_per_class = 25 # grab 25 of each class, for 400 total
  image_path_dict = {}
    
  # list of stimuli that we decided to exclude, either because they contain multiple objects, 
  # the crop removes the object or the image is weird/inappropriate. 
  do_not_include_val_list = ['ilsvrc/val/n02690373/ILSVRC2012_val_00026189.JPEG', # includes truck
                             'ilsvrc/val/n03792782/ILSVRC2012_val_00001661.JPEG', # kind of inappropriate 
                             'ilsvrc/val/n03792782/ILSVRC2012_val_00045152.JPEG', # bike removed from crop
                             'ilsvrc/val/n02835271/ILSVRC2012_val_00000953.JPEG', # bike removed from crop
                             'ilsvrc/val/n03344393/ILSVRC2012_val_00017413.JPEG', # boat removed from crop
                             'ilsvrc/val/n04560804/ILSVRC2012_val_00022034.JPEG', # not actually a bottle
                             'ilsvrc/val/n02089078/ILSVRC2012_val_00031699.JPEG', # sideways, but a good boy anyway
                             'ilsvrc/val/n02504013/ILSVRC2012_val_00025043.JPEG', # has a bottle + elephant
                             'ilsvrc/val/n03085013/ILSVRC2012_val_00001071.JPEG', # black and white
                             'ilsvrc/val/n03085013/ILSVRC2012_val_00048307.JPEG', # many other objects in photo
                             'ilsvrc/val/n03041632/ILSVRC2012_val_00009369.JPEG', # bottle in foreground
                             'ilsvrc/val/n03041632/ILSVRC2012_val_00039665.JPEG', # includes bottles
                             'ilsvrc/val/n03041632/ILSVRC2012_val_00004379.JPEG', # knife cropped out? 
                             'ilsvrc/val/n04111531/ILSVRC2012_val_00020809.JPEG', # includes bottle
                             'ilsvrc/val/n03796401/ILSVRC2012_val_00025674.JPEG', # includes car
                             'ilsvrc/val/n03977966/ILSVRC2012_val_00021717.JPEG', # police car? 
                             'ilsvrc/val/n04461696/ILSVRC2012_val_00007770.JPEG', # includes part of car
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00041846.JPEG', # car not truck
                             'ilsvrc/val/n04461696/ILSVRC2012_val_00020258.JPEG', # towtruck with car
                             'ilsvrc/val/n03977966/ILSVRC2012_val_00043114.JPEG', # police car
                             'ilsvrc/val/n03977966/ILSVRC2012_val_00009917.JPEG', # police car
                             'ilsvrc/val/n02134418/ILSVRC2012_val_00019901.JPEG', # only claws 
                             'ilsvrc/val/n03344393/ILSVRC2012_val_00008048.JPEG', # includes chairs
                             'ilsvrc/val/n04560804/ILSVRC2012_val_00005897.JPEG', # not a bottle
                             'ilsvrc/val/n03983396/ILSVRC2012_val_00044297.JPEG', # weird crop
                             'ilsvrc/val/n03937543/ILSVRC2012_val_00023209.JPEG', # includes bird (should be bottle)
                             'ilsvrc/val/n04557648/ILSVRC2012_val_00041370.JPEG', # weird crop
                             'ilsvrc/val/n04560804/ILSVRC2012_val_00005459.JPEG', # not a bottle
                             'ilsvrc/val/n04560804/ILSVRC2012_val_00044778.JPEG', # not a bottle
                             'ilsvrc/val/n04429376/ILSVRC2012_val_00035142.JPEG', # contains clock (should be chair)
                             'ilsvrc/val/n04099969/ILSVRC2012_val_00048622.JPEG', # weird crop
                             'ilsvrc/val/n04548280/ILSVRC2012_val_00021551.JPEG', # contains bottle (should be clock)
                             'ilsvrc/val/n03196217/ILSVRC2012_val_00044847.JPEG', # oven clock
                             'ilsvrc/val/n03196217/ILSVRC2012_val_00025173.JPEG', # car clock
                             'ilsvrc/val/n03345487/ILSVRC2012_val_00023724.JPEG', # truck label, contains cars
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00032380.JPEG', # car not a truck
                             'ilsvrc/val/n03796401/ILSVRC2012_val_00032494.JPEG', # truck full of chairs
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00008139.JPEG', # dog in a truck
                             'ilsvrc/val/n04461696/ILSVRC2012_val_00013149.JPEG', # tow truck with car
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00005862.JPEG', # crop cannot tell if car or truck
                             'ilsvrc/val/n02133161/ILSVRC2012_val_00020482.JPEG', # bear removed from crop
                             'ilsvrc/val/n03344393/ILSVRC2012_val_00045522.JPEG', # cannot see boat
                             'ilsvrc/val/n04579145/ILSVRC2012_val_00008601.JPEG', # not a bottle
                             'ilsvrc/val/n04579145/ILSVRC2012_val_00030522.JPEG', # bottle removed from crop
                             'ilsvrc/val/n04579145/ILSVRC2012_val_00029100.JPEG', # bottle removed from crop
                             'ilsvrc/val/n02823428/ILSVRC2012_val_00031840.JPEG', # bird on the bottle
                             'ilsvrc/val/n03376595/ILSVRC2012_val_00008185.JPEG', # where is the chair? 
                             'ilsvrc/val/n04429376/ILSVRC2012_val_00041720.JPEG', # chair removed during crop
                             'ilsvrc/val/n04548280/ILSVRC2012_val_00041547.JPEG', # clock cropped
                             'ilsvrc/val/n02090721/ILSVRC2012_val_00044529.JPEG', # contains car
                             'ilsvrc/val/n02504013/ILSVRC2012_val_00017415.JPEG', # contains car
                             'ilsvrc/val/n03977966/ILSVRC2012_val_00020382.JPEG', # car not truck
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00043939.JPEG', # van not truck
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00023556.JPEG', # van
                             'ilsvrc/val/n03345487/ILSVRC2012_val_00001491.JPEG', # truck too far away
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00016527.JPEG', # car not truck
                             'ilsvrc/val/n03930630/ILSVRC2012_val_00017360.JPEG', # car and truck
                             'ilsvrc/val/n02835271/ILSVRC2012_val_00023266.JPEG', # black and white bike
                             'ilsvrc/val/n01582220/ILSVRC2012_val_00041135.JPEG', # confusing -- hard to see bird
                             'ilsvrc/val/n04429376/ILSVRC2012_val_00004705.JPEG', # contains gun
                             'ilsvrc/val/n04548280/ILSVRC2012_val_00014635.JPEG', # contains chair
                             'ilsvrc/val/n03196217/ILSVRC2012_val_00020409.JPEG', # too zoomed
                             'ilsvrc/val/n03977966/ILSVRC2012_val_00001143.JPEG', # police cars
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00034591.JPEG', # junkyard with cars
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00035864.JPEG', # van not truck
                             'ilsvrc/val/n03977966/ILSVRC2012_val_00000452.JPEG', # van not truck
                             'ilsvrc/val/n02124075/ILSVRC2012_val_00016918.JPEG', # too dark
                             'ilsvrc/val/n03770679/ILSVRC2012_val_00006759.JPEG', # van not a truck
                             'ilsvrc/val/n02123045/ILSVRC2012_val_00030706.JPEG', # black and white
                             'ilsvrc/val/n02690373/ILSVRC2012_val_00035783.JPEG', # contains boat as well as plane
                             'ilsvrc/val/n02504458/ILSVRC2012_val_00001177.JPEG', # contains birds on elephants
                            ]  

  for image_class_idx, image_class in enumerate(image_classes):
    image_path_dict[image_class] = []
    all_image = validation_paths[image_class]
    random_images_perm = np.random.permutation(range(len(all_image)))
    class_total = 0
    check_idx = 0
    
    while class_total<total_per_class:
        random_image = all_image[random_images_perm[check_idx]]
        img_pil = Image.open(os.path.join(image_net_path, random_image))
        if img_pil.mode!='RGB':
            check_idx+=1
            continue
        elif min(img_pil.size)<224:
            check_idx+=1
            continue
        elif random_image.split('images_complete/')[-1] in do_not_include_val_list:
            check_idx+=1
            continue
            
        image_id = random_image.split('_val_')[1].split('.')[0]
        class_id = random_image.split('val/')[1].split('/ILSVR')[0]
        image_path_dict[image_class].append(random_image.split('images_complete/')[-1])
        copyfile(os.path.join(image_net_path, random_image), os.path.join(output_image_path, '%d_%d_%s_%s_%s.JPEG'%(small_dataset_idx, image_class_idx, image_class, class_id, image_id)))
        small_dataset_idx+=1
        check_idx+=1
        class_total+=1
    
    with open(os.path.join(output_image_path, 'image_list_full_400_16_class_imagenet_val_images.json'), "w") as outfile: 
        json.dump(image_path_dict, outfile)
        json_object = json.dumps(image_path_dict, indent=4)  


def make_256_16_class_imagenet():
  random_seed = 517
  image_classes = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
  image_list_path = os.path.join(ASSETS_PATH, 'image_names_16_class_imagenet_training_data')
  image_net_path = os.path.join(IMAGENET_PATH, 'ilsvrc', 'train')
  output_image_path = os.path.join(ASSETS_PATH, 'full_256_16_class_imagenet/')
  small_dataset_idx = 0
  for image_class_idx, image_class in enumerate(image_classes):
    all_image = open(os.path.join(image_list_path, '%s.txt'%image_class)).read().splitlines()
    random_images_perm = np.random.permutation(range(len(all_image)))[10:26] # grab 16 of each class, so they do not overlap with a precious set. 
    random_images_names = [all_image[i] for i in random_images_perm]
    for random_image in random_images_names:
        class_name = random_image.split('_')[0]
        copyfile(os.path.join(image_net_path, class_name, random_image), os.path.join(output_image_path, '%d_%d_%s_%s'%(small_dataset_idx, image_class_idx, image_class, random_image)))
        small_dataset_idx+=1
  
def use_image_path_specified_image(IMG_IDX, image_path=None, image_class=None, 
                                   im_shape=224, data_format='NHWC'):
  print("Loading: %s"%image_path)
  img1 = scipy.misc.imread(image_path, mode='RGB')
  img1 = preproc_imagenet_center_crop(img1, im_shape=im_shape)

  if data_format=='NCHW':
    img1 = np.rollaxis(np.array(img1),2,0)
  elif data_format=='NHWC':
    pass
  else:
    raise ValueError('Unsupported data_format %s'%data_format)

  image_dict = {}
  image_dict['image'] = img1
  image_dict['shape'] = im_shape
  image_dict['filename'] = image_path
  image_dict['filename_short'] = image_path.split('/')[-1]
  image_dict['correct_response'] = image_class
  image_dict['max_value_image_set'] = 255
  image_dict['min_value_image_set'] = 0
  return image_dict


def use_pytorch_datasets(IMG_IDX, DATA, train_or_val='val', im_shape=224, data_format='NHWC'):
  print('Using pytorch dataset %s'%DATA)
  if im_shape !=224:
    raise NotImplementedError('Pytorch datasets not implemented for arbitrary shapes yet')

  from robustness import datasets

  DATA_PATH_DICT = { # Add additional datasets here if you want. 
      'ImageNet': IMAGENET_PATH, 
  } 

  BATCH_SIZE = 1
  if train_or_val == 'train':
    raise NotImplementedError('train subset is not implemented for pytorch datasets')
  elif train_or_val == 'val':
    only_val = True
  else:
    raise ValueError("train_or_val must be 'train' or 'val', currently set as %s"%train_or_val)

  dataset_function = getattr(datasets, DATA)
  dataset = dataset_function(DATA_PATH_DICT[DATA])
  train_loader, test_loader = dataset.make_loaders(workers=0, 
                                                   batch_size=BATCH_SIZE, 
                                                   data_aug=False,
                                                   subset_val=1,
                                                   subset_start=IMG_IDX,
                                                   shuffle_val=False,
                                                   only_val=only_val)
  data_iterator = enumerate(test_loader)
  _, (im, targ) = next(data_iterator) # Images to invert

  if data_format=='NCHW':
    im = np.array(im)
  elif data_format=='NHWC':
    im = np.rollaxis(np.array(im),1,4)
  else:
    raise ValueError('Unsupported data_format %s'%data_format)

  image_dict = {}
  image_dict['image'] = im
  image_dict['shape'] = im_shape
  image_dict['filename'] = 'pytorch_%s_%s_IMG_IDX'%(DATA, train_or_val)
  image_dict['filename_short'] = 'pytorch_%s_%s_IMG_IDX'%(DATA, train_or_val)
  image_dict['correct_response'] = targ
  image_dict['max_value_image_set'] = 1
  image_dict['min_value_image_set'] = 0
  return image_dict


def get_multiple_samples_pytorch_datasets(NUM_EXAMPLES, DATA, train_or_val='val', im_shape=224, data_format='NHWC', START_IDX=0):
  print('Using pytorch dataset %s'%DATA)
  if im_shape !=224:
    raise NotImplementedError('Pytorch datasets not implemented for arbitrary shapes yet')
  # This uses the robustness code to load the datasets in
  from robustness import datasets

  DATA_PATH_DICT = { # Add additional datasets here if you want.
      'ImageNet': IMAGENET_PATH,
  }

  BATCH_SIZE = 1
  if train_or_val == 'train':
    raise NotImplementedError('train subset is not implemented for pytorch datasets')
  elif train_or_val == 'val':
    only_val = True
  else:
    raise ValueError("train_or_val must be 'train' or 'val', currently set as %s"%train_or_val)

  dataset_function = getattr(datasets, DATA)
  dataset = dataset_function(DATA_PATH_DICT[DATA])

  train_loader, test_loader = dataset.make_loaders(workers=0,
                                                   batch_size=BATCH_SIZE,
                                                   data_aug=False,
                                                   subset_start=START_IDX,
                                                   shuffle_val=True,
                                                   only_val=only_val)
  data_iterator = enumerate(test_loader)
  _, (im, targ) = next(data_iterator) # Images to invert

  all_images = []
  correct_response = []
  for IMG_IDX in range(NUM_EXAMPLES):
    _, (im, targ) = next(data_iterator)
    if data_format=='NCHW':
      im = np.array(im)
    elif data_format=='NHWC':
      im = np.rollaxis(np.array(im),1,4)
    else:
      raise ValueError('Unsupported data_format %s'%data_format)
    all_images.append(im)
    correct_response.append(np.array(targ))

  image_dict = {}
  image_dict['shape'] = im_shape
  image_dict['max_value_image_set'] = 1
  image_dict['min_value_image_set'] = 0
  return all_images, correct_response, image_dict

def read_sound_file_list(filepath, remove_extension=False):
    """
    Takes in a text file with one sound on each line. Returns a python list with one element for each line of the file. Option to remove the extension at the end of the filename.

    Inputs
    ------
    filepath : string
        The path to the text file containing a list of sounds
    remove_exnention : Boolean
        If true, removes the extension (if it exists) from the list of sounds.

    Returns
    -------
    all_sounds : list
        The sounds within filepath as a list.

    """
    with open(filepath,'r') as f:
        all_sounds = f.read().splitlines()
    if remove_extension:
        for sound_idx, sound in enumerate(all_sounds):
            all_sounds[sound_idx] = sound.split('.')[0]
    return all_sounds
