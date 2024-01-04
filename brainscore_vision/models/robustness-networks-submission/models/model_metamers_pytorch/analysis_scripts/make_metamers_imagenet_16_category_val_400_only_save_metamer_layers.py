"""
Runs metamer generation on all of the layers specified in build_network.py
Removes additional layers from saving to reduce the file size

You can either make a copy of this file into the directory with the build_network.py file
and run it from there directly, or specify the model directory containing the build_network.py
file as an argument (-D). The script will create a folder structure for the generated metamers
in the directory specified, or in the directory it is called from, if no directory is specified.
"""

import torch
import os

from analysis_scripts.input_helpers import generate_import_image_functions
from robustness.model_utils import make_and_restore_model
from analysis_scripts.helpers_16_choice import force_16_choice

import numpy as np
from robustness.model_utils import make_and_restore_model
import csv

from matplotlib import pylab as plt

import importlib.util
import scipy 

from robustness import custom_synthesis_losses

import argparse
import pickle

from PIL import Image
from robustness.tools.label_maps import CLASS_DICT
from robustness.tools.distance_measures import * 

def preproc_image(image, image_dict):
    """The image into the pytorch model should be between 0-1"""
    if image_dict['max_value_image_set']==255:
        image = image/255.
    return image
    

def calc_loss(model, inp, target, custom_loss, should_preproc=True):
    '''
    Modified from the Attacker module of Robustness. 
    Calculates the loss of an input with respect to target labels
    Uses custom loss (if provided) otherwise the criterion
    '''
    if should_preproc:
        inp = model.preproc(inp)
    return custom_loss(model.model, inp, target)

def run_audio_metamer_generation(SIDX, LOSS_FUNCTION, INPUTIMAGEFUNCNAME, RANDOMSEED, overwrite_pckl,
                                 use_dataset_preproc, step_size, NOISE_SCALE, ITERATIONS, NUMREPITER,
                                 OVERRIDE_SAVE, MODEL_DIRECTORY):
    if MODEL_DIRECTORY is None:
        import build_network
        MODEL_DIRECTORY = '' # use an empty string to append to saved files.
    else:
        build_network_spec = importlib.util.spec_from_file_location("build_network",
            os.path.join(MODEL_DIRECTORY, 'build_network.py'))
        build_network = importlib.util.module_from_spec(build_network_spec)
        build_network_spec.loader.exec_module(build_network)

    predictions_out_dict = {}
    rep_out_dict = {}
    all_outputs_out_dict = {}
    xadv_dict = {}
    all_losses_dict = {}
    predicted_labels_out_dict = {}
    predicted_16_cat_labels_out_dict = {}
    
    BATCH_SIZE=1 # TODO(jfeather): remove batch references -- they are unnecessary and not used.  
    NUM_WORKERS=1
    
    torch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    
    model, ds, metamer_layers = build_network.main(return_metamer_layers=True) 
    
    # imagenet_idx_to_wnid = {v:k for k, v in ds.wnid_to_idx.items()}
    
    # Get the WNID
    with open('/om4/group/mcdermott/user/jfeather/projects/model_metamers/16-class-ImageNet/wordnetID_to_human_identifier.txt', mode='r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        wnid_imagenet_name = {rows[0]:rows[1] for rows in reader}
    
    # Load dataset for metamer generation
    INPUTIMAGEFUNC = generate_import_image_functions(INPUTIMAGEFUNCNAME, data_format='NCHW')
    image_dict = INPUTIMAGEFUNC(SIDX)
    image_dict['image_orig'] = image_dict['image']
    # Preprocess to be in the format for pytorch
    image_dict['image'] = preproc_image(image_dict['image'], image_dict)
    if use_dataset_preproc: # Apply for some models, for instance if we have greyscale images or different sizes. 
        image_dict['image'] = ds.transform_test(Image.fromarray(np.rollaxis(np.uint8(image_dict['image']*255),0,3)))
        scale_image_save_PIL_factor = ds.scale_image_save_PIL_factor
        init_noise_mean = ds.init_noise_mean
    else:
        scale_image_save_PIL_factor = 255
        init_noise_mean = 0.5
    
    # Add a batch dimension to the input image 
    im = torch.tensor(np.expand_dims(image_dict['image'],0)).float().contiguous()
    
    # Label name for the 16 way imagenet task
    label_name = image_dict['correct_response']
    
    # Getthe full imagnet keys for printing predictions
    label_keys = CLASS_DICT['ImageNet'].keys()
    label_values = CLASS_DICT['ImageNet'].values()
    label_idx = list(label_keys)[list(label_values).index(wnid_imagenet_name[image_dict['imagenet_category']])]
    targ = torch.from_numpy(np.array([label_idx])).float()
    
    # Set up the saving folder and make sure that the file doesn't already exist
    synth_name = INPUTIMAGEFUNCNAME+'_'+LOSS_FUNCTION + '_RS%d'%RANDOMSEED + '_I%d'%ITERATIONS + '_N%d'%NUMREPITER
    base_filepath = os.path.join(MODEL_DIRECTORY, 'metamers/%s/%d_SOUND_%s/'%(synth_name, SIDX, label_name))
    pckl_path = base_filepath + '/all_metamers_pickle.pckl'
    try:
        os.makedirs(base_filepath)
    except:
        pass
    
    if os.path.isfile(pckl_path) and not overwrite_pckl:
        raise FileExistsError('The file %s already exists, and you are not forcing overwriting'%pckl_path)
    
    # Send model to GPU (b/c we haven't loaded a model, so it is not on the GPU)
    model = model.cuda()
    model.eval()
    
    # TODO: make this more permanent/flexible
    # Here because dropout may help optimization for some types of losses
    try:
        model.disable_dropout_functions()
        print('Turning off dropout functions because we are measuring activations')
    except:
        pass
    
    with torch.no_grad():
        (predictions, rep, all_outputs), orig_im = model(im.cuda(), with_latent=True, fake_relu=True) # Corresponding representation
    
    # Calculate human readable labels and 16 category labels for the original image
    orig_predictions = []
    for b in range(BATCH_SIZE):
        try:
            orig_predictions.append(predictions[b].detach().cpu().numpy())
        except KeyError:
            orig_predictions.append(predictions['signal/word_int'][b].detach().cpu().numpy())
    
    # Get the predicted 16 category label
    orig_16_cat_prediction = [force_16_choice(np.flip(np.argsort(p.ravel(),0)),
                                              CLASS_DICT['ImageNet']) for p in orig_predictions]
    print('Orig Image 16 Category Prediction: %s'%(
           orig_16_cat_prediction))
    
    # Make the noise input (will use for all of the input seeds)
    # the noise scale is typically << the noise mean, so we don't have to worry about negative values. 
    im_n_initialized = (torch.randn_like(im)*NOISE_SCALE + init_noise_mean).detach().cpu().numpy()
    
    for layer_to_invert in metamer_layers:
        # Choose the inversion parameters (will run 4x the iterations, reducing the learning rate each time)
        synth_kwargs = {
            'custom_loss': custom_synthesis_losses.LOSSES[LOSS_FUNCTION](layer_to_invert, normalize_loss=True),
            'constraint':'2',
            'eps':100000,
            'step_size': step_size,
            'iterations': ITERATIONS,
            'do_tqdm': False,
            'targeted': True,
            'use_best': False
        }
    
        if hasattr(synth_kwargs['custom_loss'], 'enable_dropout_flag'):
            model.enable_dropout_flag = synth_kwargs['custom_loss'].enable_dropout_flag
            model.enable_dropout_functions = synth_kwargs['custom_loss']._enable_dropout_functions
            model.disable_dropout_functions = synth_kwargs['custom_loss']._disable_dropout_functions
    
        # Use same noise for every layer.
    #     im_n = torch.clamp(torch.from_numpy(im_n_initialized), 0, 1).cuda()
        im_n = torch.clamp(torch.from_numpy(im_n_initialized), ds.min_value, ds.max_value).cuda()
        invert_rep = all_outputs[layer_to_invert].contiguous().view(all_outputs[layer_to_invert].size(0), -1)
    
        # Do the optimization, and save the losses occasionally
        all_losses = {}
    
        this_loss, _ = calc_loss(model, im_n, invert_rep.clone(), synth_kwargs['custom_loss'])
        all_losses[0] = this_loss.detach().cpu()
        print('Step %d | Layer %s | Loss %f'%(0, layer_to_invert, this_loss))
        # Here because dropout may help optimization for some types of losses
        try:
            model.enable_dropout_functions()
            print('Turning on dropout functions because we are starting synthesis')
        except:
            pass
        (predictions_out, rep_out, all_outputs_out), xadv = model(im_n, invert_rep.clone(), make_adv=True, **synth_kwargs, with_latent=True, fake_relu=True) 
        this_loss, _ = calc_loss(model, xadv, invert_rep.clone(), synth_kwargs['custom_loss'])
        all_losses[synth_kwargs['iterations']] = this_loss.detach().cpu()
        print('Step %d | Layer %s | Loss %f'%(synth_kwargs['iterations'], layer_to_invert, this_loss))
        for i in range(NUMREPITER-1):
            try:
                synth_kwargs['custom_loss'].optimization_count=0
            except:
                pass
    
            if i==NUMREPITER-2: # Turn off dropout for the last pass through
                # TODO: make this more permanent/flexible
                # Here because dropout may help optimization for some types of losses
                try:
                    model.disable_dropout_functions()
                    print('Turning off dropout functions because it is the last optimization pass through')
                except:
                    pass
    
            im_n = xadv
            synth_kwargs['step_size'] = synth_kwargs['step_size']/2
            (predictions_out, rep_out, all_outputs_out), xadv = model(im_n, invert_rep.clone(), make_adv=True, **synth_kwargs, with_latent=True, fake_relu=True) # Image inversion using PGD
            this_loss, _ = calc_loss(model, xadv, invert_rep.clone(), synth_kwargs['custom_loss'])
            all_losses[(i+2)*synth_kwargs['iterations']] = this_loss.detach().cpu()
            print('Step %d | Layer %s | Loss %f'%(synth_kwargs['iterations']*(i+2), layer_to_invert, this_loss))
        
        if type(predictions_out)==dict:
            predictions_out_dict[layer_to_invert] = {}
            for key, value in predictions_out.items():
                predictions_out[key] = value.detach().cpu()
            predictions_out_dict[layer_to_invert] = predictions_out
        else:
            predictions_out_dict[layer_to_invert] = predictions_out.detach().cpu()
        try:
            rep_out_dict[layer_to_invert] = rep_out.detach().cpu()
        except AttributeError:
            rep_out_dict[layer_to_invert] = rep_out
    
        for key in all_outputs_out:
            if type(all_outputs_out[key])==dict:
                for dict_key, dict_value in all_outputs_out[key].items():
                    if '%s/%s'%(key, dict_key) in metamer_layers: 
                        all_outputs_out[key][dict_key] = dict_value.detach().cpu()
                    else:
                        all_outputs_out[key][dict_key] = None
            else:
                if key in metamer_layers:
                    all_outputs_out[key] = all_outputs_out[key].detach().cpu() 
                else:
                    all_outputs_out[key] = None
    
        # Calculate the predictions and save them in the dictioary
        synth_predictions = []
        for b in range(BATCH_SIZE):
            try:
                synth_predictions.append(predictions_out[b].detach().cpu().numpy())
            except KeyError:
                synth_predictions.append(predictions['signal/word_int'][b].detach().cpu().numpy())
    
        # Get the predicted 16 category label
        synth_16_cat_prediction = [force_16_choice(np.flip(np.argsort(p.ravel(),0)),
                                                   CLASS_DICT['ImageNet']) for p in synth_predictions]
        print('Layer %s, Synth Image 16 Category Prediction: %s'%(
              layer_to_invert, synth_16_cat_prediction))
    
        all_outputs_out_dict[layer_to_invert] = all_outputs_out
        xadv_dict[layer_to_invert] = xadv.detach().cpu()
        all_losses_dict[layer_to_invert] = all_losses
        predicted_labels_out_dict[layer_to_invert] = synth_predictions
        predicted_16_cat_labels_out_dict[layer_to_invert] = synth_16_cat_prediction
    
    pckl_output_dict = {}
    pckl_output_dict['predictions_out_dict'] = predictions_out_dict
    pckl_output_dict['rep_out_dict'] = rep_out_dict
    pckl_output_dict['all_outputs_out_dict'] = all_outputs_out_dict
    pckl_output_dict['xadv_dict'] = xadv_dict
    pckl_output_dict['image_dict'] = image_dict
    pckl_output_dict['RANDOMSEED'] = RANDOMSEED
    pckl_output_dict['metamer_layers'] = metamer_layers
    pckl_output_dict['all_losses'] = all_losses
    pckl_output_dict['ITERATIONS'] = ITERATIONS
    pckl_output_dict['NUMREPITER'] = NUMREPITER
    pckl_output_dict['predicted_16_cat_labels_out_dict'] = predicted_16_cat_labels_out_dict
    pckl_output_dict['predicted_labels_out_dict'] = predicted_labels_out_dict
    pckl_output_dict['orig_16_cat_prediction'] = orig_16_cat_prediction
    pckl_output_dict['orig_predictions'] = orig_predictions
    pckl_output_dict['NOISE_SCALE'] = NOISE_SCALE
    pckl_output_dict['step_size'] = step_size
    
    for key in all_outputs:
        if type(all_outputs[key])==dict:
            for dict_key, dict_value in all_outputs[key].items():
                if '%s/%s'%(key, dict_key) in metamer_layers:
                    all_outputs[key][dict_key] = dict_value.detach().cpu()
                else:
                    all_outputs[key][dict_key] = None
        else:
            if key in metamer_layers:
                all_outputs[key] = all_outputs[key].detach().cpu()
            else:
                all_outputs[key] = None
    
    pckl_output_dict['all_outputs_orig'] = all_outputs
    if type(predictions)==dict:
        for dict_key, dict_value in predictions.items():
            predictions[dict_key] = dict_value.detach().cpu()
    else:
        predictions = predictions.detach().cpu()
    pckl_output_dict['predictions_orig'] = predictions
    if type(rep)==dict:
        for dict_key, dict_value in rep.items():
            if rep is not None:
                rep[dict_key] = dict_value.detach().cpu()
    else:
        if rep is not None:
            rep = rep.detach().cpu()
    pckl_output_dict['rep_orig'] = rep
    pckl_output_dict['sound_orig'] = orig_im.detach().cpu()
    
    # Just use the name of the loss to save synthkwargs don't save the function
    synth_kwargs['custom_loss'] = LOSS_FUNCTION
    pckl_output_dict['synth_kwargs'] = synth_kwargs
    
    # Calculate distance measures for each layer, use the cpu versions
    all_distance_measures = {}
    for layer_to_invert in metamer_layers:
        all_distance_measures[layer_to_invert] = {}
        for layer_to_measure in metamer_layers: # pckl_output_dict['all_outputs_orig'].keys():
            met_rep = pckl_output_dict['all_outputs_out_dict'][layer_to_invert][layer_to_measure].numpy().copy().ravel()
            orig_rep = pckl_output_dict['all_outputs_orig'][layer_to_measure].numpy().copy().ravel()
            spearman_rho = compute_spearman_rho_pair([met_rep, orig_rep])
            pearson_r = compute_pearson_r_pair([met_rep, orig_rep])
            dB_SNR, norm_signal, norm_noise = compute_snr_db([orig_rep, met_rep])
            all_distance_measures[layer_to_invert][layer_to_measure] = {
                                            'spearman_rho': spearman_rho,
                                            'pearson_r':pearson_r, 
                                            'dB_SNR':dB_SNR,
                                            'norm_signal':norm_signal,
                                            'norm_noise':norm_noise,
                                           }
            if layer_to_invert == layer_to_measure:
                print('Layer %s'%layer_to_measure)
                print(all_distance_measures[layer_to_invert][layer_to_measure])
    pckl_output_dict['all_distance_measures'] = all_distance_measures     
    
    with open(pckl_path, 'wb') as handle:
        pickle.dump(pckl_output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Make plots and save the image files
    for layer_idx, layer_to_invert in enumerate(metamer_layers):
        layer_filepath = base_filepath + '%d_layer_%s'%(layer_idx, layer_to_invert)
        rep_out = rep_out_dict[layer_to_invert]
        predictions_out = predictions_out_dict[layer_to_invert]
        xadv = xadv_dict[layer_to_invert]
        all_outputs_out = all_outputs_out_dict[layer_to_invert]
    
        fig = plt.figure(figsize=(BATCH_SIZE*5,12))
        for i in range(BATCH_SIZE):
            # Get labels to use for the plots
            try:
                orig_predictions = predictions[i].detach().cpu().numpy()
                synth_predictions = predictions_out[i].detach().cpu().numpy()
            except KeyError:
                orig_predictions = predictions['signal/word_int'][i].detach().cpu().numpy()
                synth_predictions = predictions['signal/word_int'][i].detach().cpu().numpy()
    
            # Get the predicted 16 category label
            orig_16_cat_prediction = force_16_choice(np.flip(np.argsort(orig_predictions.ravel(),0)),
                                                     CLASS_DICT['ImageNet'])
            synth_16_cat_prediction = force_16_choice(np.flip(np.argsort(synth_predictions.ravel(),0)),
                                                      CLASS_DICT['ImageNet'])
            print('Layer %s, Image %d, Orig Image 16 Category Prediction: %s'%(
                  layer_to_invert, i, orig_16_cat_prediction))
            print('Layer %s, Image %d, Synth Image 16 Category Prediction: %s'%(
                  layer_to_invert, i, synth_16_cat_prediction))
    
            # Set synthesis sucess based on the 16 category labels -- we can later evaluate the 1000 category labels. 
            synth_success = orig_16_cat_prediction == synth_16_cat_prediction
    
            orig_label = np.argmax(orig_predictions)
            synth_label = np.argmax(synth_predictions)
    
            plt.subplot(4, BATCH_SIZE, i+1)
            if im[i].shape[0] == 3:
                plt.imshow((np.rollaxis(np.array(im[i].cpu().numpy()),0,3)), interpolation='none')
            elif im[i].shape[0] == 1:
                plt.imshow((np.array(im[i].cpu().numpy())[0,:,:]), interpolation='none', cmap='gray')
            else: 
                raise ValueError('Image dimensions are not appropriate for saving. Check dimensions')
            plt.title('Layer %s, Image %d, Predicted Orig Label "%s" \n Orig Coch'%(layer_to_invert, i, # CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())]))
                                                                               CLASS_DICT['ImageNet'][int(orig_label)]))
    
    #     for i in range(BATCH_SIZE):
            plt.subplot(4, BATCH_SIZE, BATCH_SIZE+i+1)
            if im[i].shape[0] == 3:
                plt.imshow((np.rollaxis(np.array(xadv[i].cpu().numpy()),0,3)), interpolation='none')
            elif im[i].shape[0] == 1:
                plt.imshow((np.array(xadv[i].cpu().numpy())[0,:,:]), interpolation='none', cmap='gray')
            else:
                raise ValueError('Image dimensions are not appropriate for saving. Check dimensions')
            plt.title('Layer %s, Image %d, Predicted Synth Label "%s" \n Synth Coch'%(layer_to_invert, i, # CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())]))
                                                                               CLASS_DICT['ImageNet'][int(synth_label)]))
    
    #     for i in range(BATCH_SIZE):
            plt.subplot(4, BATCH_SIZE, BATCH_SIZE*2+i+1)
            plt.scatter(np.ravel(np.array(all_outputs[layer_to_invert].cpu())[i,:]), np.ravel(all_outputs_out[layer_to_invert].cpu().detach().numpy()[i,:]))
            plt.title('Layer %s, Image %d, Label "%s" \n Optimization'%(layer_to_invert, i, CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())]))
            plt.xlabel('Orig Activations (%s)'%layer_to_invert)
            plt.ylabel('Synth Activations (%s)'%layer_to_invert)
    
    #     for i in range(BATCH_SIZE):
            plt.subplot(4, BATCH_SIZE, BATCH_SIZE*3+i+1)
            if type(all_outputs['final'])==dict:
                dict_keys = list(all_outputs['final'].keys()) # So we ensure the same order
                plot_outputs_final = np.concatenate([np.array(all_outputs['final'][task_key].cpu()[i,:]).ravel() for task_key in dict_keys])
                plot_outputs_out_final = np.concatenate([np.array(all_outputs_out['final'][task_key].cpu().detach().numpy()[i,:]).ravel() for task_key in dict_keys]) 
                plt.scatter(plot_outputs_final.ravel(), plot_outputs_out_final.ravel()) 
            else:
                plt.scatter(np.ravel(np.array(all_outputs['final'].cpu())[i,:]), np.ravel(all_outputs_out['final'].cpu().detach().numpy()[i,:]))
            plt.title('Layer %s, Image %d, Label "%s" \n Optimization'%(layer_to_invert, i, CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())]))
            plt.xlabel('Orig Activations (Final Layer)')
            plt.ylabel('Synth Activations (Final Layer)')
    
            fig.savefig(layer_filepath + '_image_optimization.png')
    
    #     for i in range(BATCH_SIZE):
            try:
                print('Layer %s, Image %d, Label "%s", Prediction Orig "%s", Prediction Synth "%s"'%(
                                                              layer_to_invert, i,
                                                              CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())], 
                                                              CLASS_DICT['ImageNet'][int(np.argmax(predictions[i].detach().cpu().numpy()))],
                                                              CLASS_DICT['ImageNet'][int(np.argmax(predictions_out[i].detach().cpu().numpy()))]))
            except KeyError:
                print('Layer %s, Image %d, Label "%s", Prediction Orig "%s", Prediction Synth "%s"'%(
                                                              layer_to_invert, i,
                                                              CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())],
                                                              CLASS_DICT['ImageNet'][int(np.argmax(predictions['signal/word_int'][i].detach().cpu().numpy()))],
                                                              CLASS_DICT['ImageNet'][int(np.argmax(predictions_out['signal/word_int'][i].detach().cpu().numpy()))]))
    
            plt.close()
    
            if layer_idx==0:
                if im[i].shape[0] == 3:
                    orig_image = Image.fromarray((np.rollaxis(np.array(im[i].cpu().numpy()),0,3)*scale_image_save_PIL_factor).astype('uint8'))
                elif im[i].shape[0] == 1:
                    orig_image = Image.fromarray((np.array(im[i].cpu().numpy())[0]*scale_image_save_PIL_factor).astype('uint8'))
                orig_image.save(base_filepath + '/orig.png', 'PNG')
         
            # Only save the individual image if the layer optimization succeeded
            if synth_success or OVERRIDE_SAVE:
                if im[i].shape[0] == 3:
                    synth_image = Image.fromarray((np.rollaxis(np.array(xadv[i].cpu().numpy()),0,3)*scale_image_save_PIL_factor).astype('uint8'))
                elif im[i].shape[0] == 1:
                    synth_image = Image.fromarray((np.array(xadv[i].cpu().numpy())[0]*scale_image_save_PIL_factor).astype('uint8'))
                synth_image.save(layer_filepath + '_synth.png', 'PNG')


def main(raw_args=None):
    #########PARSE THE ARGUMENTS FOR THE FUNCTION#########
    parser = argparse.ArgumentParser(description='Input the sound indices and the layers to match')
    parser.add_argument('SIDX',metavar='I',type=int,help='index into the sound list for the time_average sound')
    parser.add_argument('-L', '--LOSSFUNCTION', metavar='--L', type=str, default='inversion_loss_layer', help='loss function to use for the synthesis')
    parser.add_argument('-F', '--INPUTIMAGEFUNC', metavar='--A', type=str, default='400_16_class_imagenet_val', help='function to use for grabbing the input image sources')
    parser.add_argument('-R', '--RANDOMSEED', metavar='--R', type=int, default=0, help='random seed to use for synthesis')
    parser.add_argument('-I', '--ITERATIONS', metavar='--I', type=int, default=3000, help='number of iterations in robustness synthesis kwargs')
    parser.add_argument('-N', '--NUMREPITER', metavar='--N', type=int, default=8, help='number of repetitions to run the robustness synthesis, each time decreasing the learning rate by half')
    parser.add_argument('-S', '--OVERRIDE_SAVE', metavar='--S', type=bool, default=False, help='set to true to save, even if the optimization does not succeed')
    parser.add_argument('-O', '--OVERWRITE_PICKLE', action='store_true', help='set to true to overwrite the saved pckl file, if false then exits out if the file already exists')
    parser.add_argument('-DP', '--DATASET_PREPROC', action='store_true', dest='use_dataset_preproc')
    parser.add_argument('-E', '--NOISE_SCALE', metavar='--E', type=float, default=1/20, help='multiply the noise by this value for the synthesis initialization')
    parser.add_argument('-Z', '--STEP_SIZE', metavar='--Z', type=float, default=1, help='Initial step size for the metamer generation')
    parser.add_argument('-D', '--DIRECTORY', metavar='--D', type=str, default=None, help='The directory with the location of the `build_network.py` file. Folder structure for saving metamers will be created in this directory. If not specified, assume this script is located in the same directory as the build_network.py file.')

    args=parser.parse_args(raw_args)
    SIDX = args.SIDX
    LOSS_FUNCTION = args.LOSSFUNCTION
    INPUTIMAGEFUNCNAME= args.INPUTIMAGEFUNC
    RANDOMSEED = args.RANDOMSEED
    overwrite_pckl = args.OVERWRITE_PICKLE
    use_dataset_preproc = args.use_dataset_preproc
    step_size = args.STEP_SIZE
    ITERATIONS = args.ITERATIONS
    NUMREPITER = args.NUMREPITER
    NOISE_SCALE = args.NOISE_SCALE
    OVERRIDE_SAVE = args.OVERRIDE_SAVE
    MODEL_DIRECTORY = args.DIRECTORY

    run_audio_metamer_generation(SIDX, LOSS_FUNCTION, INPUTIMAGEFUNCNAME, RANDOMSEED, overwrite_pckl,
                                 use_dataset_preproc, step_size, NOISE_SCALE, ITERATIONS, NUMREPITER,
                                 OVERRIDE_SAVE, MODEL_DIRECTORY)

if __name__ == '__main__':
    main()
