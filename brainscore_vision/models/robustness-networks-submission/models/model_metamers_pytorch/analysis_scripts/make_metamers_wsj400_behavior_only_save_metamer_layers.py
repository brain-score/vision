"""
Runs audio metamer generation on all of the layers specified in build_network.py
Removes additional layers from saving to reduce the file size

You can either make a copy of this file into the directory with the build_network.py file
and run it from there directly, or specify the model directory containing the build_network.py
file as an argument (-D). The script will create a folder structure for the generated metamers
in the directory specified, or in the directory it is called from, if no directory is specified. 
"""

import torch
import os

from analysis_scripts.input_helpers import generate_import_audio_functions
from robustness.model_utils import make_and_restore_model

from matplotlib import pylab as plt
import numpy as np

import importlib.util
import scipy 

from robustness.audio_functions import audio_transforms
from scipy.io import wavfile

from robustness import custom_synthesis_losses
from robustness.tools.distance_measures import *

import argparse
import pickle

def preproc_sound_np(sound):
    '''
    Sounds going into the pytorch models are normalized to have rms=0.1
    Additional preprocessing happens inside of the model loop. 
    '''
    sound = sound - np.mean(sound)
    sound = sound/np.sqrt(np.mean(sound**2))*0.1
    sound = np.expand_dims(sound, 0)
    sound = torch.from_numpy(sound).float().cuda()
    return sound

def calc_loss(model, inp, target, custom_loss, should_preproc=True):
    '''
    Modified from the Attacker module of Robustness. 
    Calculates the loss of an input with respect to target
    Uses custom loss (if provided) otherwise the criterion
    '''
    if should_preproc:
        inp = model.preproc(inp)
    return custom_loss(model.model, inp, target)

def run_audio_metamer_generation(SIDX, LOSS_FUNCTION, INPUTAUDIOFUNCNAME, RANDOMSEED, 
                                 STRICT, overwrite_pckl, step_size, NOISE_SCALE, 
                                 ITERATIONS, NUMREPITER, OVERRIDE_SAVE, MODEL_DIRECTORY):
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
    
    BATCH_SIZE=1 # TODO(jfeather): remove batch references -- they are unnecessary and not used.
    NUM_WORKERS=1
    
    torch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    
    # Currently transforms are unused, but could be incorporated at a future time. 
    # TRANSFORMS_TEST_NO_BACKGROUND=audio_transforms.AudioCompose([
    #      audio_transforms.AudioToTensor(),
    #      audio_transforms.RMSNormalizeForegroundAndBackground(rms_level=0.1),
    #      audio_transforms.UnsqueezeAudio(dim=0),
    #      ])
    
    model, ds, metamer_layers = build_network.main(return_metamer_layers=True,
                                                   include_identity_sequential=False,
    #                                                ds_kwargs={'transform_test':TRANSFORMS_TEST_NO_BACKGROUND, 
    #                                                          },
                                                   strict=STRICT,
                                                  )
    
    # We have many different GPUs and the input size doesn't change, so run the autotuner
    torch.backends.cudnn.benchmark = True
    
    print(ds.transform_test)
    # Load dataset for metamer generation
    INPUTAUDIOFUNC = generate_import_audio_functions(INPUTAUDIOFUNCNAME, rms_normalize=0.1)
    audio_dict = INPUTAUDIOFUNC(SIDX, SR=ds.SR)
    audio_dict['wav_orig'] = audio_dict['wav'].copy()
    # audio_dict['wav'] = preproc_sound_np(audio_dict['wav_orig'])
    # Use the dataset transforms, but we are not including any background for the metamer generation
    audio_dict['wav'], _ = ds.transform_test(audio_dict['wav_orig'], None)
    assert audio_dict['SR']==ds.SR, 'Metamer input sound SR is %d while dataset SR is %d'%(audio_dict['SR'], ds.SR)
    
    # This will not work if the inputs are batched.
    im = audio_dict['wav'].float()
    label_keys = ds.label_mapping.keys()
    label_values = ds.label_mapping.values()
    label_idx = list(label_keys)[list(label_values).index(audio_dict['correct_response'])]
    targ = torch.from_numpy(np.array([label_idx])).float()
    
    # Set up saving and check that the file doesn't already exist (exit if it does)
    synth_name = INPUTAUDIOFUNCNAME+'_'+LOSS_FUNCTION + '_RS%d'%RANDOMSEED + '_I%d'%ITERATIONS + '_N%d'%NUMREPITER
    base_filepath = os.path.join(MODEL_DIRECTORY, 'metamers/%s/%d_SOUND_%s/'%(synth_name, SIDX, ds.label_mapping[int(targ[0].cpu().numpy())]))
    try:
        os.makedirs(base_filepath)
    except:
        pass
    
    pckl_path = base_filepath + '/all_metamers_pickle.pckl'
    if os.path.isfile(pckl_path) and not overwrite_pckl:
        raise FileExistsError('The file %s already exists, and you are not forcing overwriting'%pckl_path)
    
    # Send model to GPU (b/c we haven't loaded a model, so it is not on the GPU)
    model = model.cuda()
    model.eval()
    
    # Here because dropping out some units may help optimization for some types of losses and models. 
    # Specfically used for the spectrotemporal model. 
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
    
    # Get the predicted category label
    orig_cat_prediction = [ds.label_mapping[np.argmax(p.ravel())] for p in orig_predictions]
    
    print('Orig Audio Category Prediction: %s'%(
           orig_cat_prediction))
    # Make the noise input (will use for all of the input seeds)
    im_n_initialized = ((torch.randn_like(im)) * NOISE_SCALE ).detach().cpu().numpy()
    # im_n_initialized = ((torch.rand_like(im)-0.5) * NOISE_SCALE ).detach().cpu().numpy()
    
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
        im_n = torch.from_numpy(im_n_initialized).cuda()
        invert_rep = all_outputs[layer_to_invert].contiguous().view(all_outputs[layer_to_invert].size(0), -1)
    
        # Do the optimization, and save the losses occasionally
        all_losses = {}
    
        this_loss, _ = calc_loss(model, im_n, invert_rep.clone(), synth_kwargs['custom_loss'])
        all_losses[0] = this_loss.detach().cpu()
        print('Step %d | Layer %s | Loss %f'%(0, layer_to_invert, this_loss), flush=True)
        # TODO: this performs more computation than necessary at each step, because we run the 
        # whole forward graph. Speed could be significantly improved by only running the model up
        # until the layer where we generate the metamers. 
    
        # Here because dropout may help optimization for some types of losses
        try:
            model.enable_dropout_functions()
            print('Turning on dropout functions because we are starting synthesis')
        except:
            pass
    
        (predictions_out, rep_out, all_outputs_out), xadv = model(im_n, invert_rep.clone(), make_adv=True, **synth_kwargs, with_latent=True, fake_relu=True) 
        this_loss, _ = calc_loss(model, xadv, invert_rep.clone(), synth_kwargs['custom_loss'])
        all_losses[synth_kwargs['iterations']] = this_loss.detach().cpu()
        print('Step %d | Layer %s | Loss %f'%(synth_kwargs['iterations'], layer_to_invert, this_loss), flush=True)
        for i in range(NUMREPITER-1):
            try: 
                synth_kwargs['custom_loss'].optimization_count=0
            except:
                pass
                
            if i==NUMREPITER-2: # Turn off dropout for the last pass through
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
            print('Step %d | Layer %s | Loss %f'%(synth_kwargs['iterations']*(i+2), layer_to_invert, this_loss), flush=True)
        
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
    
        # Get the predicted category label
        synth_cat_prediction  = [ds.label_mapping[np.argmax(p.ravel())] for p in synth_predictions]
    
        print('Layer %s, Synth Audio Category Prediction: %s'%(
              layer_to_invert, synth_cat_prediction))
    
        all_outputs_out_dict[layer_to_invert] = all_outputs_out
        xadv_dict[layer_to_invert] = xadv.detach().cpu()
        all_losses_dict[layer_to_invert] = all_losses
        predicted_labels_out_dict[layer_to_invert] = synth_predictions
    
    pckl_output_dict = {}
    pckl_output_dict['predictions_out_dict'] = predictions_out_dict
    pckl_output_dict['rep_out_dict'] = rep_out_dict
    pckl_output_dict['all_outputs_out_dict'] = all_outputs_out_dict
    pckl_output_dict['xadv_dict'] = xadv_dict
    pckl_output_dict['audio_dict'] = audio_dict
    pckl_output_dict['RANDOMSEED'] = RANDOMSEED
    pckl_output_dict['label_idx'] = label_idx
    pckl_output_dict['metamer_layers'] = metamer_layers
    pckl_output_dict['all_losses'] = all_losses
    pckl_output_dict['ITERATIONS'] = ITERATIONS
    pckl_output_dict['NUMREPITER'] = NUMREPITER
    pckl_output_dict['predicted_labels_out_dict'] = predicted_labels_out_dict
    pckl_output_dict['orig_predictions'] = orig_predictions
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
    
    # Make plots and save the wav files
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
            orig_cat_prediction = ds.label_mapping[np.argmax(orig_predictions.ravel())]
            synth_cat_prediction  = ds.label_mapping[np.argmax(synth_predictions.ravel())]
            print('Layer %s, Image %d, Orig Word Category Prediction: %s'%(
                  layer_to_invert, i, orig_cat_prediction))
            print('Layer %s, Image %d, Synth Word Category Prediction: %s'%(
                  layer_to_invert, i, synth_cat_prediction))
    
            # Set synthesis sucess based on the 16 category labels -- we can later evaluate the 1000 category labels.
            synth_success = orig_cat_prediction == synth_cat_prediction
    
            orig_label = np.argmax(orig_predictions)
            synth_label = np.argmax(synth_predictions)
    
            plt.subplot(4, BATCH_SIZE, i+1)
            plt.imshow(np.squeeze(np.array(all_outputs['input_after_preproc'][i].cpu())), origin='lower')
            plt.colorbar()
            plt.title('Layer %s, Word %d, Predicted Orig Label "%s" \n Orig Coch'%(layer_to_invert, i, 
                                                                                   ds.label_mapping[int(orig_label)]))
            plt.ylabel('Cochleagram Frequency Bins')
            plt.xlabel('Cochleagram Time Bins')
    
            plt.subplot(4, BATCH_SIZE, BATCH_SIZE+i+1)
            plt.imshow(np.squeeze(np.array(all_outputs_out['input_after_preproc'][i].cpu())), origin='lower')
            plt.colorbar()
            plt.title('Layer %s, Word %d, Predicted Synth Label "%s" \n Synth Coch'%(layer_to_invert, i, 
                                                                                     ds.label_mapping[int(synth_label)]))
            plt.ylabel('Cochleagram Frequency Bins')
            plt.xlabel('Cochleagram Time Bins')
    
            plt.subplot(4, BATCH_SIZE, BATCH_SIZE*2+i+1)
            plt.scatter(np.ravel(np.array(all_outputs[layer_to_invert].cpu())[i,:]), np.ravel(all_outputs_out[layer_to_invert].cpu().detach().numpy()[i,:]))
            plt.title('Layer %s, Word %d, Label "%s" \n Optimization'%(layer_to_invert, i, ds.label_mapping[int(targ[i].cpu().numpy())]))
            plt.xlabel('Orig Activations (%s)'%layer_to_invert)
            plt.ylabel('Synth Activations (%s)'%layer_to_invert)
    
            plt.subplot(4, BATCH_SIZE, BATCH_SIZE*3+i+1)
            if type(all_outputs['final'])==dict:
                dict_keys = list(all_outputs['final'].keys()) # So we ensure the same order
                plot_outputs_final = np.concatenate([np.array(all_outputs['final'][task_key].cpu()[i,:]).ravel() for task_key in dict_keys])
                plot_outputs_out_final = np.concatenate([np.array(all_outputs_out['final'][task_key].cpu().detach().numpy()[i,:]).ravel() for task_key in dict_keys]) 
                plt.scatter(plot_outputs_final.ravel(), plot_outputs_out_final.ravel()) 
            else:
                plt.scatter(np.ravel(np.array(all_outputs['final'].cpu())[i,:]), np.ravel(all_outputs_out['final'].cpu().detach().numpy()[i,:]))
            plt.title('Layer %s, Word %d, Label "%s" \n Optimization'%(layer_to_invert, i, ds.label_mapping[int(targ[i].cpu().numpy())]))
            plt.xlabel('Orig Activations (Final Layer)')
            plt.ylabel('Synth Activations (Final Layer)')
    
            fig.savefig(layer_filepath + '_cochleagrams_optimization.png')
    
            try:
                print('Layer %s, Word %d, Label "%s", Prediction Orig "%s", Prediction Synth "%s"'%(
                                                              layer_to_invert, i,
                                                              ds.label_mapping[int(targ[i].cpu().numpy())], 
                                                              ds.label_mapping[int(np.argmax(predictions[i].detach().cpu().numpy()))],
                                                              ds.label_mapping[int(np.argmax(predictions_out[i].detach().cpu().numpy()))]))
            except KeyError:
                print('Layer %s, Word %d, Label "%s", Prediction Orig "%s", Prediction Synth "%s"'%(
                                                              layer_to_invert, i,
                                                              ds.label_mapping[int(targ[i].cpu().numpy())],
                                                              ds.label_mapping[int(np.argmax(predictions['signal/word_int'][i].detach().cpu().numpy()))],
                                                              ds.label_mapping[int(np.argmax(predictions_out['signal/word_int'][i].detach().cpu().numpy()))]))
    
            if layer_idx==0:
                wavfile.write(base_filepath + '/orig.wav',
                               20000,
                               im[i].cpu().numpy().ravel())
            
            # Only save the individual wav if the layer optimization succeeded
            if synth_success or OVERRIDE_SAVE:
                wavfile.write(layer_filepath + '_synth.wav',
                              20000,
                              xadv[i].cpu().numpy().ravel())

def main(raw_args=None):
    #########PARSE THE ARGUMENTS FOR THE FUNCTION#########
    parser = argparse.ArgumentParser(description='Input the sound indices and the layers to match')
    parser.add_argument('SIDX',metavar='I',type=int,help='index into the sound list for the time_average sound')
    parser.add_argument('-L', '--LOSSFUNCTION', metavar='--L', type=str, default='inversion_loss_layer', help='loss function to use for the synthesis')
    parser.add_argument('-F', '--INPUTAUDIOFUNC', metavar='--A', type=str, default='psychophysics_wsj400_jsintest', help='function to use for grabbing the input audio sources for the "time_average" layers')
    parser.add_argument('-R', '--RANDOMSEED', metavar='--R', type=int, default=0, help='random seed to use for synthesis')
    parser.add_argument('-I', '--ITERATIONS', metavar='--I', type=int, default=3000, help='number of iterations in robustness synthesis kwargs')
    parser.add_argument('-N', '--NUMREPITER', metavar='--N', type=int, default=8, help='number of repetitions to run the robustness synthesis, each time decreasing the learning rate by half')
    parser.add_argument('-E', '--NOISE_SCALE', metavar='--E', type=float, default=0.0000001, help='multiply the noise by this value for the synthesis initialization')
    parser.add_argument('-S', '--STRICT', default=False, action='store_true', help='set to true to enforce strict loading of model')
    parser.add_argument('-V', '--OVERRIDE_SAVE', action='store_true', help='set to true to save, even if the optimization does not succeed')
    parser.add_argument('-O', '--OVERWRITE_PICKLE', action='store_true', help='set to true to overwrite the saved pckl file, if false then exits out if the file already exists')
    parser.add_argument('-Z', '--STEP_SIZE', metavar='--Z', type=float, default=1, help='Initial step size for the metamer generation')
    parser.add_argument('-D', '--DIRECTORY', metavar='--D', type=str, default=None, help='The directory with the location of the `build_network.py` file. Folder structure for saving metamers will be created in this directory. If not specified, assume this script is located in the same directory as the build_network.py file.')
    
    args=parser.parse_args(raw_args)
    SIDX = args.SIDX
    LOSS_FUNCTION = args.LOSSFUNCTION
    INPUTAUDIOFUNCNAME= args.INPUTAUDIOFUNC
    RANDOMSEED = args.RANDOMSEED
    STRICT = args.STRICT
    overwrite_pckl = args.OVERWRITE_PICKLE
    step_size = args.STEP_SIZE
    NOISE_SCALE = args.NOISE_SCALE
    ITERATIONS = args.ITERATIONS
    NUMREPITER = args.NUMREPITER
    OVERRIDE_SAVE = args.OVERRIDE_SAVE
    MODEL_DIRECTORY = args.DIRECTORY

    run_audio_metamer_generation(SIDX, LOSS_FUNCTION, INPUTAUDIOFUNCNAME, RANDOMSEED, STRICT, 
                                 overwrite_pckl, step_size, NOISE_SCALE, ITERATIONS, NUMREPITER, 
                                 OVERRIDE_SAVE, MODEL_DIRECTORY)
                                 
if __name__ == '__main__':
    main()

