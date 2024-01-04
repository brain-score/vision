from ipywidgets import widgets, interactive
import matplotlib.pyplot as plt
import urllib.request, json 
from model_analysis_folders import all_model_info
from ipywidgets import Output, GridspecLayout

from IPython import display
from IPython.display import Image
from IPython.core.display import HTML

visual_experiment_dict_by_name = {
    'Visual Experiment 1 (Standard Models, Figure 2)': 1,
    'Visual Experiment 2 (Self-Supervised Models, Figure 3)': 5,
    'Visual Experiment 3 (HMAX, Figure 4)': 6,
    'Visual Experiment 4 (ResNet50 Adversarially Robust, Figure 5)': 3,
    'Visual Experiment 5 (AlexNet Adversarially Robust, Figure 5)': 4,
    'Visual Experiment 6 (Lowpass AlexNet and VOneNet, Figure 7)': 9,
}

audio_experiment_dict_by_name = {
    'Auditory Experiment 1 (Standard Models, Figure 2)': 1,
    'Auditory Experiment 2 (Spectemp, Figure 4)': 6,
    'Auditory Experiment 3 (CochResNet50 Waveform Adverarial Training, Figure 6)': 3,
    'Auditory Experiment 4 (CochCNN9 Waveform Adverarial Training, Figure 6)': 4,
    'Auditory Experiment 5 (CochResNet50 Cochleagram Adverarial Training, Figure 6)': 7,
    'Auditory Experiment 6 (CochCNN9 Cochleagram Adverarial Training, Figure 6)': 8,
}

def display_all_visual_models_for_experiment(exp_name, 
                               example_idx, 
                               ):
    
    exp_num = visual_experiment_dict_by_name[exp_name]
    experiment_list = [1,5,6,3,4,9]

    experiment_name = all_model_info.TURK_IMAGE_EXPERIMENTS_GROUPINGS['experiment_%d'%exp_num]['paper_experiment_name']
    jsin_configs = all_model_info.TURK_IMAGE_EXPERIMENTS_GROUPINGS['experiment_%d'%exp_num]['experiment_params_web']

    experiment_folder_web= all_model_info.TURK_IMAGE_EXPERIMENTS_GROUPINGS['experiment_%d'%exp_num]['experiment_folder_web']

    with urllib.request.urlopen(jsin_configs) as url:
        experiment_params = json.loads(url.read().decode())

    condition_names = experiment_params['experiment_conditions']
    plot_model_order = experiment_params['experiment_info']['networks']

    max_num_plots = 10
    plt.figure(figsize=(4*max_num_plots,4*(1+len(plot_model_order))))
    orig_html_path = experiment_folder_web + '/%d_%s/%d_%s/%d_%s_%s.png'%(0, 'orig', 
                                                                         0, 'orig',
                                                                         example_idx, 'orig', 'orig')
    ax = plt.subplot(1+len(plot_model_order),max_num_plots,1)
    
    try:
        orig_png = urllib.request.urlopen(orig_html_path)
        if 'hmax_standard' in plot_model_order:
            plt.imshow(plt.imread(metamer_png_path), interpolation='none', cmap='Greys_r')
        else:
            plt.imshow(plt.imread(orig_png))
        plt.title('Orig', fontsize=16)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
    except urllib.request.HTTPError: 
        print('Missing %s, this image was not included in this experiment. Try another example.'%orig_html_path)
    
    for model_idx, model in enumerate(plot_model_order):
        model_folder_web = experiment_folder_web + '/%d_%s'%(model_idx+1, model)

        all_png_images = []
        images = []

        plot_layers = all_model_info.ALL_NETWORKS_AND_LAYERS_IMAGES[model]['layers']
        model_name = all_model_info.ALL_NETWORKS_AND_LAYERS_IMAGES[model]['paper_name']
        plot_layers = [p for p in plot_layers if 'input_after_preproc' not in p]

        for plot_idx, layer in enumerate(plot_layers):
            try:
                metamer_html_path = model_folder_web + '/%d_%s/%d_%s_%s.png'%(plot_idx, layer, 
                                                                                example_idx, model, layer)
                metamer_png_path = urllib.request.urlopen(metamer_html_path)
                # Uncomment if you want to view the full paths to the metamers. 
                # Note: some of the metamers may look slightly full size without the downsampling for matplotlib
                # viewing, so it can be helpful to look at or download the pngs directly. 
                # print(metamer_html_path)
                
                ax = plt.subplot(1+len(plot_model_order),max_num_plots,1+(max_num_plots * (model_idx+1))+plot_idx)
                if 'hmax' in model:
                    plt.imshow(plt.imread(metamer_png_path), interpolation='none', cmap='Greys_r')
                else:
                    plt.imshow(plt.imread(metamer_png_path), interpolation='none')
                plt.title(layer.split('_fake_relu')[0].split('_fakerelu')[0], fontsize=16)

                if plot_idx==0:
                    plt.ylabel(model_name, fontsize=16)

                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_xticks([])          

            except urllib.request.HTTPError:
                print('Missing %s, this image was not included in this experiment. Try another example.'%orig_html_path)
    plt.show()


def display_all_audio_models_for_experiment(exp_name, 
                               example_idx, 
                               ):
    exp_num = audio_experiment_dict_by_name[exp_name]

    experiment_name = all_model_info.TURK_AUDIO_EXPERIMENTS_GROUPINGS['experiment_%d'%exp_num]['paper_experiment_name']
    jsin_configs = all_model_info.TURK_AUDIO_EXPERIMENTS_GROUPINGS['experiment_%d'%exp_num]['experiment_params_web']

    experiment_folder_web = all_model_info.TURK_AUDIO_EXPERIMENTS_GROUPINGS['experiment_%d'%exp_num]['experiment_folder_web']

    with urllib.request.urlopen(jsin_configs) as url:
        experiment_params = json.loads(url.read().decode())

    condition_names = experiment_params['experiment_conditions']
    plot_model_order = experiment_params['experiment_info']['networks']

    max_num_plots = 10
    orig_html_path = experiment_folder_web + '/%d_%s/%d_%s/%d_%s_%s.wav'%(0, 'orig', 
                                                                         0, 'orig',
                                                                         example_idx, 'orig', 'orig')
    
    grid = GridspecLayout(1+len(plot_model_order), 1+max_num_plots)

    all_paths = []
    
    try:
        urllib.request.urlopen(orig_html_path).getcode()
        load_audio = True
    except urllib.request.HTTPError: 
        load_audio = False
        print('Missing %s, this audio was not included in this experiment. Try another example.'%orig_html_path)
        
    if load_audio:
        out = Output()
        with out:
            print('Orig')
            display.display(display.Audio(url = orig_html_path))
        grid[0, 1] = out
    
    for model_idx, model in enumerate(plot_model_order):
        model_folder_web = experiment_folder_web + '/%d_%s'%(model_idx+1, model)

        all_png_images = []
        images = []

        plot_layers = all_model_info.ALL_NETWORKS_AND_LAYERS_AUDIO[model]['layers']
        model_name = all_model_info.ALL_NETWORKS_AND_LAYERS_AUDIO[model]['paper_name']

        for plot_idx, layer in enumerate(plot_layers):
            metamer_html_path = model_folder_web + '/%d_%s/%d_%s_%s.wav'%(plot_idx, layer, 
                                                                            example_idx, model, layer)
            # Uncomment if you want to view the full paths to the metamers. 
            # print(metamer_html_path)
            try:
                urllib.request.urlopen(metamer_html_path).getcode()
                load_audio = True
            except urllib.request.HTTPError: 
                load_audio = False
                print('Missing %s, this audio was not included in this experiment. Try another example.'%metamer_html_path)
                
            if load_audio:
                out = Output()
                with out:
                    if layer == 'input_after_preproc':
                        print('cochleagram')
                    else:
                        print(layer.split('_fake_relu')[0].split('_fakerelu')[0])
                    display.display(display.Audio(url = metamer_html_path))
                grid[model_idx+1, plot_idx+1] = out
                all_paths.append(metamer_html_path)
                
            if plot_idx == 0:
                out = Output()
                with out:
                    plt.figure()
                    plt.text(0, 0, model_name, rotation=90, fontsize=40)
                    plt.axis('off')
                    plt.show()
# This doesn't seem to work as expected. Use matplotlib as a hack.
#                     display.display_latex(model_name, raw=True)

                grid[model_idx+1, 0] = out
    display.display(grid)
