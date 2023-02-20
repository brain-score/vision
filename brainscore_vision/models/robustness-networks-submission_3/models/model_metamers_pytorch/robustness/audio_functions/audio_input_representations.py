# Contains arguments for generating input audio representations
# These can be used either as part of the transformation or as 
# part of the preprocessing (which will be included in the graph 
# for gradient computation)
# Feather et al. 2022 used cochleagram representations, however parameters
# for a mel-spectrogram representation are also provided. 
import sys
sys.path.append('/om/user/jfeather/python-packages/chcochleagram')
import chcochleagram

log_mel_spec_0 = {'rep_type': 'mel_spec',
                  'rep_kwargs': {'sample_rate':20000,
                                 'n_mels':256,
                                 'win_length': 1200,
                                 'hop_length': 100,
                                 'n_fft': 1200,
                                 'f_min':50,
                                 'f_max':10000},
                 'compression_type': 'log',
                 'compression_kwargs': {'offset':1e-6},
                 }

mel_spec_0 = {'rep_type': 'mel_spec',
              'rep_kwargs': {'sample_rate':20000,
                             'n_mels':256,
                             'win_length': 1200,
                             'hop_length': 100,
                             'n_fft': 1200,
                             'f_min':50,
                             'f_max':10000},
             'compression_type': 'none',
             'compression_kwargs': {'offset':1e-6},
             }


# This is the cochleagram representation used in the Feather et al. 2022 
# paper "Model metamers illuminate divergences between biological and 
# artificial neural networks" Cochleagram is a fixed size. 
cochleagram_1 = {'rep_type': 'cochleagram',
                 'rep_kwargs': {'signal_size':40000,
                                'sr':20000,
                                'env_sr': 200,
                                'pad_factor':None,
                                'use_rfft':True,
                                'coch_filter_type': chcochleagram.cochlear_filters.ERBCosFilters,
                                'coch_filter_kwargs': {
                                    'n':50,
                                    'low_lim':50,
                                    'high_lim':10000,
                                    'sample_factor':4,
                                    'full_filter':False,
                                    },
                                'env_extraction_type': chcochleagram.envelope_extraction.HilbertEnvelopeExtraction,
                                'downsampling_type': chcochleagram.downsampling.SincWithKaiserWindow,
                                'downsampling_kwargs': {
                                    'window_size':1001},
                               },
                 'compression_type': 'coch_p3',
                 'compression_kwargs': {'scale': 1,
                                        'offset':1e-8,
                                        'clip_value': 5, # This wil clip cochleagram values < ~0.04
                                        'power': 0.3}
                }

# Same as cochleagram 1, but for 7 seconds of audio, as an example. 
cochleagram_1_7_secs = {'rep_type': 'cochleagram',
                 'rep_kwargs': {'signal_size':140000,
                                'sr':20000,
                                'env_sr': 200,
                                'pad_factor':None,
                                'use_rfft':True,
                                'coch_filter_type': chcochleagram.cochlear_filters.ERBCosFilters,
                                'coch_filter_kwargs': {
                                    'n':50,
                                    'low_lim':50,
                                    'high_lim':10000,
                                    'sample_factor':4,
                                    'full_filter':False,
                                    },
                                'env_extraction_type': chcochleagram.envelope_extraction.HilbertEnvelopeExtraction,
                                'downsampling_type': chcochleagram.downsampling.SincWithKaiserWindow,
                                'downsampling_kwargs': {
                                    'window_size':1001},
                               },
                 'compression_type': 'coch_p3',
                 'compression_kwargs': {'scale': 1,
                                        'offset':1e-8,
                                        'clip_value': 5, # This wil clip cochleagram values < ~0.04
                                        'power': 0.3}
                }


AUDIO_INPUT_REPRESENTATIONS = {'log_mel_spec_0': log_mel_spec_0,
                               'mel_spec_0': mel_spec_0,
                               'cochleagram_1': cochleagram_1,
                               'cochleagram_1_7_secs':cochleagram_1_7_secs,
                              }
