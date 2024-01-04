import tensorflow as tf
from tnn import main as tnn_main
from tnn.reciprocalgaternn import tnn_ReciprocalGateCell
from candidate_models.base_models.convrnn.median_rgcell import tnn_ReciprocalGateCell as legacy_tnn_ReciprocalGateCell
from collections import OrderedDict
import copy

dropout10L = {'conv'+str(l):1.0 for l in range(1,11)}
dropout10L['imnetds'] = 1.0

config_dict = {'model_params': {'cell_params': OrderedDict([('tau_depth_separable', True),
               ('tau_filter_size', 7.0),
               ('residual_to_out_gate', False),
               ('feedback_activation',
                tf.nn.elu),
               ('residual_to_cell_gate', False),
               ('out_residual', True),
               ('feedback_depth_separable', True),
               ('tau_nonlinearity',
                tf.nn.sigmoid),
               ('ff_depth_separable', False),
               ('gate_nonlinearity',
                tf.nn.tanh),
               ('cell_to_out', False),
               ('input_to_cell', True),
               ('cell_residual', False),
               ('in_out_depth_separable', False),
               ('layer_norm', False),
               ('input_activation',
                tf.nn.elu),
               ('ff_filter_size', 2.0),
               ('feedback_filter_size', 8.0),
               ('feedback_entry', 'input'),
               ('input_to_out', True),
               ('in_out_filter_size', 3.0),
               ('tau_offset', 0.9219348064291611),
               ('tau_multiplier', -0.9219348064291611),
               ('tau_bias', 4.147336708899556),
               ('out_activation',
                tf.nn.elu),
               ('weight_decay', 0.0002033999204146308),
               ('kernel_initializer', 'variance_scaling'),
               ('kernel_initializer_kwargs', {'scale': 0.6393378386273998}),
               ('cell_depth', 64),
               ('gate_depth_separable', True),
               ('gate_filter_size', 7.0),
               ('gate_offset', 0.7006566684988862),
               ('gate_multiplier', -0.7006566684988862),
               ('gate_bias', 2.776542926439013),
               ('cell_activation',
                tf.nn.elu)]),
  'image_off': 12,
  'times': 17}}

edges_2 = [(('conv8', 'conv5'), 0.0), (('conv9', 'conv6'), 0.0)]
edges_3 = edges_2 + [(('conv10', 'conv7'), 0.0)]
edges_5 = edges_3 + [(('conv7', 'conv6'), 0.0), (('conv10', 'conv9'), 0.0)]

def tnn_base_edges(inputs, train=True, basenet_layers=['conv'+str(l) for l in range(1,11)], alter_layers=None,
             unroll_tf=False, const_pres=False, out_layers='imnetds', base_name='model_jsons/10Lv9_imnet128_res23_rrgctx', 
             times=range(18), image_on=0, image_off=11, delay=10, random_off=None, dropout=dropout10L, 
             edges_arr=[], convrnn_type='recipcell', mem_val=0.0, train_tau_fg=False, apply_bn=False,
             channel_op='concat', seed=0, min_duration=11, use_legacy_cell=False, 
             layer_params={},
             p_edge=1.0,
             decoder_start=18,
             decoder_end=26,
             decoder_type='last',
             ff_weight_decay=0.0,
             ff_kernel_initializer_kwargs={},
             final_max_pool=True,
             tpu_name=None,
             gcp_project=None,
             tpu_zone=None,
             num_shards=None,
             iterations_per_loop=None, **kwargs):  

    mo_params = {}
    print("using multicell model!")
    # set ds dropout
    # dropout[out_layers] = ds_dropout
    
    # times may be a list or array, where t = 10t-10(t+1)ms.
    # if times is a list, it must be a subset of range(26).
    # input reaches convT at time t (no activations at t=0)
    image_off = int(image_off)
    decoder_start = int(decoder_start)
    decoder_end = int(decoder_end)
    if isinstance(times, (int, float)):
        ntimes = times
        times = range(ntimes)
    else:
        ntimes = times[-1]+1
    
    if random_off is not None and train == True:
        print("max duration", random_off - image_on)
        print("min duration", min_duration)
        image_times = np.random.choice(range(min_duration, random_off - image_on + 1))
        image_off = image_on + image_times
        print("image times", image_times)
        times = range(image_on + delay, image_off + delay)
        readout_time = times[-1]
        print("model times", times)
        print("readout_time", readout_time)
    else:
        image_times = image_off - image_on
    
    # set up image presentation, note that inputs is a tensor now, not a dictionary
    ims = tf.identity(inputs, name='split')
    batch_size = ims.get_shape().as_list()[0]
    print('IM SHAPE', ims.shape)
    
    if const_pres:
        print('Using constant image presentation')
        pres = ims
    else:
        print('Making movie')
        blank = tf.constant(value=0.5, shape=ims.get_shape().as_list(), name='split')
        pres = ([blank] * image_on) +  ([ims] * image_times) + ([blank] * (ntimes - image_off))
        
    # graph building stage
    with tf.compat.v1.variable_scope('tnn_model'):
        if '.json' not in base_name:
            base_name += '.json'
        print('Using base: ', base_name)
        G = tnn_main.graph_from_json(base_name)
        print("graph build from JSON")
        
        # memory_cell_params = cell_params.copy()
        # print("CELL PARAMS:", cell_params)
            
        # dealing with dropout between training and validation
        for node, attr in G.nodes(data=True):
            if apply_bn:
                if 'conv' in node:
                    print('Applying batch norm to ', node)
                    # set train flag of batch norm for conv layers
                    attr['kwargs']['pre_memory'][0][1]['batch_norm'] = True
                    attr['kwargs']['pre_memory'][0][1]['is_training'] = train

            this_layer_params = layer_params[node]
            # set ksize, out depth, and training flag for batch_norm
            for func, kwargs in attr['kwargs']['pre_memory'] + attr['kwargs']['post_memory']:

                if func.__name__ in ['component_conv', 'conv']:
                    ksize_val = this_layer_params.get('ksize')
                    if ksize_val is not None:
                        kwargs['ksize'] = ksize_val
                    print("using ksize {} for {}".format(kwargs['ksize'], node))
                    out_depth_val = this_layer_params.get('out_depth')
                    if out_depth_val is not None:
                        kwargs['out_depth'] = out_depth_val
                    print("using out depth {} for {}".format(kwargs['out_depth'], node)) 
                    if ff_weight_decay is not None:     # otherwise uses json               
                        kwargs['weight_decay'] = ff_weight_decay
                    if kwargs['kernel_init'] == "variance_scaling":
                        if ff_kernel_initializer_kwargs is not None: # otherwise uses json
                            kwargs['kernel_init_kwargs'] = ff_kernel_initializer_kwargs


            # # optional max pooling at end of conv10
            if node == 'conv10':
                if final_max_pool:
                    attr['kwargs']['post_memory'][-1] = (tf.nn.max_pool2d,
                                                        {'ksize': [1,2,2,1],
                                                         'strides': [1,2,2,1],
                                                         'padding': 'SAME'})
                    print("using a final max pool")
                else:
                    attr['kwargs']['post_memory'][-1] = (tf.identity, {})
                    print("not using a final max pool")

            # set memory params, including cell config
            memory_func, memory_param = attr['kwargs']['memory']
            
            if any(s in memory_param for s in ('gate_filter_size', 'tau_filter_size')):
                if convrnn_type == 'recipcell':
                    print('using reciprocal gated cell for ', node)
                    if use_legacy_cell:
                        print('Using legacy cell to preserve scoping to load checkpoint')
                        attr['cell'] = legacy_tnn_ReciprocalGateCell
                    else:
                        attr['cell'] = tnn_ReciprocalGateCell

                    recip_cell_params = this_layer_params['cell_params'].copy()
                    assert recip_cell_params is not None
                    for k,v in recip_cell_params.items():
                        attr['kwargs']['memory'][1][k] = v

            else:
                if alter_layers is None:
                    alter_layers = basenet_layers
                if node in alter_layers: 
                    attr['kwargs']['memory'][1]['memory_decay'] = mem_val
                    attr['kwargs']['memory'][1]['trainable'] = train_tau_fg
                if node in basenet_layers:
                    print(node, attr['kwargs']['memory'][1])

        # add non feedforward edges
        if len(edges_arr) > 0:
            edges = []
            for edge, p in edges_arr:
                if p <= p_edge:
                    edges.append(edge)
            print("applying edges,", edges)
            G.add_edges_from(edges)

        # initialize graph structure
        tnn_main.init_nodes(G, input_nodes=['conv1'], batch_size=batch_size, channel_op=channel_op)
        
        # unroll graph
        if unroll_tf:
            print('Unroll tf way')
            tnn_main.unroll_tf(G, input_seq={'conv1': pres}, ntimes=ntimes)
        else:
            print('Unrolling tnn way')
            tnn_main.unroll(G, input_seq={'conv1': pres}, ntimes=ntimes)

        # collect last timestep output
        logits_list = [G.node['imnetds']['outputs'][t] for t in range(decoder_start, decoder_end)]

        print("decoder_type", decoder_type, "from", decoder_start, "to", decoder_end)
        if decoder_type == 'last':
            logits = logits_list[-1]
        elif decoder_type == 'sum':
            logits = tf.add_n(logits_list)
        elif decoder_type == 'avg':
            logits = tf.add_n(logits_list) / len(logits_list)
        elif decoder_type == 'random':
            if train:
                logits = np.random.choice(logits_list)
            elif not train: #eval -- use last timepoint with image on
                t_eval = image_off + delay - 1
                t_eval = t_eval - decoder_start
                logits = logits_list[t_eval]
        else:
            raise ValueError

        logits = tf.squeeze(logits)
        print("logits shape", logits.shape)

    outputs = {}
    outputs['logits'] = logits  
    outputs['times'] = {} 
    for t in times:
        outputs['times'][t] = tf.squeeze(G.node[out_layers]['outputs'][t])     
    return outputs, mo_params

def load_median_model(inputs, train=False, tnn_json=None, edges_arr=edges_5, neural_presentation=False,
               cell_layers = ['conv' + str(i) for i in range(4, 11)], use_legacy_cell=True, tau_adjust=False,
               decoder_type='last'):

    train_config = copy.deepcopy(config_dict)
    model_params = train_config['model_params']

    cell_params = model_params.pop('cell_params')
    # add fb specific params
    cell_params['feedback_activation'] = tf.identity
    cell_params['feedback_entry'] = 'out'
    cell_params['feedback_depth_separable'] = False
    cell_params['feedback_filter_size'] = 1

    layer_params = {'conv1':{'cell_params': None}, 
                    'conv2':{'cell_params':None}, 
                    'conv3':{'cell_params':None}, 
                    'conv4':{'cell_params':None}, 
                    'conv5':{'cell_params':None}, 
                    'conv6':{'cell_params':None}, 
                    'conv7':{'cell_params':None}, 
                    'conv8':{'cell_params':None}, 
                    'conv9':{'cell_params':None}, 
                    'conv10':{'cell_params':None}, 
                    'imnetds':{'cell_params':None}}

    for k in cell_layers:
        layer_params[k]['cell_params'] = cell_params.copy()

    if neural_presentation: # set up image presentation of model when mapping to neural data
        model_params['times'] = range(26) # unroll for entire temporal trajectory
        model_params['image_on'] = 0
        model_params['image_off'] = 10
        
    model_params['layer_params'] = layer_params

    model_params['ff_weight_decay'] = None # use the values specified in json
    model_params['ff_kernel_initializer_kwargs'] = None # use the values specified in json
    model_params['final_max_pool'] = True
    if decoder_type == 'last':
        print('Using default last timestep decoder')
        model_params['decoder_end'] = model_params['times']
        model_params['decoder_start'] = model_params['decoder_end'] - 1
        model_params['decoder_type'] = 'last'

    model_params['edges_arr'] = edges_arr
    model_params['base_name'] = tnn_json
    model_params['use_legacy_cell'] = use_legacy_cell

    if tau_adjust: # when presenting 224 sized images to 128 image size models for neural/behavioral fits
        print('Fixing filter sizes to pass 224 sized images to a model trained with 128 sized ImageNet')
        for layer in ['conv8', 'conv9', 'conv10']:
            for ksize in ['ff_filter_size', 'tau_filter_size', 'gate_filter_size']:
                if model_params['layer_params'][layer]['cell_params'][ksize] > 4:
                    model_params['layer_params'][layer]['cell_params'][ksize] = 4

    return tnn_base_edges(inputs, train=train, **model_params)
