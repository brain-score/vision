import os
import socket
import yaml

DEFAULTS = {
    'suffix': '',
    'file_pattern': '*.hdf5',
}

class DataSpaceBuilder():
    def __init__(self, data_settings, readout_protocol):
        self.data_settings = data_settings
        self.readout_protocol = readout_protocol
        self.pretraining_train_scenarios = self.get_scenarios('train', 'pretraining')
        self.pretraining_test_scenarios = self.get_scenarios('test', 'pretraining')
        self.readout_train_scenarios = self.get_scenarios('train', 'readout')
        self.readout_test_scenarios = self.get_scenarios('test', 'readout')

    def get_setting(self, setting, mode, phase):
        try:
            val = self.data_settings[phase][mode][setting]
        except KeyError:
            try:
                val = self.data_settings[phase][setting]
            except KeyError:
                try:
                    val = self.data_settings[setting]
                except KeyError:
                    if setting in DEFAULTS:
                        val = DEFAULTS[setting]
                    else:
                        print(self.data_settings)
                        print(f'{setting} not found for phase {phase} and mode {mode}')
                        raise
        return val

    def get_scenarios(self, mode, phase):
        scenarios = self.get_setting('scenarios', mode, phase)
        suffix = self.get_setting('suffix', mode, phase)
        assert isinstance(scenarios, list), f'Scenarios for {phase} {mode} has type {type(scenarios)}, expected list'
        return [scenario + suffix for scenario in scenarios]

    def get_data_dir(self, mode, phase):
        data_dir = self.get_setting('dir', mode, phase)
        if not os.path.isabs(data_dir):
            data_dir = self.add_basedir(data_dir)
        return data_dir

    def get_file_pattern(self, mode, phase):
        return self.get_setting('file_pattern', mode, phase)

    @staticmethod
    def add_basedir(reldir):
        dirname =  os.path.dirname(__file__)
        hostname = socket.gethostname()
        basedir_file = os.path.join(dirname, 'basedir.yaml')
        if os.path.isfile(basedir_file):
            basedir_dict = yaml.safe_load(open(basedir_file, 'rb'))
            assert hostname in basedir_dict, f'{hostname} not found in {basedir_file}'
            basedir = basedir_dict.get(hostname)
        return os.path.join(basedir, reldir)

    def build_paths(self, name, scenarios, phase):
        res = {
            'name': name, 
            } 
        for mode in ['train', 'test']:
            curr_scenarios = scenarios[mode]
            if not isinstance(curr_scenarios, list):
                assert isinstance(curr_scenarios, str)
                curr_scenarios = [curr_scenarios]
            data_dir = self.get_data_dir(mode, phase)
            file_pattern = self.get_file_pattern(mode, phase)
            res[mode] = [os.path.join(data_dir, scenario, file_pattern) for scenario in curr_scenarios]
        return res

    def get_readout_paths(self, pretraining_train_scenario):
        if self.readout_protocol == 'full':
            readout_paths = [self.build_paths(readout_train_scenario, {'train': readout_train_scenario, 'test': readout_test_scenario}, 'readout') for readout_train_scenario, readout_test_scenario in zip(self.readout_train_scenarios, self.readout_test_scenarios)]
        else: # "minimal" protocal matches readout train scenario to pretraining train scenario
            pretrain_train_wo_suffix = pretraining_train_scenario.replace(self.get_setting('suffix', 'train', 'pretraining'), '', 1)
            assert pretrain_train_wo_suffix in self.readout_train_scenarios, '{} not in {}, but using "{}" readout protocol'.format(pretrain_train_wo_suffix, self.readout_train_scenarios, self.readout_protocol)
            readout_train_scenario = pretrain_train_wo_suffix
            readout_test_scenario = self.readout_test_scenarios[self.readout_train_scenarios.index(readout_train_scenario)] # get readout test scenario corresponding to train
            readout_paths = [self.build_paths(readout_train_scenario, {'train': readout_train_scenario, 'test': readout_test_scenario}, 'readout')]
        return readout_paths

    def get_only_space(self):
        data_spaces = []
        for pretraining_train_scenario, pretraining_test_scenario in zip(self.pretraining_train_scenarios, self.pretraining_test_scenarios):
            space = {
                'pretraining': self.build_paths(pretraining_train_scenario, {'train': pretraining_train_scenario, 'test': pretraining_test_scenario}, 'pretraining'),
                'readout': self.get_readout_paths(pretraining_train_scenario),
                }
            data_spaces.append(space)
        return data_spaces

    def get_abo_space(self):
        data_spaces = []
        assert len(self.pretraining_train_scenarios) > 1, 'Must have more than one scenario to do all-but-one pretraining protocol.' # just check train since train and test should be same length
        for pretraining_train_scenario, pretraining_test_scenario in zip(self.pretraining_train_scenarios, self.pretraining_test_scenarios):
            # build abo scenarios
            abo_pretraining_scenarios = list(zip(self.pretraining_train_scenarios, self.pretraining_test_scenarios))
            abo_pretraining_scenarios.remove((pretraining_train_scenario, pretraining_test_scenario))
            abo_pretraining_train_scenarios, abo_pretraining_test_scenarios = [list(t) for t in zip(*abo_pretraining_scenarios)]

            space = {
                'pretraining': self.build_paths('no_'+pretraining_train_scenario, {'train': abo_pretraining_train_scenarios, 'test': abo_pretraining_test_scenarios}, 'pretraining'),
                'readout': self.get_readout_paths(pretraining_train_scenario),
                }
            data_spaces.append(space)
        return data_spaces

    def get_all_space(self):
        assert len(self.pretraining_train_scenarios) > 1, f'Must have more than one scenario to do all pretraining protocol.' # just check train since train and test should be same length
        pretraining_train_suffix = self.get_setting('suffix', 'train', 'pretraining')
        space = {
            'pretraining': self.build_paths('all'+pretraining_train_suffix, {'train': self.pretraining_train_scenarios, 'test': self.pretraining_test_scenarios}, 'pretraining'),
            'readout': [self.build_paths(readout_train_scenario, {'train': readout_train_scenario, 'test': readout_test_scenario}, 'readout') for readout_train_scenario, readout_test_scenario in zip(self.readout_train_scenarios, self.readout_test_scenarios)] # TODO: implement "minimal" readout protocol for "all" which matches the training scenarios?
            }
        return [space]

def get_data_spaces(
    pretraining_protocols=('all', 'abo', 'only'),
    readout_protocol='minimal', # {'full'|'minimal'}: 'minimal' only does readout on matching scenario to pretraining
    **data_settings
    ):
    builder = DataSpaceBuilder(data_settings, readout_protocol)

    data_spaces = [] # only pretraining and readout spaces, without seed
    if 'only' in pretraining_protocols:
        data_spaces.extend(builder.get_only_space())
    if 'abo' in pretraining_protocols:
        data_spaces.extend(builder.get_abo_space())
    if 'all' in pretraining_protocols:
        data_spaces.extend(builder.get_all_space())
    # print(*data_spaces, sep='\n')
    return data_spaces
