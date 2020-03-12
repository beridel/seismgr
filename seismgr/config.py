import os

class Config():
    def __init__(self, path, parameters):
        self.__dict__ = parameters
        self.base = self.chk_trailing(os.path.abspath(path))
        self.input = os.path.join(self.base, 'input')
        self.data = os.path.join(self.base, 'data')

    def chk_folder(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def chk_trailing(self, to_test):
        if to_test.endswith('/'):
            return to_test
        else:
            return to_test + '/'


def read_config(path):
    """read in study parameters
    """
    parameters_path = os.path.join(path, 'parameters.cfg')
    with open(parameters_path) as file:
        param_dict = {}
        for line in file:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            try:
                value = float(value)
            except Exception:
                pass
            if isinstance(value, str) and len(value.split(',')) > 1:
                value = [float(freq) for freq in value.split(',')]
            param_dict[key] = value

    # specific parameter config
    # frequency bands
    if 'min_freq' and 'max_freq' in param_dict:
        param_dict['freq_bands'] = []
        if isinstance(param_dict['min_freq'], list):
            n_freqs = len(param_dict['min_freq'])
            min_freq = param_dict.pop('min_freq')
            max_freq = param_dict.pop('max_freq')
            for f in range(n_freqs):
                param_dict['freq_bands'].append([min_freq[f], max_freq[f]])
        else:
            param_dict['freq_bands'].append(
                [param_dict['min_freq'], param_dict['max_freq']])

    # sampling_rate (needs to be an int)
    if 'sampling_rate' in param_dict:
        param_dict['sampling_rate'] = int(param_dict['sampling_rate'])

    # downsampling (needs to be an int)
    if 'dwnsample' in param_dict:
        param_dict['dwnsample'] = int(param_dict['dwnsample'])

    return Config(path, param_dict)

path = os.getcwd()
config = read_config(path)
