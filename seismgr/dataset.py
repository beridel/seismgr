from .config import config as cfg
import copy as cp
import numpy as np
import datetime as dt
import obspy as obs
import ast
import os
import collections as coll

def to_samples(value, sampling_rate=None):
    if sampling_rate is None:
        sampling_rate = cfg.sampling_rate
    value = np.asarray(np.around(value * sampling_rate), dtype=np.int32)
    return value


def to_seconds(value, sampling_rate=None):
    if sampling_rate is None:
        sampling_rate = cfg.sampling_rate
    value = np.asarray(np.float64(value) / sampling_rate, dtype=np.float64)
    return np.around(value, 2)


def smooth(signal, smooth_win):
    win = np.repeat(1, smooth_win)
    return np.convolve(signal, win, 'same')


def rms(signal):
    return np.sqrt(np.sum(signal ** 2) / signal.size)


def mad(signal):
    return np.median(np.abs(signal - np.median(signal)))


def bandstr(band_list):
   return '{0[0]:.1f}_{0[1]:.1f}'.format(band_list)


def bandlist(band_str):
    return [float(band_str[:3]), float(band_str[-3:])]


def datetime2matlab(dtime):
    mdn = dtime + dt.timedelta(days = 366)
    frac_seconds = (dtime - dt.datetime(dtime.year, dtime.month, dtime.day,
                                        0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dtime.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds


def matlab2datetime(dtime):
    return dt.datetime.fromordinal(int(dtime)) + dt.timedelta(days=float(dtime) % 1) - dt.timedelta(days=366)


class Network():
    """Station data:
    Contains stations and geographical coordinates.

    network_file = station ascii file name.
    """
    def __init__(self, network_file):
        self.where = os.path.join(cfg.input, network_file)

    def n_stations(self):
        return np.int32(len(self.stations))

    def n_components(self):
        return np.int32(len(self.components))

    def read(self):
        stations = []
        components = []
        with open(self.where, 'r') as file:
            # read in start and end dates
            columns = file.readline().strip().split()
            self.start_date = dt.date(
                int(columns[0][0:0 + 4]),
                int(columns[0][4:4 + 2]),
                int(columns[0][6:6 + 2]))
            self.end_date = dt.date(
                int(columns[1][0:0 + 4]),
                int(columns[1][4:4 + 2]),
                int(columns[1][6:6 + 2]))
            self.stations = stations

            # read in component names
            columns = file.readline().strip().split()
            for component in columns[1:]:
                components.append(component)
            self.components = components

            # read in station names and coordinates
            latitude, longitude, depth, bands = [], [], [], []
            for line in file:
                columns = line.strip().split()
                stations.append(columns[0])
                latitude.append(np.float32(columns[1]))
                longitude.append(np.float32(columns[2]))
                depth.append(np.float32(columns[3]) / 1000.)  # convert m to km
                bands.append([np.float32(columns[4]), np.float32(columns[5])])

            self.latitude = np.asarray(latitude, dtype=np.float32)
            self.longitude = np.asarray(longitude, dtype=np.float32)
            self.depth = np.asarray(depth, dtype=np.float32)
            self.bands = bands

    def datelist(self):
        dates = []
        date = self.start_date
        while date <= self.end_date:
            dates.append(date)
            date += dt.timedelta(days=1)

        return dates

    def subset(self, stations, components=None):
        subnetwork = cp.deepcopy(self)

        if components is None:
            components = self.components

        if not isinstance(stations, list):
            stations = [stations]
        if not isinstance(components, list):
            components = [components]

        stations_to_remove = [station for station in self.stations if station not in stations]
        for station in stations_to_remove:
            idx = subnetwork.stations.index(station)
            subnetwork.stations.remove(station)
            np.delete(subnetwork.latitude, idx)
            np.delete(subnetwork.longitude, idx)
            np.delete(subnetwork.depth, idx)
            subnetwork.bands.pop(idx)

        components_to_remove = [component for component in self.components if component not in components]
        for component in components_to_remove:
            idx = subnetwork.components.index(component)
            subnetwork.components.remove(component)

        return subnetwork

    def combine(self, network):
        self.components.extend(network.components)
        self.components = list(coll.OrderedDict.fromkeys(self.components))

        self.stations.extend(network.stations)

        self.latitude = np.concatenate((self.latitude, network.latitude))
        self.longitude = np.concatenate((self.longitude, network.longitude))
        self.depth = np.concatenate((self.depth, network.depth))

        if self.start_date > network.start_date:
            self.start_date = network.start_date

        if self.end_date < network.end_date:
            self.end_date = network.end_date


class Region():
    """Region descriptor that contains velocity model and grid of
    theoretical source points.
    grid_file = regular theoretical source point grid file name
    model_file = velocity model file name
    """

    def __init__(self, region_file):
        self.where = os.path.join(cfg.input, region_file)

    def read(self):
        top, vp, ratio_ps = [], [], []
        n_layers = 0
        with open(self.where) as file:
            # read in latitude min, max, and regular interval
            columns = file.readline().strip().split()
            latitude = np.linspace(
                np.float32(columns[0]),
                np.float32(columns[1]),
                np.float32(columns[2]))

            # read in longitude min, max, and regular interval
            columns = file.readline().strip().split()
            longitude = np.linspace(
                np.float32(columns[0]),
                np.float32(columns[1]),
                np.float32(columns[2]))

            # read in depth min, max, and regular interval
            columns = file.readline().strip().split()
            depth = np.linspace(
                np.float32(columns[0]),
                np.float32(columns[1]),
                np.float32(columns[2]))

            # read in velocity model layer by layer
            line = file.readline()
            while line:
                n_layers += 1
                columns = line.strip().split()
                top.append(np.float32(columns[0]))
                vp.append(np.float32(columns[1]))
                ratio_ps.append(np.float32(columns[2]))
                line = file.readline()

        self.latitude = np.asarray(latitude, dtype=np.float32)
        self.longitude = np.asarray(longitude, dtype=np.float32)
        self.depth = np.asarray(depth, dtype=np.float32)
        self.top = np.asarray(top, dtype=np.float32)
        self.vp = np.asarray(vp, dtype=np.float32)
        self.ratio_ps = np.asarray(ratio_ps, dtype=np.float32)
        self.n_layers = np.int32(n_layers)


class Data():
    """Basic data class. Contains the following
        attributes:
    stations - list of station names
    components - list of component names
    band - filtered frequency band (empty for unfiltered)
    dwnsample - T/F whether it's downsampled
    trace_idx - internal organisation of traces based on stations/components
        methods:
    set_band - change the band attribute (filtering must be done manually)
    n_stations - return the number of stations
    n_components - return the number of components
    """

    def __init__(self, stations, components, band):
        if not isinstance(stations, list):
            stations = [stations]
        if not isinstance(components, list):
            components = [components]
        self.stations = stations
        self.components = components

        if not band:
            self.band = ''
        elif isinstance(band, str):
            self.band = band
        elif isinstance(band, list):
            self.set_band(band)

        self.sampling_rate = cfg.sampling_rate
        trace_idx = 0
        self.trace_idx = {}
        for station in self.stations:
            self.trace_idx[station] = {}
            for component in self.components:
                self.trace_idx[station][component] = trace_idx
                trace_idx += 1

        self.downsampled = False

    def set_band(self, band):
        """Takes a list and sets the frequency band attribute
        """

        self.band = '{0[0]:.1f}_{0[1]:.1f}'.format(band)

    def n_stations(self):
        return np.int32(len(self.stations))

    def n_components(self):
        return np.int32(len(self.components))

    def chk_stations(self, stations):
        """Compares self.stations to stations (list) for order and equalivancy
        """

        if not stations:
            print('No stations to compare with!')
            return

        return all([True for i, j in zip(self.stations, stations) if i == j])

    def chk_components(self, components):
        """Compares self.components to components (list) for order and
        equalivancy
        """

        if not components:
            print('No components to compare with!')
            return

        return all([True for i, j in zip(self.components,
                                         components) if i == j])


class RealData(Data):
    """Data attributed to a specific real time. Contains the same as the Data
    class, plus
        attributes:
    date - the date as a UTCDateTime (from obspy)
        methods:
    set_times - set the start and stop times (in seconds) of the data
    duration - return the duration (in seconds)
    get_trace - returns a trace from a station/component
    get_matrix - returns the traces that correspond to stations/components
    set_trace - puts a trace into a station/component
    """
    def __init__(self, date, start_time, end_time, stations, components, band):
        Data.__init__(self, stations, components, band)
        self.date = obs.UTCDateTime(date)
        self.set_times(start_time, end_time)
        self.loaded = False
        self.operational = {station: False for station in self.stations}

    def set_times(self, start_time, end_time):
        """Set start and end time attributes. Times can be:
        - strings in the form of YYYYMMDD
        - seconds
        - obspy.UTCDateTime objects
        """

        if isinstance(start_time, str):
            start_time = obs.UTCDateTime(self.date + start_time)
            end_time = obs.UTCDateTime(self.date + end_time)
        elif not isinstance(start_time, obs.UTCDateTime):
            dtimes = [dt.timedelta(seconds=start_time),
                      dt.timedelta(seconds=end_time)]

            start_time = obs.UTCDateTime(self.date + dtimes[0])
            end_time = obs.UTCDateTime(self.date + dtimes[1])

        if (end_time - start_time) > 0:
            self.start_time = start_time
            self.end_time = end_time
        else:
            print('Error: end_time before start_time; times not set')

    def duration(self):
        return self.end_time - self.start_time + 1. / self.sampling_rate

    def seconds(self, time):
        delta = time.datetime - dt.datetime.combine(time.date, dt.time(0))
        return delta.total_seconds()

    def samples(self, time):
        seconds = self.seconds(time)
        return to_samples(seconds, self.sampling_rate)

    def get_trace(self, station, component):
        """Returns an ndarray for the given station and component
        """

        if not self.loaded:
            print('Error: Cannot get trace; self.traces not yet read in')
        else:
            idx = self.trace_idx[station][component]
            return np.float32(self.traces[idx].data)

    def get_matrix(self, stations=None, components=None, three_d=False):
        """Returns an ndarray for the given stations and components. If three_d
        is False, returns a 2D ndarray [stations*components, time]; if true,
        returns a 3D ndarray [stations, components, time]
        """

        if not stations:
            stations = self.stations
        if not components:
            components = self.components

        if isinstance(stations, str):
            stations = [stations]
        if isinstance(components, str):
            components = [components]

        if not self.loaded:
            print('Error: Cannot get trace; self.traces not yet read in')
            return np.zeros(1)

        n_stations = len(stations)
        n_components = len(components)
        n_samples = self.traces[0].data.size

        if three_d:
            dimensions = (n_stations, n_components, n_samples)
        elif not three_d:
            dimensions = (n_stations * n_components, n_samples)

        matrix = np.zeros(dimensions, dtype=np.float32)
        for s in range(n_stations):
            for c in range(n_components):
                trace = self.get_trace(stations[s], components[c])

                if three_d:
                    matrix[s, c, :] = trace
                elif not three_d:
                    matrix[s * n_components + c, :] = trace

        return matrix

    def set_trace(self, trace, station, component, override=False):
        """Takes an ndarray and sets the obspy.trace.data
        """

        if not isinstance(trace, np.ndarray):
            print('Error: Not an ndarray')
            return

        idx = self.trace_idx[station][component]

        if override is False and trace.size != self.traces[idx].data.size:
            print('Error: Cannot set trace; not the same number of samples')
            return

        try:
            self.traces[idx].data = np.float32(trace)
        except:
            print('Error: Cannot set trace; self.traces not yet read in')

    def downsample(self):
        """Downsample all traces by the downsampling factor set in
        parameters.cfg
        """

        if not self.loaded:
            print('Error: Cannot downsample without loaded data!')
            return

        self.traces.decimate(int(cfg.dwnsample))
        self.sampling_rate /= int(cfg.dwnsample)
        self.downsampled = True

    def n_operational(self):
        return np.int32(sum(self.operational.values()))

    def operational_list(self, stations=None):
        if stations is None:
            stations = self.stations
        return [self.operational[station] for station in stations]


class DayData(RealData):
    """Class for a daily network seismogram. Contains the same as RealData,
    plus
        methods:
    read - read in the daily records for the stations/components
    write - writes the data in the DayData object to the daily records
    where - fetches the path to the daily records based on stations/components
    """

    def __init__(self, date, stations, components, band=None, downsample=None):
        RealData.__init__(self, date, 0, 86400, stations, components, band)

        if downsample:
            self.downsampled = True
            self.sampling_rate = cfg.sampling_rate / cfg.dwnsample

        self.end_time -= 1. / self.sampling_rate

    def read(self, raw=False):
        """Reads in data.
        """

        path = self.where(raw)
        end_time = self.date + (86400. - 1. / self.sampling_rate)

        self.traces = obs.Stream()
        for station in self.stations:
            folder = cfg.chk_trailing(os.path.join(path, station))
            read_in = obs.Stream()
            n_operational = 0

            for component in self.components:
                filename = '{}.{}.{}.sac'.format(
                    self.date.strftime('%Y%m%d'),
                    station,
                    component)
                try:
                    read_in += obs.read(os.path.join(folder, filename), endtime=end_time)
                    n_operational += 1
                except:
                    trace = obs.Trace(data=np.zeros(86400 * cfg.sampling_rate,
                                                    dtype=np.float32))
                    trace.stats.sampling_rate = cfg.sampling_rate
                    trace.stats.delta = 1. / cfg.sampling_rate
                    trace.stats.starttime = self.start_time
                    trace.stats.station = station
                    trace.stats.channel = component
                    read_in += trace

            self.traces.extend(read_in)
            if n_operational > 0:
                self.operational[station] = True

        self.loaded = True

    def write(self, raw=False):
        """Writes out data.
        """

        path = self.where(raw=raw)

        n_written = 0
        for station in self.stations:
            if not self.operational[station]:
                continue

            folder = os.path.join(path, station)
            cfg.chk_folder(folder)
            for component in self.components:
                filename = '{}.{}.{}.sac'.format(
                    self.date.strftime('%Y%m%d'),
                    station,
                    component)
                self.traces[n_written].write(os.path.join(folder, filename), format='SAC')
                n_written += 1

    def where(self, raw=False):
        """Returns path to data files.
        """

        if raw:
            processing = ''
            self.sampling_rate = cfg.sampling_rate
            self.downsampled = False
            self.band = None

            path = cfg.data
        else:
            if not self.band:
                processing = 'preproc'
                if self.downsampled:
                    processing += '_dwn'

                path = os.path.join(cfg.data, processing)
            else:
                processing = 'filtered'
                if self.downsampled:
                    processing += '_dwn'

                path = os.path.join(cfg.data, processing, self.band)

        return path

    def read_buffer(self):
        # read in 100s before day
        path = self.where()

        for station in self.stations:
            folder = cfg.chk_trailing(path + station)
            date = self.date - dt.timedelta(days=1)
            start_time = self.date - cfg.data_buffer
            end_time = self.date - 1. / self.sampling_rate

            for component in self.components:
                idx = self.trace_idx[station][component]
                filename = '{}{}.{}.{}.sac'.format(
                    folder,
                    date.strftime('%Y%m%d'),
                    station,
                    component)
                try:
                    self.traces[idx] += obs.read(filename,
                                                 starttime=start_time,
                                                 endtime=end_time)
                except:
                    trace = obs.Trace(
                        data=np.zeros(int(cfg.data_buffer) * cfg.sampling_rate,
                                      dtype=np.float32))
                    trace.stats.sampling_rate = cfg.sampling_rate
                    trace.stats.delta = 1. / cfg.sampling_rate
                    trace.stats.starttime = start_time
                    trace.stats.network = self.traces[idx].stats.network
                    trace.stats.station = station
                    trace.stats.channel = component
                    self.traces[idx] += trace

            date = self.date + dt.timedelta(days=1)
            start_time = date
            end_time = date + cfg.data_buffer - 1. / cfg.sampling_rate

            for component in self.components:
                idx = self.trace_idx[station][component]
                filename = '{}{}.{}.{}.sac'.format(
                    folder,
                    date.strftime('%Y%m%d'),
                    station,
                    component)
                try:
                    self.traces[idx] += obs.read(filename,
                                                 starttime=start_time,
                                                 endtime=end_time)
                except:
                    trace = obs.Trace(
                        data=np.zeros(int(cfg.data_buffer) * cfg.sampling_rate,
                                      dtype=np.float32))
                    trace.stats.sampling_rate = cfg.sampling_rate
                    trace.stats.delta = 1. / cfg.sampling_rate
                    trace.stats.starttime = start_time
                    trace.stats.network = self.traces[idx].stats.network
                    trace.stats.station = station
                    trace.stats.channel = component
                    self.traces[idx] += trace


class Event(RealData):
    """Contains an event that has a moveout. Contains the same as DayData, plus
        methods:
    REWRITES read - reads the events based on start/stop times and moveout
    """

    def __init__(self, start_time, end_time, day_data,
                 moveout=None, coordinates=None):
        RealData.__init__(self,
                          day_data.date,
                          start_time,
                          end_time,
                          day_data.stations,
                          day_data.components,
                          day_data.band)
        self.moveout = moveout
        self.parent = day_data
        self.downsampled = day_data.downsampled
        self.sampling_rate = day_data.sampling_rate
        self.operational = day_data.operational

        if moveout is None:
            self.moveout = np.zeros(self.n_stations(), np.int32)

        if coordinates is not None:
            self.latitude = coordinates[0]
            self.longitude = coordinates[1]
            self.depth = coordinates[2]

        self.loaded = False
        if not day_data.loaded:
            print('DayData not loaded! Read data and reinstance Event!')

    def read(self, buf=False):
        """Reads in data from day_data. Forces all traces to have the correct
        number of samples (self.duration() * self.sampling_rate).
        """

        self.traces = obs.Stream()
        n_samples = to_samples(self.duration(), self.sampling_rate)

        if buf:
            n_samples += to_samples(2 * cfg.filter_buffer,
                                        self.sampling_rate)

        for s in range(self.n_stations()):
            moveout = round(self.moveout[s], 2)

            for component in self.components:
                idx = self.parent.trace_idx[self.stations[s]][component]
                trace = self.parent.traces[idx]

                start = self.start_time + moveout
                end = self.end_time + moveout
                if buf:
                    start -= cfg.filter_buffer
                    end += cfg.filter_buffer

                chk_trace = trace.slice(starttime=start, endtime=end)
                while chk_trace.data.size != n_samples:
                    difference = n_samples - chk_trace.data.size

                    if difference > 0:
                        end = end + 0.25 / self.sampling_rate
                    elif difference < 0:
                        end = end - 0.25 / self.sampling_rate

                    chk_trace = trace.slice(start,
                                            end)
                    if chk_trace.data.size != n_samples:
                        debug_here()

                self.traces.append(chk_trace)

        self.loaded = True


class Template(RealData):
    def __init__(self, metadata, waveforms):
        RealData.__init__(
            self,
            obs.UTCDateTime(metadata['date']),
            obs.UTCDateTime(metadata['datetime']),
            obs.UTCDateTime(metadata['datetime']) + metadata['duration'],
            ast.literal_eval(metadata['stations']),
            ast.literal_eval(metadata['components']),
            cfg.freq_bands[0])
        self.s_moveout = metadata['moveout']
        self.template_id = metadata['template_id']
        self.operational = ast.literal_eval(metadata['operational'])
        self.latitude = metadata['latitude']
        self.longitude = metadata['longitude']
        self.depth = metadata['depth']
        self.peak = metadata['peak']
        self.peak_ratio = metadata['peak_ratio']

        waveforms = waveforms.reshape((self.n_stations(),
                                       self.n_components(),
                                       -1))
        n_samples_buf = to_samples(cfg.filter_buffer, cfg.sampling_rate)
        self.traces = obs.Stream()
        for s in range(self.n_stations()):
            for c in range(self.n_components()):
                trace = obs.Trace(
                    data=waveforms[s, c, n_samples_buf:-n_samples_buf])
                trace.stats.station = self.stations[s]
                trace.stats.channel = self.components[c]
                trace.stats.sampling_rate = self.sampling_rate
                trace.stats.starttime = self.start_time + self.s_moveout[s]
                self.traces += trace

        self.buffer = waveforms
        self.loaded = True

    def buffer2traces(self):
        """Sets the traces to the buffer so that downsampling/filtering can be
        performed without modifying the template signal in the middle.
        """

        for s in range(self.n_stations()):
            for c in range(self.n_components()):
                self.set_trace(self.buffer[s, c, :],
                               self.stations[s],
                               self.components[c],
                               override=True)

    def trim_traces(self):
        """Trims the buffer from the traces.
        """

        n_samples_buf = to_samples(cfg.filter_buffer, cfg.sampling_rate)
        for station in self.stations:
            for component in self.components:
                trace = self.get_trace(station, component)
                self.set_trace(trace[n_samples_buf:-n_samples_buf],
                               station,
                               component,
                               override=True)

    def bandpass(self, band):
        """Bandpass filters the traces after tapering the buffer.
        """

        self.buffer2traces()
        self.traces.taper(
            cfg.filter_buffer / (self.duration() + 2 * cfg.filter_buffer),
            type='cosine')
        self.traces.filter('bandpass', freqmin=band[0], freqmax=band[1])
        self.set_band(band)
        self.trim_traces()


class Multiplet():

    def __init__(self, metadata):
        RealData.__init__(
            self,
            obs.UTCDateTime(metadata['date']),
            obs.UTCDateTime(metadata['datetime']),
            obs.UTCDateTime(metadata['datetime']) + metadata['duration'],
            ast.literal_eval(metadata['stations']),
            ast.literal_eval(metadata['components']),
            '')
        #self.date = 
        self.template_id = metadata['template_id']
        self.multiplet_id = metadata['multiplet_id']
        self.corr = metadata['corr']
        self.operational = ast.literal_eval(metadata['operational'])
        self.ampl = metadata['ampl']


class Stack(Data):

    def __init__(self, multiplets, stations, components):
        Data.init(self, stations, components, None)
        if isinstance(template_id, list):
            self.n_families = len(template_id)
        else:
            self.n_families = 1

        self.template_id = template_id
        self.traces = np.zeros((n_families,
                                self.n_stations(),
                                self.n_components(),
                                to_samples(cfg.stack_len)),
                               dtype=np.float32)

    def load_metadata(self, template_id=None):
        if not template_id:
            pass

