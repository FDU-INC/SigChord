%% channels

tgaxChannel = wlanTGaxChannel;
tgaxChannel.NumTransmitAntennas = 1;
tgaxChannel.NumReceiveAntennas = 1;
tgaxChannel.SampleRate = 2e9;

nrchannel = nrTDLChannel;
nrchannel.NumReceiveAntennas = 1;
nrchannel.SampleRate = 2e9;

% ricianchan = comm.RicianChannel;
%
% rayleighchan = comm.RayleighChannel;
ricianchan = comm.RicianChannel( ...
    SampleRate=2e9, ...
    PathDelays=[0 20 40] / 3e9, ...
    AveragePathGains=[0 -5 -10], ...
    KFactor=10, ...
    MaximumDopplerShift=200e3);

% rayleighchan = comm.RayleighChannel;
rayleighchan = comm.RayleighChannel( ...
    SampleRate=2e9, ...
    PathDelays=[0 20 40] / 3e9, ...
    AveragePathGains=[0 -10 -20], ...
    MaximumDopplerShift=200e3);

tgnChannel = wlanTGnChannel('SampleRate',2e9,'LargeScaleFadingEffect', ...
    'Pathloss and shadowing','DelayProfile','Model-F', 'CarrierFrequency', 0.1);

% channels = {tgaxChannel, nrchannel, rayleighchan, tgnChannel};
% channels = {tgaxChannel, ricianchan, rayleighchan};
channels = {ricianchan, rayleighchan};

% for i = 1:20000
%     signal = randn(96000, 1) + 1i * randn(96000, 1);
%     new_sig = funcChannel(signal, channels);
% end
%
%
% function newWave = funcChannel(waveform, channels)
%     id = randi([1, 1]);
%     if id == 4
%         newWave = waveform;
%     else
%         chan = channels(id);
%         chan = chan{1};
%         newWave = chan(waveform);
%     end
% end
