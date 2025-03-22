function newWave = funcChannel(waveform, channels)
    id = randi([1, 3]);
    if id == 3
        newWave = waveform;
    else
        chan = channels(id);
        chan = chan{1};
        newWave = chan(waveform);
    end
end