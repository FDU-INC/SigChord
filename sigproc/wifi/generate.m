for mod = 0:7
    for i = 0:9
        cfg = wlanHTConfig("MCS", mcs);
        psdulength = cfg.PSDULength;
        psdu = randi([0 1], 8 * psdulength, 1);
        waveform = wlanWaveformGenerator(psdu, cfg);
        wlanSampleRate(cfg, 'OversamplingFactor', 1)
        save(sprintf('./data/wifi_mcs%d_id%d.mat', mod, i), "waveform");
    end
end