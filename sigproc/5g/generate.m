for mcs = 0:27
    for i = 0:9
        [mod, rate] = mcs_query(mcs);
        PDSCH_conf = nrWavegenPDSCHConfig("Modulation", mod, "TargetCodeRate", rate, "DataSource", {'PN9-ITU', i});
        wave_conf = nrDLCarrierConfig("PDSCH", {PDSCH_conf});
        [waveform, waveformInfo] = nrWaveformGenerator(wave_conf);
        save(sprintf('./data/5g_mcs%d_id%d.mat', mcs, i), "waveform");
    end
end

% samplerate = waveformInfo.ResourceGrids(1).Info.SampleRate;
% nfft = waveformInfo.ResourceGrids(1).Info.Nfft;
% figure;
% spectrogram(waveform(:,1),ones(nfft,1),0,nfft,'centered',samplerate,'yaxis','MinThreshold',-130);
% title('Spectrogram of 5G Downlink Baseband Waveform');