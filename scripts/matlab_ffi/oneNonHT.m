function [siginfo] = oneNonHT(batch)
    oversamplingFactor = 100;
    sigLen = 4 * round(20e-6 * 2e9);
    siginfo.waveforms = zeros(batch, sigLen);
    siginfo.bits = zeros(batch, 24);
    siginfo.lengths = zeros(batch, 1);

    %% non-HT
    for i = 1:batch
        psdulength = randi([1, 4095]);
        mcs = randi([0, 7]);
        cfg = wlanNonHTConfig("MCS", mcs, "PSDULength", psdulength);

        [~, lsigBits] = wlanLSIG(cfg);
        psdu = randi([0 1], 8 * psdulength, 1);
        waveform = wlanWaveformGenerator(psdu, cfg, "OversamplingFactor", oversamplingFactor);

        len = min(sigLen, length(waveform));
        waveform = waveform(1:len);
        waveform = waveform / sqrt(mean(abs(waveform) .^ 2));
        siginfo.waveforms(i, 1:len) = waveform.';
        siginfo.bits(i, :) = lsigBits.';
        siginfo.lengths(i) = len;
    end
end