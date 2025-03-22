function [siginfo] = oneHT(batch)
    %% HT
    oversamplingFactor = 100;
    sigLen = 4 * round(20e-6 * 20e6 * oversamplingFactor);

    siginfo.waveforms = zeros(batch, sigLen);
    siginfo.bits = zeros(batch, 48);
    siginfo.lengths = zeros(batch, 1);

    for i = 1:batch
        validCfg = false;
        psdulength = randi([0, 2^16-1]);
        cbw = randi([0, 1]);
        % one spatial stream
        mcs = randi([0, 7]);
        while ~validCfg
            try
                if cbw == 0
                    cbwStr = "CBW20";
                else
                    cbwStr = "CBW40";
                end
                cfg = wlanHTConfig("ChannelBandwidth", cbwStr, "MCS", mcs, "PSDULength", psdulength);
                [~, lsigBits] = wlanLSIG(cfg);
                [~, htBits] = wlanHTSIG(cfg);
                resBits = [lsigBits; htBits(1:24)];
                psdu = randi([0 1], 8 * psdulength, 1);
                waveform = wlanWaveformGenerator(psdu, cfg, "OversamplingFactor", oversamplingFactor / (cbw + 1));
                validCfg = true;
            catch exception
                % try another length
                validCfg = false;
                psdulength = fix(psdulength / 2);
            end
        end
        
        len = min(sigLen, length(waveform));
        waveform = waveform(1:len);
        waveform = waveform / sqrt(mean(abs(waveform) .^ 2));
        siginfo.waveforms(i, 1:len) = waveform.';
        siginfo.bits(i, :) = resBits.';
        siginfo.lengths(i) = len;
    end
end
