function [siginfo] = oneDvbs2(batch)
    persistent txPkts
    if isempty(txPkts)
        numFrames = 1;
        syncBits = [0 1 0 0 0 1 1 1]';    % Sync byte for TS packet is 47 Hex
        pktLen = 1496;                    % UP length without sync bits is 1496
        numPkts = 58112*numFrames;
        txRawPkts = randi([0 1],pktLen,numPkts);
        txPkts = [repmat(syncBits,1,numPkts); txRawPkts];
    end

    nDummySymb = 5;
    samplesPerSymbol = 100;
    nHeadSymb = 90;
    nSlotSymb = 1440;
    nPilotSymb = 36;
    sigLen = samplesPerSymbol * (nHeadSymb + nSlotSymb + nPilotSymb);

    siginfo.waveforms = zeros(batch, sigLen);
    siginfo.modcods = zeros(batch, 1);
    siginfo.fecFrames = zeros(batch, 1);
    siginfo.hasPilots = zeros(batch, 1);
    siginfo.lengths = zeros(batch, 1);

    for i = 1:batch
        validCfg = false;
        modcod = randi([1, 28]);
        rolloffCandis = [0.35, 0.25, 0.2];
        rolloff = rolloffCandis(randi(3));
        fecFrame = randi([0, 1]);
        hasPilots = randi([0, 1]);
        while ~validCfg
            try
                % when config is not correct, change fecFrame only
                if fecFrame == 1
                    fecFrameStr = "short";
                else
                    fecFrameStr = "normal";
                end

                s2WaveGen = dvbs2WaveformGenerator(...
                    "FECFrame", fecFrameStr,...
                    "MODCOD", modcod,...
                    "RolloffFactor", rolloff,...
                    "DFL", 2048, ...
                    "HasPilots", hasPilots, ...
                    "SamplesPerSymbol", samplesPerSymbol);
                nPkts = s2WaveGen.MinNumPackets*1;
                data = txPkts(:, 1:nPkts);
                data = data(:);
                waveform = s2WaveGen(data);
                validCfg = true;
            catch exception
                % try another length
                if fecFrame == 1
                    fecFrame = 0;
                else
                    fecFrame = 1;
                end
                validCfg = false;
            end
        end
        waveform = waveform(nDummySymb * samplesPerSymbol + 1 : end);
        len = min(sigLen, length(waveform));
        waveform = waveform(1:len);
        waveform = waveform / sqrt(mean(abs(waveform) .^ 2));
        siginfo.waveforms(i, 1:len) = waveform.';
        siginfo.modcods(i) = modcod;
        siginfo.fecFrames(i) = fecFrame;
        siginfo.hasPilots(i) = hasPilots;
        siginfo.lengths(i) = len;
    end
end