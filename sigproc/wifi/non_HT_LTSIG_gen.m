nSig = 1000;
sigLen = 512;

mcss = randi([0, 7], 1, nSig);
psduLens = randi([1, 4095], 1, nSig);
res = zeros(sigLen, nSig);
for i = 1:nSig
    psdulength = psduLens(i);
    cfg = wlanNonHTConfig("MCS", mcss(i), "PSDULength", psdulength);
    % psdulength = cfg.PSDULength;
    psdu = randi([0 1], 8 * psdulength, 1);
    waveform = wlanWaveformGenerator(psdu, cfg, "OversamplingFactor", 1);
    len = min(sigLen, length(waveform));
    res(1:len, i) = waveform(1:len);
end

save('./wifi-non-HT.mat', "res");