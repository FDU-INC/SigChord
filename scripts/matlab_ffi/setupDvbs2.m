%% DVB-S2 config
addpath('../s2xLDPCParityMatrices');
if ~exist('../dvbs2xLDPCParityMatrices.mat','file')
    if ~exist('../s2xLDPCParityMatrices.zip','file')
        url = 'https://ssd.mathworks.com/supportfiles/spc/satcom/DVB/s2xLDPCParityMatrices.zip';
        websave('../s2xLDPCParityMatrices.zip',url);
        unzip('../s2xLDPCParityMatrices.zip');
    end
addpath('../s2xLDPCParityMatrices');
end