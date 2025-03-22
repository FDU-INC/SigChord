function [mod, rate] = mcs_query(mcs)
    % only for table 2
    dict = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    dict(0) = {'QPSK', 120/1024};
    dict(1) = {'QPSK', 193/1024};
    dict(2) = {'QPSK', 308/1024};
    dict(3) = {'QPSK', 449/1024};
    dict(4) = {'QPSK', 602/1024};

    dict(5) = {'16QAM', 378/1024};
    dict(6) = {'16QAM', 434/1024};
    dict(7) = {'16QAM', 490/1024};
    dict(8) = {'16QAM', 553/1024};
    dict(9) = {'16QAM', 616/1024};
    dict(10) = {'16QAM', 658/1024};

    dict(11) = {'64QAM', 466/1024};
    dict(12) = {'64QAM', 517/1024};
    dict(13) = {'64QAM', 567/1024};
    dict(14) = {'64QAM', 616/1024};
    dict(15) = {'64QAM', 666/1024};
    dict(16) = {'64QAM', 719/1024};
    dict(17) = {'64QAM', 772/1024};
    dict(18) = {'64QAM', 822/1024};
    dict(19) = {'64QAM', 873/1024};

    dict(20) = {'256QAM', 682.5/1024};
    dict(21) = {'256QAM', 711/1024};
    dict(22) = {'256QAM', 754/1024};
    dict(23) = {'256QAM', 797/1024};
    dict(24) = {'256QAM', 841/1024};
    dict(25) = {'256QAM', 885/1024};
    dict(26) = {'256QAM', 916.5/1024};
    dict(27) = {'256QAM', 948/1024};
    if isKey(dict, mcs)
        result = dict(mcs);
        mod = result{1};
        rate = result{2};
    else
        error("MCS key %d not found in dictionary", mcs)
    end
end