function test_poseSignalTools()
    history = [0 0 0; 2 4 6];
    [smoothed, thresholds] = poseSignalTools([10 20 30], history, 3);

    assert(abs(smoothed(1) - 4) < 1e-9, 'Yaw smoothing should average the recent samples.');
    assert(abs(smoothed(2) - 8) < 1e-9, 'Pitch smoothing should average the recent samples.');
    assert(abs(smoothed(3) - 12) < 1e-9, 'Roll smoothing should average the recent samples.');
    assert(all(thresholds >= 8), 'Adaptive thresholds should be at least the minimum threshold.');

    fprintf('Pose signal helper test passed.\n');
end
