function test_detectAnomalies_sustained()
    frame = zeros(120, 160, 3, 'uint8');
    grayFrame = zeros(120, 160, 'uint8');
    landmarks = [];
    faceDetector = [];
    yoloDetector = [];

    [flags, ~, ~, ~] = detectAnomalies( ...
        frame, grayFrame, landmarks, faceDetector, yoloDetector, ...
        30, 0, 0, 'LOOKING RIGHT', 0, 1, false);

    types = {flags.Type};
    assert(~any(strcmp(types, 'Sustained Pose Turning')), 'Transient pose should not trigger a pose anomaly flag.');
    assert(~any(strcmp(types, 'Off-Screen Eye Shift')), 'Transient gaze should not trigger an eye-shift flag.');

    fprintf('Transient anomaly test passed.\n');
end
