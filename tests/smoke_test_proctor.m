function smoke_test_proctor()
    addpath('modules');
    v = VideoReader('data/Phone.mp4');
    frame = readFrame(v);

    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
    params = importONNXFunction('models/face_landmarks.onnx', 'landmarkFcn');
    landmarkNet = @(img, varargin) landmarkFcn(img, params, varargin{:});

    landmarks = getFacialLandmarks(frame, faceDetector, landmarkNet);
    assert(~isempty(landmarks), 'Landmarks should be detected on the sample frame.');

    camIntrinsics = cameraIntrinsics([v.Width, v.Width], [v.Width/2, v.Height/2], [v.Height, v.Width]);
    [yaw, pitch, roll, ~] = estimateHeadPose(landmarks, camIntrinsics);
    assert(any([yaw, pitch, roll] ~= 0), 'Head pose should not remain at zero.');

    grayFrame = rgb2gray(frame);
    [gazeDir, gazeTarget] = estimateGaze(grayFrame, landmarks, size(frame, 2), size(frame, 1));
    assert(~strcmp(gazeDir, 'UNKNOWN'), 'Gaze direction should be resolved.');
    assert(all(isfinite(gazeTarget)), 'Gaze target should be finite.');

    fprintf('Smoke test passed: pose=%g %g %g gaze=%s target=%s\n', yaw, pitch, roll, gazeDir, mat2str(gazeTarget));
end
