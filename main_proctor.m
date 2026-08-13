%% MAIN_PROCTOR.M - Computer Vision Exam Proctoring System
clear; clc; close all;
addpath('modules');

% --- INITIALIZE PYTHON ENGINE ---

% Uncomment if needed
% pathToPython = 'D:\Anaconda\envs\proctor_env\python.exe'; 
% pyenv('Version', pathToPython, 'ExecutionMode', 'OutOfProcess');

if count(py.sys.path, pwd) == 0
    insert(py.sys.path, int32(0), pwd); % Add current directory to Python search path
end

% Load the PyTorch Gaze Engine
gazeEngine = py.gaze_engine.GazeTracker();

% 1. System Setup & Model Initialization
videoPath = 'C:\Users\PC\Downloads\frame_skip.mp4'; % just change
vReader   = VideoReader(videoPath);
vWriter   = VideoWriter('proctor_output_explained_cheaterJames(frame_skip_2).mp4', 'MPEG-4');
open(vWriter);

% Frame metadata
frameWidth  = vReader.Width;
frameHeight = vReader.Height;

% Performance tuning for faster processing
procScale = 0.5;
detectEvery = 4;
useLowResProcessing = true;

% Load Deep Learning / Computer Vision Models
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');

% Import ONNX model as a callable function (bypasses invalid graph issues)
landmarkParams = importONNXFunction('models/face_landmarks.onnx', 'landmarkFcn');

% Define a wrapper to call the ONNX function safely
landmarkNet = @(img, varargin) landmarkFcn(img, landmarkParams, varargin{:});

% landmarkNet = dlnetwork(landmarkNet);   % Convert to dlnetwork explicitly
yoloDetector = yolov4ObjectDetector('csp-darknet53-coco'); % Requires COCO pretrained package

% Camera Intrinsics Approximation (Assuming standard webcam FOV ~60 deg)
focalLength = [frameWidth, frameWidth]; 
principalPoint = [frameWidth/2, frameHeight/2];
camIntrinsics = cameraIntrinsics(focalLength, principalPoint, [frameHeight, frameWidth]);

% State Trackers & Analytics Containers
anomalyLog = struct('Time', {}, 'Type', {}, 'Severity', {});
gazeHeatmap = zeros(frameHeight, frameWidth);
prevIntensity = 0;
frameIdx = 0;
startTime = tic;
lastReportTime = 0;
reportIntervalSec = 2;
prevLandmarks = [];
prevPoseAxes = [];
prevGazeDir = 'CENTER / FORWARD';
prevGazeTarget = [frameWidth/2, frameHeight/2];
prevObjectBboxes = [];
prevObjectLabels = {};
prevFrameFlags = struct('Time', {}, 'Type', {}, 'Severity', {});
skipState = struct('isFrozen', false, 'freezeStartTime', NaN); 
prevGrayRawForSkip = [];
initialYaw = 0;
initialPitch = 0;
initialRoll = 0;
initialGazeDir = 'CENTER / FORWARD';
screenLeft = 300; 
screenRight = 900; 
screenTop = 350; 
screenBottom = 650;
initialGazeTarget = [frameWidth/2, frameHeight/2];
twoFaceStreak = 0;
twoFaceActive = false;
faceAbsenceStartTime = NaN;
faceAbsenceThresholdSec = 3;
faceAbsenceAlarmActive = false;
faceIdentityMismatchActive = false;
faceIdentityMismatchStreak = 0;
faceIdentityMismatchStreakThreshold = 2;
gazeDeviationStartTime = NaN;
gazeDeviationThresholdSec = 3;
gazeDeviationAlarmActive = false;
headPoseDeviationStartTime = NaN;
headPoseDeviationThresholdSec = 3;
headPoseDeviationAlarmActive = false;

fprintf('Processing video stream...\n');

referenceFrame = readFrame(vReader);
referenceGray = rgb2gray(referenceFrame);

referenceData = getReferenceData( ...
    referenceFrame, ...
    referenceGray, ...
    faceDetector);

%% 2. Processing Loop
while hasFrame(vReader)
    frame = readFrame(vReader);
    frameIdx = frameIdx + 1;
    currentTime = frameIdx / vReader.FrameRate;

    if useLowResProcessing
        procFrame = imresize(frame, procScale);
        grayRaw = rgb2gray(procFrame);
    else
        procFrame = frame;
        grayRaw = rgb2gray(frame);
    end

    % Use the raw grayscale signal for brightness measurements, and the enhanced
    % version for downstream face/landmark processing.
    grayEnhanced = im2double(grayRaw);
    grayEnhanced = histeq(grayEnhanced);
    grayEnhanced = imadjust(grayEnhanced, [0.05 0.95], [0 1]);
    grayFrame = im2uint8(grayEnhanced);

    elapsedSec = toc(startTime);
    if elapsedSec - lastReportTime >= reportIntervalSec || frameIdx == 1
        totalFrames = vReader.NumFrames;
        if isempty(totalFrames)
            totalFrames = '?';
        end
        fprintf('Frame %d/%d | Time %.1fs | Elapsed %.1fs\n', frameIdx, totalFrames, currentTime, elapsedSec);
        lastReportTime = elapsedSec;
    end

    runFullAnalysis = (frameIdx == 1) || (mod(frameIdx, detectEvery) == 1);

    if runFullAnalysis
        % --- Module 1: Landmark Detection ---
        landmarksProc = getFacialLandmarks(procFrame, faceDetector, landmarkNet);

        % Extra check: treat the two-face flag as a conservative, sustained cue only.
        faceBboxes = step(faceDetector, procFrame);
        [reliableFaceCount, ~] = countReliableFaceBoxes(faceBboxes, size(procFrame));

        if reliableFaceCount >= 2
            twoFaceStreak = twoFaceStreak + 1;
        else
            twoFaceStreak = max(0, twoFaceStreak - 1);
            if twoFaceStreak == 0
                twoFaceActive = false;
            end
        end

        if reliableFaceCount >= 2 && twoFaceStreak >= 3 && ~twoFaceActive
            twoFaceVisible = true;
            twoFaceActive = true;
        else
            twoFaceVisible = false;
        end
        
        phoneLightDetected = detectPhoneLight( ...
            procFrame, ...
            grayRaw, ...
            grayFrame, ...
            faceDetector, ...
            referenceData);


        % Default outputs if face is absent/occluded
        yaw = initialYaw; pitch = initialPitch; roll = initialRoll;
        gazeDir = initialGazeDir;
        gazeTarget = initialGazeTarget;
        poseAxes = [];
        landmarksFull = [];

        if ~isempty(landmarksProc)
            landmarksFull = landmarksProc / procScale;

            % --- Module 2: Head Pose Estimation (PnP) ---
            [yaw, pitch, roll, poseAxes] = estimateHeadPose(landmarksFull, camIntrinsics);
            if frameIdx == 1
                initialYaw = yaw; initialPitch = pitch; initialRoll = roll;
            end

            % --- Module 3: Gaze & Iris Tracking ---
            % 1. Get the bounding box of the face to crop it
            faceBoxes = step(faceDetector, procFrame);
            
            if ~isempty(faceBoxes)
                % 1. Crop the face
                [~, idx] = max(faceBoxes(:, 3) .* faceBoxes(:, 4));
                faceBox = round(faceBoxes(idx, :));
                
                faceBox(1) = max(1, faceBox(1));
                faceBox(2) = max(1, faceBox(2));
                faceBox(3) = min(frameWidth - faceBox(1), faceBox(3));
                faceBox(4) = min(frameHeight - faceBox(2), faceBox(4));
                
                faceCrop = imcrop(procFrame, faceBox);
                
                % 2. Save the crop as a temporary file for Python to read perfectly
                tempImgPath = fullfile(pwd, 'temp_face.jpg');
                imwrite(faceCrop, tempImgPath);
                
                % 3. Get absolute Radians from Python
                gazeAngles = gazeEngine.estimate_from_crop(tempImgPath);
                
                % Assign Pitch to 1 and Yaw to 2 to un-swap the axes
                eyePitch = double(gazeAngles{1});
                eyeYaw   = double(gazeAngles{2});
                
                % 4. Project the true absolute eye-gaze 
                eyeMidpoint = landmarksFull(28, :); 
                projectionDistance = 500; 
                
                % yawSensitivity boosts the horizontal movement (increase it if still too slight)
                yawSensitivity = 1.8; 
                
                % Added a negative sign to dx to fix the mirror inversion
                dx = -sin(eyeYaw) * projectionDistance * yawSensitivity; 
                dy = -sin(eyePitch) * projectionDistance; 
                
                gazeTarget = [eyeMidpoint(1) + dx, eyeMidpoint(2) + dy];
                gazeDir = 'L2CS-NET TRACKING';
            else
                % Fallback if face detector drops the frame
                gazeTarget = prevGazeTarget;
                gazeDir = prevGazeDir;
            end
            
            if frameIdx == 1
                initialGazeDir = gazeDir;
                initialGazeTarget = gazeTarget;
            end
        end

        % --- Module 4: Anomaly & Contraband Detection ---
        [frameFlags, objectBboxes, objectLabels, currentIntensity] = detectAnomalies(...
            procFrame, grayFrame, landmarksProc, faceDetector, yoloDetector, yaw, pitch, roll, gazeDir, prevIntensity, currentTime, runFullAnalysis, ...
            initialYaw, initialPitch, initialRoll);

        if twoFaceVisible
            frameFlags(end+1) = struct('Time', currentTime, 'Type', 'Multiple Faces Visible', 'Severity', 'Medium');
        end

        if phoneLightDetected
            frameFlags(end+1) = struct('Time', currentTime, 'Type', 'Possible Phone Light', 'Severity', 'Medium');
        end

        [faceIdentityMatch, faceIdentityScore, faceIdentityPresent] = verifyFaceAgainstReference(...
            procFrame, grayFrame, faceDetector, referenceData);

        if reliableFaceCount > 0 && faceIdentityPresent
            if ~faceIdentityMatch
                faceIdentityMismatchStreak = faceIdentityMismatchStreak + 1;
            else
                faceIdentityMismatchStreak = max(0, faceIdentityMismatchStreak - 1);
            end

            if faceIdentityMismatchStreak >= faceIdentityMismatchStreakThreshold && ~faceIdentityMismatchActive
                frameFlags(end+1) = struct('Time', currentTime, 'Type', 'Face Identity Mismatch', 'Severity', 'High');
                faceIdentityMismatchActive = true;
            elseif faceIdentityMatch && faceIdentityMismatchStreak < faceIdentityMismatchStreakThreshold
                faceIdentityMismatchActive = false;
            end
        else
            faceIdentityMismatchStreak = 0;
            faceIdentityMismatchActive = false;
        end

        if reliableFaceCount == 0
            if isnan(faceAbsenceStartTime)
                faceAbsenceStartTime = currentTime;
            elseif currentTime - faceAbsenceStartTime >= faceAbsenceThresholdSec && ~faceAbsenceAlarmActive
                frameFlags(end+1) = struct('Time', currentTime, 'Type', 'Face Absent for Extended Period', 'Severity', 'High');
                faceAbsenceAlarmActive = true;
            end
        else
            faceAbsenceStartTime = NaN;
            faceAbsenceAlarmActive = false;
        end

        if ~isempty(objectBboxes)
            objectBboxes = objectBboxes;
        end

        if ~isempty(poseAxes)
            poseAxes = poseAxes;
        end

        prevLandmarks = landmarksFull;
        prevPoseAxes = poseAxes;
        prevGazeDir = gazeDir;
        prevGazeTarget = gazeTarget;
        prevObjectBboxes = objectBboxes;
        prevObjectLabels = objectLabels;
        prevFrameFlags = frameFlags;
    else
        landmarksFull = prevLandmarks;
        poseAxes = prevPoseAxes;
        gazeDir = prevGazeDir;
        gazeTarget = prevGazeTarget;
        objectBboxes = prevObjectBboxes;
        objectLabels = prevObjectLabels;
        frameFlags = prevFrameFlags;
        currentIntensity = prevIntensity;
    end

    % Evaluate if the 3D gaze target coordinate falls completely outside the predefined screen box
gazeDeviating = (gazeTarget(1) < screenLeft) || (gazeTarget(1) > screenRight) || (gazeTarget(2) < screenTop) || (gazeTarget(2) > screenBottom);
    headPoseDeviating = abs(yaw - initialYaw) > 25 || abs(pitch - initialPitch) > 25 || abs(roll - initialRoll) > 25;

    if gazeDeviating && frameIdx > 1
        if isnan(gazeDeviationStartTime)
            gazeDeviationStartTime = currentTime;
        elseif currentTime - gazeDeviationStartTime >= gazeDeviationThresholdSec && ~gazeDeviationAlarmActive
            if ~any(strcmp({frameFlags.Type}, 'Sustained Off-Screen Gaze'))
                frameFlags(end+1) = struct('Time', currentTime, 'Type', 'Sustained Off-Screen Gaze', 'Severity', 'Medium');
            end
            gazeDeviationAlarmActive = true;
        end
    else
        gazeDeviationStartTime = NaN;
        gazeDeviationAlarmActive = false;
    end

    if headPoseDeviating && frameIdx > 1
        if isnan(headPoseDeviationStartTime)
            headPoseDeviationStartTime = currentTime;
        elseif currentTime - headPoseDeviationStartTime >= headPoseDeviationThresholdSec && ~headPoseDeviationAlarmActive
            if ~any(strcmp({frameFlags.Type}, 'Sustained Head Pose Deviation'))
                frameFlags(end+1) = struct('Time', currentTime, 'Type', 'Sustained Head Pose Deviation', 'Severity', 'Medium');
            end
            headPoseDeviationAlarmActive = true;
        end
    else
        headPoseDeviationStartTime = NaN;
        headPoseDeviationAlarmActive = false;
    end
    
    % --- Module 5: Video Skip / Freeze Detection ---
    % We pass the raw grayscale frame for accurate pixel differencing
    [skipFlags, skipState] = detectVideoSkip(grayRaw, prevGrayRawForSkip, currentTime, skipState);
    prevGrayRawForSkip = grayRaw;

    % Append any skip flags to the current frame's flags so it appears on the HUD
    if ~isempty(skipFlags)
        frameFlags = [frameFlags, skipFlags];
    end

    prevIntensity = currentIntensity;
    if ~isempty(frameFlags)
        anomalyLog = [anomalyLog, frameFlags]; %#ok<AGROW>
    end

    if ~isnan(gazeTarget(1)) && ~isnan(gazeTarget(2))
        gx = max(1, min(frameWidth, round(gazeTarget(1))));
        gy = max(1, min(frameHeight, round(gazeTarget(2))));
        gazeHeatmap(gy, gx) = gazeHeatmap(gy, gx) + 1;
    end

    %% --- Module 6S: Explainability Overlay (HUD) ---
    hudFrame = frame;

    % 1. Draw Object Bounding Boxes (YOLO)
    if ~isempty(objectBboxes)
        objectBboxes = objectBboxes / procScale;
        hudFrame = insertObjectAnnotation(hudFrame, 'rectangle', objectBboxes, cellstr(objectLabels), 'LineWidth', 3);
    end

    % 2. Draw Facial Landmarks & Pose Projection
    if ~isempty(landmarksFull)
        hudFrame = insertMarker(hudFrame, landmarksFull, '*', 'Color', 'yellow', 'Size', 3);

        % Draw 3D Pose Projection Arrows (Red=X/Pitch, Green=Y/Yaw, Blue=Z/Roll)
        if ~isempty(poseAxes)
            hudFrame = insertShape(hudFrame, 'Line', poseAxes, 'Color', {'red', 'green', 'blue'}, 'LineWidth', 3);
        end

        % Draw Gaze Direction Line
        noseTip = landmarksFull(31, :);
        hudFrame = insertShape(hudFrame, 'Line', [noseTip, gazeTarget], 'Color', 'cyan', 'LineWidth', 3);
        % Draw the Screen "Safe Zone" Box on the HUD
        boxWidth = screenRight - screenLeft;
        boxHeight = screenBottom - screenTop;
        hudFrame = insertShape(hudFrame, 'Rectangle', [screenLeft, screenTop, boxWidth, boxHeight], ...
            'Color', 'green', 'LineWidth', 2);
    end

    % 3. Status Display Panel
    statusText = sprintf('Time: %.1fs | Yaw: %.1f | Pitch: %.1f | Roll: %.1f\nGaze: %s', ...
        currentTime, yaw, pitch, roll, gazeDir);
    hudFrame = insertText(hudFrame, [10 10], statusText, 'FontSize', 16, 'BoxColor', 'black', 'TextColor', 'white');

    % Highlight active flags on the HUD
    if ~isempty(frameFlags)
        flagMsg = sprintf('FLAG: %s', frameFlags(end).Type);
        hudFrame = insertText(hudFrame, [10 frameHeight - 50], flagMsg, 'FontSize', 18, 'BoxColor', 'red', 'TextColor', 'yellow');
    end

    writeVideo(vWriter, hudFrame);
end

close(vWriter);
fprintf('Video processing complete. Total elapsed time: %.1fs\n', toc(startTime));

%% 3. Post-Exam Analytics & Reporting
generateReport(anomalyLog, gazeHeatmap);

function [isMatch, score, hasFace] = verifyFaceAgainstReference(frame, grayFrame, faceDetector, referenceData)
    isMatch = false;
    score = 0;
    hasFace = false;

    if nargin < 4 || isempty(referenceData) || isempty(faceDetector)
        return;
    end

    faceBoxes = step(faceDetector, frame);
    if isempty(faceBoxes) || size(faceBoxes, 2) < 4
        return;
    end

    [~, idx] = max(faceBoxes(:, 3) .* faceBoxes(:, 4));
    faceBox = round(faceBoxes(idx, :));

    rows = max(1, faceBox(2)) : min(size(grayFrame, 1), faceBox(2) + faceBox(4) - 1);
    cols = max(1, faceBox(1)) : min(size(grayFrame, 2), faceBox(1) + faceBox(3) - 1);

    if isempty(rows) || isempty(cols)
        return;
    end

    faceROI = grayFrame(rows, cols);
    if isempty(faceROI) || numel(faceROI) < 64
        return;
    end

    refROI = referenceData.faceROI;
    if isempty(refROI) || numel(refROI) < 64
        return;
    end

    targetSize = size(refROI);
    faceROI = imresize(faceROI, targetSize);
    refROI = imresize(refROI, targetSize);

    faceROI = im2double(faceROI);
    refROI = im2double(refROI);

    % Normalize both ROIs so illumination changes do not dominate the comparison.
    faceROI = faceROI - mean(faceROI(:));
    refROI = refROI - mean(refROI(:));

    faceROI = faceROI ./ (std(faceROI(:)) + 1e-6);
    refROI = refROI ./ (std(refROI(:)) + 1e-6);

    % Use normalized cross-correlation plus a simple intensity-difference check.
    denom = sqrt(sum(faceROI(:) .^ 2) * sum(refROI(:) .^ 2));
    if denom <= eps
        score = 0;
    else
        score = sum(faceROI(:) .* refROI(:)) / denom;
    end

    diffScore = 1 - min(mean(abs(faceROI(:) - refROI(:))), 1.0);
    score = 0.7 * score + 0.3 * diffScore;

    hasFace = true;
    isMatch = score >= 0.62;
end

function [faceCount, filteredBoxes] = countReliableFaceBoxes(faceBoxes, frameSize)
    faceCount = 0;
    filteredBoxes = [];

    if isempty(faceBoxes) || size(faceBoxes, 2) < 4
        return;
    end

    faceBoxes = double(faceBoxes);
    boxAreas = faceBoxes(:, 3) .* faceBoxes(:, 4);
    minBoxArea = max(800, 0.008 * frameSize(1) * frameSize(2));
    faceBoxes = faceBoxes(boxAreas >= minBoxArea, :);

    if isempty(faceBoxes)
        return;
    end

    keep = true(size(faceBoxes, 1), 1);
    for i = 1:size(faceBoxes, 1)
        for j = i+1:size(faceBoxes, 1)
            if ~keep(i) || ~keep(j)
                continue;
            end

            x1 = max(faceBoxes(i, 1), faceBoxes(j, 1));
            y1 = max(faceBoxes(i, 2), faceBoxes(j, 2));
            x2 = min(faceBoxes(i, 1) + faceBoxes(i, 3), faceBoxes(j, 1) + faceBoxes(j, 3));
            y2 = min(faceBoxes(i, 2) + faceBoxes(i, 4), faceBoxes(j, 2) + faceBoxes(j, 4));
            interW = max(0, x2 - x1 + 1);
            interH = max(0, y2 - y1 + 1);
            interArea = interW * interH;
            unionArea = (faceBoxes(i, 3) * faceBoxes(i, 4)) + (faceBoxes(j, 3) * faceBoxes(j, 4)) - interArea;

            if unionArea > 0 && (interArea / unionArea) > 0.6
                if (faceBoxes(i, 3) * faceBoxes(i, 4)) >= (faceBoxes(j, 3) * faceBoxes(j, 4))
                    keep(j) = false;
                else
                    keep(i) = false;
                end
            end
        end
    end

    faceBoxes = faceBoxes(keep, :);
    if isempty(faceBoxes)
        return;
    end

    if size(faceBoxes, 1) < 2
        faceCount = size(faceBoxes, 1);
        filteredBoxes = faceBoxes;
        return;
    end

    centers = [faceBoxes(:, 1) + faceBoxes(:, 3)/2, faceBoxes(:, 2) + faceBoxes(:, 4)/2];
    strongSeparation = false;
    for i = 1:size(faceBoxes, 1)
        for j = i+1:size(faceBoxes, 1)
            dx = abs(centers(i, 1) - centers(j, 1));
            dy = abs(centers(i, 2) - centers(j, 2));
            if dx > 0.25 * frameSize(2) || dy > 0.25 * frameSize(1)
                strongSeparation = true;
                break;
            end
        end
        if strongSeparation
            break;
        end
    end

    faceCount = size(faceBoxes, 1);
    if ~strongSeparation
        faceCount = 1;
    end
    filteredBoxes = faceBoxes;
end