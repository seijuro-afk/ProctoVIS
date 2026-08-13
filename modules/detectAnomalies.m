function [flags, objBboxes, objLabels, currentIntensity] = detectAnomalies(...
    frame, grayFrame, landmarks, faceDetector, yoloDetector, yaw, pitch, roll, gazeDir, prevIntensity, timeSec, useHeavyDetection ...
    , initialYaw, initialPitch, initialRoll)

    flags = struct('Time', {}, 'Type', {}, 'Severity', {});
    objBboxes = [];
    objLabels = {};

    if nargin < 12
        useHeavyDetection = true;
    end

    % 1. Object detection: use YOLO only for contraband/device boxes and use the
    %    face detector for the person/face box so the overlay is anchored to the face.
    if useHeavyDetection
        [bboxes, scores, labels] = detect(yoloDetector, frame, 'Threshold', 0.45);

        % Keep only contraband/device objects from YOLO.
        validClasses = {'cell phone', 'laptop', 'book'};
        labelCells = cellstr(labels);
        keepIdx = ismember(labelCells, validClasses);
        objBboxes = bboxes(keepIdx, :);
        objLabels = labelCells(keepIdx);

        if ~isempty(faceDetector)
            faceBboxes = step(faceDetector, frame);
            if ~isempty(faceBboxes)
                faceBboxes = double(faceBboxes);
                objBboxes = [objBboxes; faceBboxes];
                faceLabels = repmat({'person'}, size(faceBboxes, 1), 1);
                objLabels = [objLabels; faceLabels];
            end
        end
    elseif ~isempty(faceDetector)
        bboxes = step(faceDetector, frame);
        if ~isempty(bboxes)
            objBboxes = double(bboxes);
            objLabels = repmat({'person'}, size(bboxes, 1), 1);
        end
    end

    if isempty(objLabels)
        objLabels = {};
    else
        objLabels = cellstr(objLabels);
    end

    personCount = sum(cellfun(@(x) strcmp(x, 'person'), objLabels));
    phoneCount  = sum(cellfun(@(x) strcmp(x, 'cell phone'), objLabels));

    personBoxes = [];
    if ~isempty(objBboxes) && ~isempty(objLabels)
        personMask = strcmp(objLabels, 'person');
        if any(personMask)
            personBoxes = objBboxes(personMask, :);
        end
    end

    reliablePersonCount = 0;
    if ~isempty(personBoxes)
        [reliablePersonCount, ~] = countReliableFaceBoxes(personBoxes, size(frame));
    end

    if reliablePersonCount > 1
        flags(end+1) = struct('Time', timeSec, 'Type', 'Multiple People Detected', 'Severity', 'High');
    elseif personCount == 0 || isempty(landmarks)
        flags(end+1) = struct('Time', timeSec, 'Type', 'Student Occluded / Missing', 'Severity', 'High');
    end

    if phoneCount > 0
        flags(end+1) = struct('Time', timeSec, 'Type', 'Secondary Device Detected', 'Severity', 'Critical');
    end
    
    % 2. Sudden Illumination Spikes (Screen Glow / Secondary Monitor)
    currentIntensity = mean2(grayFrame);
    if prevIntensity > 0 && abs(currentIntensity - prevIntensity) > 25
        flags(end+1) = struct('Time', timeSec, 'Type', 'Sudden Light Flare', 'Severity', 'Medium');
    end
    
    % 3. Head Pose Anomalies (only after sustained deviation)
    persistent poseDeviationCount poseAlarmActive;
    persistent gazeDeviationCount;
    
    if isempty(poseAlarmActive)
        poseAlarmActive = false;
    end

    if isempty(poseDeviationCount)
        poseDeviationCount = 0;
    end
    if isempty(gazeDeviationCount)
        gazeDeviationCount = 0;
    end

    poseIsDeviated = abs(yaw - initialYaw) > 25 || abs(pitch - initialPitch) > 25 || abs(roll - initialRoll) > 25;
    gazeIsDeviated = strcmp(gazeDir, 'LOOKING LEFT') || strcmp(gazeDir, 'LOOKING RIGHT');

    if poseIsDeviated
        poseDeviationCount = poseDeviationCount + 1;
    else
        poseDeviationCount = 0;
        poseAlarmActive = false;
    end

    if gazeIsDeviated
        gazeDeviationCount = gazeDeviationCount + 1;
    else
        gazeDeviationCount = 0;
    end

    if poseDeviationCount >= 6
        flags(end+1) = struct('Time', timeSec, 'Type', 'Sustained Pose Turning', 'Severity', 'Medium');
        poseAlarmActive = true;
    end

    % 4. Off-Screen Gaze Flags (only after sustained deviation)
    if gazeDeviationCount >= 6
        flags(end+1) = struct('Time', timeSec, 'Type', 'Off-Screen Eye Shift', 'Severity', 'Low');
    end
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