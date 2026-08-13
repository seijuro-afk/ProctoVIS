function detected = detectPhoneLight(procFrame, grayRaw, grayEnhanced, faceDetector, referenceData)
% detectPhoneLight Detect a possible phone light using a reference-based brightness rule.
%
% Inputs:
%   procFrame      - processed RGB frame used for face detection
%   grayRaw        - raw grayscale frame for brightness measurement
%   grayEnhanced   - histogram-equalized/imadjusted grayscale frame for detection
%   faceDetector   - MATLAB cascade detector
%   referenceData  - baseline face/background brightness stats from the reference frame
%
% Output:
%   detected       - true if a phone-light-like increase is detected

    detected = false;

    if nargin < 5 || isempty(referenceData)
        return;
    end

    faceBoxes = step(faceDetector, procFrame);
    if isempty(faceBoxes) || size(faceBoxes, 2) < 4
        return;
    end

    % Pick the largest face box.
    [~, idx] = max(faceBoxes(:, 3) .* faceBoxes(:, 4));
    faceBox = round(faceBoxes(idx, :));

    x1 = max(1, faceBox(1));
    y1 = max(1, faceBox(2));
    x2 = min(size(procFrame, 2), x1 + faceBox(3) - 1);
    y2 = min(size(procFrame, 1), y1 + faceBox(4) - 1);

    faceROI = grayRaw(y1:y2, x1:x2);
    if isempty(faceROI)
        return;
    end

    faceMean = mean(double(faceROI(:)));
    faceStd = std(double(faceROI(:)));

    mask = true(size(grayRaw));
    mask(y1:y2, x1:x2) = false;
    bgROI = grayRaw(mask);
    bgMean = mean(double(bgROI(:)));
    bgStd = std(double(bgROI(:)));

    faceRef = referenceData.faceBrightness;
    bgRef = referenceData.backgroundBrightness;

    % Use the baseline reference to detect a sustained, face-localized increase.
    faceIncrease = faceMean - faceRef;
    faceVsBg = faceMean - bgMean;
    faceVsBgRef = faceRef - bgRef;

    % A phone light typically brightens the face more than the background and
    % especially the lower half of the face.
    lowerHalf = max(1, round(size(faceROI, 1) * 0.5));
    if lowerHalf >= size(faceROI, 1)
        lowerROI = faceROI;
        upperROI = faceROI;
    else
        lowerROI = faceROI(lowerHalf:end, :);
        upperROI = faceROI(1:lowerHalf-1, :);
    end
    lowerMean = mean(double(lowerROI(:)));
    upperMean = mean(double(upperROI(:)));
    lowerBoost = lowerMean - upperMean;

    % Heuristic thresholds tuned for gradual illumination changes.
    brightEnough = faceIncrease > 8;
    faceDominatesBg = faceVsBg > max(5, 0.2 * max(faceRef, 1));
    lowerHalfBoost = lowerBoost > 3;
    stableFace = faceStd < 80;
    backgroundNotTooBright = bgMean < (bgRef + 25);

    detected = brightEnough && faceDominatesBg && lowerHalfBoost && stableFace && backgroundNotTooBright;
end
