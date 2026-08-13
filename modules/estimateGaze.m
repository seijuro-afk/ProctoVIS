function [gazeDir, gazeTarget] = estimateGaze(grayFrame, landmarks, frameWidth, frameHeight)
    gazeDir = 'FORWARD';
    gazeTarget = [frameWidth/2, frameHeight/2];

    if nargin < 4 || isempty(grayFrame) || isempty(landmarks)
        return;
    end

    landmarks = double(landmarks);
    if size(landmarks, 1) < 6
        return;
    end

    % Extract eye regions (Landmarks 37-42: Left Eye, 43-48: Right Eye)
    leftEyePts  = landmarks(37:42, :);
    rightEyePts = landmarks(43:48, :);
    leftEyePts  = leftEyePts(all(isfinite(leftEyePts), 2), :);
    rightEyePts = rightEyePts(all(isfinite(rightEyePts), 2), :);

    if size(leftEyePts, 1) < 3 || size(rightEyePts, 1) < 3
        % Fallback from coarse face geometry if the eye landmarks are incomplete.
        if size(landmarks, 1) >= 31
            nose = landmarks(31, :);
            leftEye = landmarks(37, :);
            rightEye = landmarks(46, :);
            if all(isfinite([nose; leftEye; rightEye]))
                eyeMid = (leftEye + rightEye) / 2;
                faceWidth = max(1.0, norm(rightEye - leftEye));
                hRatio = (nose(1) - eyeMid(1)) / faceWidth;
                vRatio = (nose(2) - eyeMid(2)) / max(1.0, faceWidth);
                gazeDir = 'CENTER / FORWARD';
                if hRatio < -0.15
                    gazeDir = 'LOOKING LEFT';
                elseif hRatio > 0.15
                    gazeDir = 'LOOKING RIGHT';
                end
                if vRatio < -0.1
                    gazeDir = [gazeDir, ' / UP'];
                elseif vRatio > 0.1
                    gazeDir = [gazeDir, ' / DOWN'];
                end
                targetX = frameWidth * max(0.1, min(0.9, 0.5 + 0.35 * hRatio));
                targetY = frameHeight * max(0.1, min(0.9, 0.5 + 0.25 * vRatio));
                gazeTarget = [targetX, targetY];
            end
        end
        return;
    end

    % Analyze both pupils; fall back to a simple center estimate if needed.
    leftPupil  = locatePupilCentroid(grayFrame, leftEyePts);
    rightPupil = locatePupilCentroid(grayFrame, rightEyePts);

    if isempty(leftPupil) || isempty(rightPupil)
        leftPupil = mean(leftEyePts, 1);
        rightPupil = mean(rightEyePts, 1);
    end

    % Compute eye bounding boxes
    leftCorner  = leftEyePts(1, :);
    rightCorner = leftEyePts(4, :);
    leftWidth   = norm(rightCorner - leftCorner);

    rLeftCorner  = rightEyePts(1, :);
    rRightCorner = rightEyePts(4, :);
    rightWidth   = norm(rRightCorner - rLeftCorner);

    if leftWidth == 0 || rightWidth == 0
        leftWidth = max(1.0, norm(mean(leftEyePts,1) - mean(rightEyePts,1)));
        rightWidth = leftWidth;
    end

    % Horizontal displacement ratios for both eyes
    hRatioLeft  = (leftPupil(1) - leftCorner(1)) / leftWidth;
    hRatioRight = (rightPupil(1) - rLeftCorner(1)) / rightWidth;

    % Average displacement ratio
    hRatio = mean([hRatioLeft, hRatioRight]);

    % Determine discrete direction vector with a wider neutral band for frontal viewing.
    if hRatio < -0.30
        gazeDir = 'LOOKING LEFT';
        targetX = frameWidth * 0.1;
    elseif hRatio > 0.30
        gazeDir = 'LOOKING RIGHT';
        targetX = frameWidth * 0.9;
    else
        gazeDir = 'CENTER / FORWARD';
        targetX = frameWidth * 0.5;
    end

    % Vertical displacement (optional refinement)
    vRatioLeft  = (leftPupil(2) - min(leftEyePts(:,2))) / max(1.0, max(leftEyePts(:,2)) - min(leftEyePts(:,2)));
    vRatioRight = (rightPupil(2) - min(rightEyePts(:,2))) / max(1.0, max(rightEyePts(:,2)) - min(rightEyePts(:,2)));
    vRatio = mean([vRatioLeft, vRatioRight]);

    if vRatio < 0.35
        gazeDir = [gazeDir, ' / UP'];
    elseif vRatio > 0.65
        gazeDir = [gazeDir, ' / DOWN'];
    end

    targetY = frameHeight * (0.5 + 0.2 * (vRatio - 0.5));
    targetY = max(1, min(frameHeight, targetY));
    gazeTarget = [targetX, targetY];
end
