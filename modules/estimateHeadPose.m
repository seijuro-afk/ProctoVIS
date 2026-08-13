function [yaw, pitch, roll, poseAxes] = estimateHeadPose(landmarks, camIntrinsics)
    yaw = 0; pitch = 0; roll = 0; poseAxes = [];

    if nargin < 2 || isempty(camIntrinsics)
        return;
    end

    landmarks = double(landmarks);
    if ndims(landmarks) ~= 2 || size(landmarks, 2) ~= 2
        return;
    end

    landmarks = landmarks(all(isfinite(landmarks), 2), :);
    if size(landmarks, 1) < 3
        return;
    end

    % 3D reference model points [X, Y, Z] in mm
    model3D = double([ ...
        0.0,    0.0,    0.0;     % Nose tip
        0.0,  -330.0,  -65.0;    % Chin
       -225.0, 170.0, -135.0;    % Left Eye outer corner
        225.0, 170.0, -135.0;    % Right Eye outer corner
       -150.0,-150.0, -125.0;    % Left Mouth corner
        150.0,-150.0, -125.0]);  % Right Mouth corner

    % Matching 2D landmarks (indices for 68-point model)
    landmarkIdx = [31, 9, 37, 46, 49, 55];
    availableIdx = landmarkIdx(landmarkIdx <= size(landmarks, 1));

    if numel(availableIdx) >= 3
        points2D = landmarks(availableIdx, :);
        points3D = model3D(1:numel(availableIdx), :);
    else
        nPoints = min(6, size(landmarks, 1));
        points2D = landmarks(1:nPoints, :);
        points3D = model3D(1:nPoints, :);
    end

    if size(points2D, 1) < 3 || size(points3D, 1) < 3
        return;
    end

    poseValid = false;
    worldOrientation = eye(3);
    worldTranslation = [0; 0; 0];

    % Estimate camera pose (PnP) and fall back safely if the solver rejects the input.
    try
        [worldOrientation, worldTranslation, poseValid] = estimateWorldCameraPose( ...
            points2D, points3D, camIntrinsics, ...
            'Confidence', 99.0, ...
            'MaxNumTrials', 5000);
    catch
        poseValid = false;
    end

    if ~poseValid
        if size(landmarks, 1) >= 46
            leftEye = mean(landmarks(37:42, :), 1);
            rightEye = mean(landmarks(43:48, :), 1);
            nose = landmarks(31, :);
            chin = landmarks(9, :);
            eyeMid = (leftEye + rightEye) / 2;
            faceWidth = max(1.0, norm(rightEye - leftEye));
            faceHeight = max(1.0, norm(nose - chin));

            yaw = rad2deg(atan2(nose(1) - eyeMid(1), faceWidth));
            pitch = rad2deg(atan2(eyeMid(2) - nose(2), faceHeight));
            roll = rad2deg(atan2(rightEye(2) - leftEye(2), faceWidth));

            yaw = max(-90, min(90, yaw));
            pitch = max(-90, min(90, pitch));
            roll = max(-90, min(90, roll));
        end
    else
        % Convert rotation matrix to yaw/pitch/roll
        R = worldOrientation;
        pitch = rad2deg(atan2(-R(3,1), sqrt(R(3,2)^2 + R(3,3)^2)));
        yaw   = rad2deg(atan2(R(2,1), R(1,1)));
        roll  = rad2deg(atan2(R(3,2), R(3,3)));
    end

    % Build axes for HUD overlay
    axisLength = 100;
    axes3D = [0 0 0; axisLength 0 0; 0 axisLength 0; 0 0 axisLength];
    if size(landmarks, 1) < 31
        return;
    end

    nose2D = landmarks(31, :);
    if any(~isfinite(nose2D))
        return;
    end

    if poseValid
        projected2D = worldToImage(camIntrinsics, worldOrientation, worldTranslation, axes3D);
        poseAxes = [ nose2D, projected2D(2,:);  % Pitch / Red
                     nose2D, projected2D(3,:);  % Yaw / Green
                     nose2D, projected2D(4,:)]; % Roll / Blue
    else
        poseAxes = [ nose2D, [nose2D(1)+axisLength, nose2D(2)];
                     nose2D, [nose2D(1), nose2D(2)+axisLength];
                     nose2D, [nose2D(1)+axisLength, nose2D(2)+axisLength]];
    end
end
