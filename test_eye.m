function simpleEyeTracker
    % Initialize webcam
    cam = webcam;

    % Face detector
    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
    eyeDetector  = vision.CascadeObjectDetector('EyePairBig');

    % --- Calibration ---
    screenSize = get(0, 'ScreenSize');
    calibTargets = [screenSize(3)*0.5 screenSize(4)*0.5;
                    screenSize(3)*0.1 screenSize(4)*0.1;
                    screenSize(3)*0.9 screenSize(4)*0.1;
                    screenSize(3)*0.1 screenSize(4)*0.9;
                    screenSize(3)*0.9 screenSize(4)*0.9];
    pupilCenters = zeros(size(calibTargets));

    disp('Calibration: Look at each red dot and press a key.');

    for i = 1:size(calibTargets,1)
        figure(2);
        set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]); % full screen
        clf;

        % Draw calibration dot (flipped Y for screen coordinates)
        scatter(calibTargets(i,1), screenSize(4) - calibTargets(i,2), 300, 'r', 'filled');
        axis([0 screenSize(3) 0 screenSize(4)]);
        title('Calibration');
        pause(2); % keep visible for 2 seconds
        waitforbuttonpress;

        frame = snapshot(cam);
        faceBbox = step(faceDetector, frame);

        if ~isempty(faceBbox)
            eyesBbox = step(eyeDetector, imcrop(frame, faceBbox(1,:)));

            if ~isempty(eyesBbox)
                eyeRegion = imcrop(frame, eyesBbox(1,:));
                grayEye   = rgb2gray(eyeRegion);
                bwEye     = imbinarize(grayEye, 'adaptive');
                stats     = regionprops(bwEye, 'Centroid', 'Area');

                if ~isempty(stats)
                    [~, idx] = max([stats.Area]);
                    pupilCenters(i,:) = stats(idx).Centroid;
                else
                    disp('No pupil detected, retrying...');
                    i = i - 1;
                end
            else
                disp('No eyes detected, retrying...');
                i = i - 1;
            end
        else
            disp('No face detected, retrying...');
            i = i - 1;
        end
    end

    % Fit mapping functions
    coeffX = polyfit(pupilCenters(:,1), calibTargets(:,1), 2);
    coeffY = polyfit(pupilCenters(:,2), calibTargets(:,2), 2);

    % --- Live Tracking ---
    hFig = figure('Name','Eye Tracker','NumberTitle','off');
    frameCount = 0;  % counter for downsampling
    lastPupil = [];
    lastGaze  = [];

    while ishandle(hFig)
        frame = snapshot(cam);
        frameCount = frameCount + 1;

        % Only process every 4th frame
        if mod(frameCount, 4) == 0
            faceBbox = step(faceDetector, frame);

            if ~isempty(faceBbox)
                eyesBbox = step(eyeDetector, imcrop(frame, faceBbox(1,:)));

                if ~isempty(eyesBbox)
                    eyeRegion = imcrop(frame, eyesBbox(1,:));
                    grayEye   = rgb2gray(eyeRegion);
                    bwEye     = imbinarize(grayEye, 'adaptive');
                    stats     = regionprops(bwEye, 'Centroid', 'Area');

                    if ~isempty(stats)
                        [~, idx] = max([stats.Area]);
                        pupilCenter = stats(idx).Centroid;

                        % Adjust pupil coordinates to full-frame space
                        pupilCenterGlobal = pupilCenter + eyesBbox(1,1:2);

                        % Predict gaze position (normalized → pixels)
                        gazeX = polyval(coeffX, pupilCenter(1));
                        gazeY = polyval(coeffY, pupilCenter(2));

                        % Clamp gaze coordinates to frame size
                        gazeX = max(1, min(size(frame,2), gazeX));
                        gazeY = max(1, min(size(frame,1), gazeY));

                        % Save last positions
                        lastPupil = pupilCenterGlobal;
                        lastGaze  = [gazeX, gazeY];
                    end
                end
            end
        end

        % Always draw the last known markers
        if ~isempty(lastPupil)
            frame = insertMarker(frame, lastPupil, 'o', 'Color', 'red');
        end
        if ~isempty(lastGaze)
            frame = insertMarker(frame, lastGaze, 'o', 'Color', 'green');
        end

        imshow(frame);
        drawnow;
    end

    clear cam
end
