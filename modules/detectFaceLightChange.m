function [flagTriggered, state] = detectFaceLightChange(grayRawFrame, faceBox, currentTime, state)
%DETECTFACELIGHTCHANGE Flag sudden brightness jumps measured on the face ROI.
%
% Scoped to the face region (not the whole frame) so it reacts to light
% actually falling on the subject -- e.g. a monitor going from a bright
% exam page to a dark IDE/terminal during alt-tab, or vice versa -- rather
% than to whatever background clutter is in frame. Uses a rolling
% z-score baseline (not a single frame-to-frame diff) so it self-adapts
% to normal room lighting/flicker and only fires on genuine outliers.
%
% IMPORTANT: pass the RAW grayscale frame (grayRaw in main_proctor.m),
% not the histeq/imadjust-enhanced one (grayFrame). The enhanced version
% is contrast-normalized per frame and will suppress the exact signal
% this function looks for.
%
% INPUTS
%   grayRawFrame - current raw grayscale frame (uint8), same scale/coords
%                  as faceBox (i.e. procFrame-scale when
%                  useLowResProcessing is true)
%   faceBox      - [x y w h] of the primary/largest detected face.
%                  Pass [] when no face was detected this call.
%   currentTime  - current video/session time in seconds (currentTime in
%                  main_proctor.m)
%   state        - struct from initFaceLightState(), threaded through
%                  calls
%
% OUTPUTS
%   flagTriggered - true if a sudden lighting change was detected now
%   state         - updated state to pass into the next call

    flagTriggered = false;

    if isempty(faceBox) || numel(faceBox) < 4
        return;  % don't corrupt the baseline with no-face frames
    end

    rows = max(1, round(faceBox(2))) : min(size(grayRawFrame, 1), round(faceBox(2) + faceBox(4) - 1));
    cols = max(1, round(faceBox(1))) : min(size(grayRawFrame, 2), round(faceBox(1) + faceBox(3) - 1));
    if isempty(rows) || isempty(cols)
        return;
    end

    faceROI = grayRawFrame(rows, cols);
    faceBrightness = mean(double(faceROI(:)));

    % Update rolling circular buffer
    slot = mod(state.bufIdx, state.windowSize) + 1;
    state.buffer(slot) = faceBrightness;
    state.bufIdx = state.bufIdx + 1;

    validVals = state.buffer(~isnan(state.buffer));
    if numel(validVals) < max(5, round(state.windowSize / 3))
        return;  % not enough history yet to trust a baseline
    end

    baselineMean = mean(validVals);
    baselineStd = std(validVals);
    if baselineStd < 1e-3
        baselineStd = 1e-3;  % guard divide-by-zero on very flat sequences
    end

    zscore = (faceBrightness - baselineMean) / baselineStd;
    suddenChange = abs(zscore) >= state.zScoreThreshold && ...
                   abs(faceBrightness - baselineMean) >= state.absDeltaThreshold;

    if suddenChange && (currentTime - state.lastFlagTime) >= state.minSecBetweenFlags
        flagTriggered = true;
        state.lastFlagTime = currentTime;
    end
end
