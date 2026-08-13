function [smoothed, thresholds] = poseSignalTools(currentPose, history, windowSize)
    if nargin < 3 || isempty(windowSize)
        windowSize = 5;
    end

    if nargin < 2 || isempty(history)
        history = zeros(0, 3);
    end

    if isempty(currentPose)
        smoothed = [];
        thresholds = [];
        return;
    end

    currentPose = double(currentPose(:)');
    if numel(currentPose) < 3
        currentPose = [currentPose, zeros(1, 3 - numel(currentPose))];
    end

    history = double(history);
    if size(history, 2) ~= 3
        history = history(:)';
        if numel(history) < 3
            history = [history, zeros(1, 3 - numel(history))];
        end
        history = reshape(history, [max(1, size(history, 1)), 3]);
    end

    if size(history, 1) > windowSize
        history = history(end-windowSize+1:end, :);
    end

    if isempty(history)
        smoothed = currentPose;
    else
        recent = [history; currentPose];
        smoothed = mean(recent, 1);
    end

    baselineStd = std(history, 0, 1);
    if isempty(baselineStd)
        baselineStd = zeros(1, 3);
    end
    thresholds = max(8, 2 * baselineStd);
end
