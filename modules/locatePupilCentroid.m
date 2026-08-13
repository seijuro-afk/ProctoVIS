function pupilCenter = locatePupilCentroid(grayFrame, eyePts)
    pupilCenter = [];

    if nargin < 2 || isempty(grayFrame) || isempty(eyePts)
        return;
    end

    eyePts = double(eyePts);
    eyePts = eyePts(isfinite(eyePts(:,1)) & isfinite(eyePts(:,2)), :);
    if size(eyePts, 1) < 3 || size(eyePts, 2) ~= 2
        return;
    end

    if ndims(grayFrame) ~= 2
        grayFrame = rgb2gray(grayFrame);
    end

    xMin = max(1, floor(min(eyePts(:,1))));
    xMax = min(size(grayFrame, 2), ceil(max(eyePts(:,1))));
    yMin = max(1, floor(min(eyePts(:,2))));
    yMax = min(size(grayFrame, 1), ceil(max(eyePts(:,2))));

    if xMax <= xMin || yMax <= yMin
        return;
    end

    eyePatch = im2double(grayFrame(yMin:yMax, xMin:xMax));
    [rows, cols] = size(eyePatch);
    centerY = round(rows / 2);
    centerX = round(cols / 2);
    border = max(2, round(min(rows, cols) * 0.15));

    mask = false(rows, cols);
    yStart = max(1, centerY - border);
    yEnd = min(rows, centerY + border);
    xStart = max(1, centerX - border);
    xEnd = min(cols, centerX + border);
    mask(yStart:yEnd, xStart:xEnd) = true;

    if ~any(mask(:))
        mask(:) = true;
    end

    candidateCoords = find(mask);
    candidateValues = eyePatch(mask);
    if numel(candidateValues) < 1
        pupilCenter = mean(eyePts, 1);
        return;
    end

    [~, sortIdx] = sort(candidateValues);
    sampleCount = min(20, numel(sortIdx));
    pickedCoords = candidateCoords(sortIdx(1:sampleCount));
    [py, px] = ind2sub(size(mask), pickedCoords);
    pupilCenter = [mean(px), mean(py)] + [xMin - 1, yMin - 1];
end
