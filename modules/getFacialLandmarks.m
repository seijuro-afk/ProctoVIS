function [landmarks, bestVariant] = getFacialLandmarks(frame, faceDetector, landmarkNet)
    % Detect face bounding boxes
    bboxes = step(faceDetector, frame);
    if isempty(bboxes)
        landmarks = [];
        bestVariant = struct('swapXY', false, 'flipY', false, 'score', 0);
        return;
    end

    % Pick largest bounding box
    [~, maxIdx] = max(bboxes(:,3) .* bboxes(:,4));
    bbox = double(bboxes(maxIdx, :));

    % Add padding
    pad = 15;
    x = max(1, bbox(1) - pad);
    y = max(1, bbox(2) - pad);
    w = min(size(frame,2) - x, bbox(3) + 2*pad);
    h = min(size(frame,1) - y, bbox(4) + 2*pad);

    % Crop and resize
    faceCrop = imcrop(frame, [x, y, w, h]);
    targetSize = [128, 128];
    faceResized = imresize(faceCrop, targetSize);

    % Ensure RGB
    if size(faceResized,3) == 1
        faceResized = repmat(faceResized, [1 1 3]);
    end

    % Normalize to [-1,1]
    imgNorm = (single(faceResized) / 127.5) - 1.0;
    imgBatch = reshape(imgNorm, [1, targetSize(1), targetSize(2), 3]);

    % Run ONNX model
    rawPred = landmarkNet(imgBatch, 'InputDataPermutation','none');
    if isa(rawPred, 'dlarray')
        rawPred = extractdata(rawPred);
    end

    % Reshape output
    dense = reshape(rawPred, [2, numel(rawPred)/2])';
    assumeNorm = max(dense(:)) <= 1.01;

    % Try variants
    swapOptions = [false true];
    flipOptions = [false true];
    variants = struct(); k = 1;
    for s = swapOptions
        for f = flipOptions
            pts = dense;
            if s, pts = pts(:, [2 1]); end
            if assumeNorm
                pts(:,1) = pts(:,1) * targetSize(2);
                pts(:,2) = pts(:,2) * targetSize(1);
            end
            scaleX = w / targetSize(2);
            scaleY = h / targetSize(1);
            pts(:,1) = pts(:,1) * scaleX + x;
            pts(:,2) = pts(:,2) * scaleY + y;
            if f
                pts(:,2) = y + h - (pts(:,2) - y);
            end
            margin = 5;
            inside = pts(:,1) >= (x - margin) & pts(:,1) <= (x + w + margin) & ...
                     pts(:,2) >= (y - margin) & pts(:,2) <= (y + h + margin);
            score = sum(inside) / size(pts,1);
            variants(k).swapXY = s;
            variants(k).flipY = f;
            variants(k).score = score;
            variants(k).pts = pts;
            k = k + 1;
        end
    end

    % Pick best variant
    [~, bestIdx] = max([variants.score]);
    bestVariant = variants(bestIdx);
    landmarks = bestVariant.pts;
end
