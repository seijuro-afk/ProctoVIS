function referenceData = getReferenceData(frame,grayFrame,faceDetector)

referenceData = struct();

faceBoxes = step(faceDetector,frame);

if isempty(faceBoxes)

    error("No face detected in reference frame.");

end

% Largest face
[~,idx] = max(faceBoxes(:,3).*faceBoxes(:,4));

faceBox = round(faceBoxes(idx,:));

rows = faceBox(2):(faceBox(2)+faceBox(4)-1);
cols = faceBox(1):(faceBox(1)+faceBox(3)-1);

rows = max(1,min(size(grayFrame,1),rows));
cols = max(1,min(size(grayFrame,2),cols));

faceROI = grayFrame(rows,cols);

mask = true(size(grayFrame));
mask(rows,cols) = false;

backgroundROI = grayFrame(mask);

referenceData.faceBox = faceBox;
referenceData.faceROI = faceROI;

referenceData.faceBrightness = ...
    mean(double(faceROI(:)));

referenceData.faceStd = ...
    std(double(faceROI(:)));

referenceData.backgroundBrightness = ...
    mean(double(backgroundROI(:)));

referenceData.backgroundStd = ...
    std(double(backgroundROI(:)));

referenceData.frame = frame;

referenceData.gray = grayFrame;

end