
function quickPlot(matArray)
figure()
imagesc(matArray)
colorbar;
axis image;
clim([prctile(matArray(:),5),prctile(matArray(:),95)]);



