

figure;
a=randn(5,8);
imagesc(a)
colormap colorcube
axis equal;
axis off

figure;
b=randn(5,8);
imagesc(b)
colormap colorcube
axis equal;
axis off

figure;
c=randn(5,8);
imagesc(c)
colormap colorcube
axis equal;
axis off

d=[max(a'),max(b'),max(c')]

figure
imagesc(d')
colormap colorcube
axis equal;
axis off
