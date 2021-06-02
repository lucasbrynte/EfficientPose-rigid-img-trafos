function imwarp = warp_an_image(im,K,interpolation_method,one_based_indexing);
%imwarp 
% input: im - m x n x 3 in UINT8

if nargin < 3
    interpolation_method = 'bilinear';
    % interpolation_method = 'nearest_neighbor';
end

if nargin < 4
    % one_based_indexing = true;
    one_based_indexing = false;
end

warpmethod = 2;

[mm,nn,~]=size(im);
imwarp=zeros(mm,nn,3);

px = K(1,3);
py = K(2,3);
if one_based_indexing
    % xx:
    % -----------------------------------
    % || px  px  1   nn  1  1   nn  nn ||
    % || 1   mm  py  py  1  mm  mm  1  ||
    % -----------------------------------
    % xx=pextend([px,1;px,mm;1,py;nn,py;1,1;1,mm;nn,mm;nn,1]');

    % xx:
    % -----------------------------------
    % || px  px  1   nn ||
    % || 1   mm  py  py ||
    % -----------------------------------
    xx=pextend([px,1;px,mm;1,py;nn,py]');
else
    % xx:
    % -----------------------------------
    % || px  px     0  nn-1  0     0   nn-1  nn-1  ||
    % || 0   mm-1  py    py  0  mm-1   mm-1     0  ||
    % -----------------------------------
    % xx=pextend([px,0;px,mm-1;0,py;nn-1,py;0,0;0,mm-1;nn-1,mm-1;nn-1,0]');

    % xx:
    % -----------------------------------
    % || px  px     0  nn-1 ||
    % || 0   mm-1  py    py ||
    % -----------------------------------
    xx=pextend([px,0;px,mm-1;0,py;nn-1,py]');
end
xxnorm=inv(K)*xx;
xxsph = psphere(xxnorm); % Backproject 2D points (in homogeneous coord) to unit sphere

th1 = acos(xxsph(:,1)'*xxsph(:,2)); % Angle between image rays at min/max y coord.
th1start=-acos(xxsph(3,1)); % Angle between principal axis and image ray at min y coord.
th1end=acos(xxsph(3,2)); % Angle between principal axis and image ray at max y coord.

% th1list = th1start:th1/(mm-1):th1end;
th1list = linspace(th1start, th1end, mm);
n1 = [zeros(1,mm);cos(th1list);-sin(th1list)];

th2 = acos(xxsph(:,3)'*xxsph(:,4)); % Angle between image rays at min/max x coordinate.
th2start=-acos(xxsph(3,3));
th2end=acos(xxsph(3,4));

% th2list = th2start:th2/(nn-1):th2end;
th2list = linspace(th2start, th2end, nn);
n2 = [cos(th2list);zeros(1,nn);-sin(th2list)];

xxgrid=zeros(3,mm,nn);
if warpmethod ==1,
    for ii=1:mm,
        for jj=1:nn,
            xxgrid(:,ii,jj)=normc(cross(n2(:,jj),n1(:,ii)));
        end
    end
else
    e1 = [1,0,0]';
    e2 = [0,1,0]';
    e3 = [0,0,1]';
    E1 = skew(e1);
    E2 = skew(e2);
    for ii=1:mm,
        for jj=1:nn,
            % Note: th1 angles correspond to y-shift, but is multiplied with Sx, and vice versa.
            % Probably this is due to an in-plane 90 deg rotation of the (th1, th2) point, in order to get the rotation axis, further explaining the negative sign for the Sy component.
            R = expm(th1list(ii)*E1-th2list(jj)*E2);
            xxgrid(:,ii,jj)=R'*e3;
        end
    end    
end

xxflat = xxgrid;
for jj=1:nn,
    xxflat(:,:,jj) = K*pflat(xxflat(:,:,jj));
end
if ~one_based_indexing
    % 2D points are not indexed starting from zero, but Matlab's indexing is inherently one-based.
    % Add 1 to all (per-ii-jj-pixel) target points before warping.
    % Could alternatively have been done in the warping, on the xc / xcf variables.
    xxflat(1:2,:,:) = xxflat(1:2,:,:) + 1;
end

imwarp = uint8(zeros(size(im)));

for ii=1:mm,
    for jj=1:nn,
        xc = xxflat(1:2,ii,jj);
        if strcmp(interpolation_method, 'bilinear')
            xcf = floor(xc);
            l1 = xc(1)-xcf(1);
            l2 = xc(2)-xcf(2);
            tmp = zeros(2,2,3);
            if xcf(1)>=1 && xcf(1)<=nn && xcf(2)>=1 && xcf(2)<=mm,
                tmp(1,1,:) = double(im(xcf(2),xcf(1),:));
            end
            if xcf(1)>=1 && xcf(1)<=nn && xcf(2)>=0 && xcf(2)<mm,
                tmp(2,1,:) = double(im(xcf(2)+1,xcf(1),:));
            end
            if xcf(1)>=0 && xcf(1)<nn && xcf(2)>=1 && xcf(2)<=mm,
                tmp(1,2,:) = double(im(xcf(2),xcf(1)+1,:));
            end
            if xcf(1)>=0 && xcf(1)<nn && xcf(2)>=0 && xcf(2)<mm,
                tmp(2,2,:) = double(im(xcf(2)+1,xcf(1)+1,:));
            end
            tmp2 = [(1-l2)*((1-l1)*tmp(1,1,:)+l1*tmp(1,2,:))+l2*((1-l1)*tmp(2,1,:)+l1*tmp(2,2,:))];
            col = uint8(round(tmp2));
            imwarp(ii, jj, :) = col;
        elseif strcmp(interpolation_method, 'nearest_neighbor')
            xcr = round(xc);
            val = 0;
            if xcr(1)>=1 && xcr(1)<=nn && xcr(2)>=1 && xcr(2)<=mm
                val = im(xcr(2), xcr(1), :);
            end
            imwarp(ii, jj, :) = val;
        else
            assert(false);
        end
    end
end
