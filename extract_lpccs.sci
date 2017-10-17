
function extract_lpcc(file_name,P,wsize,overlap)
    [y,fs,bt]=wavread(file_name);
    y=y./(1.01*abs(max(y)));
    row=1
    for N=1:overlap:length(y)-overlap
        if N+wsize<length(y)
            yi=y(N:N+wsize);
            w=window('hm',length(yi));
            yi=yi.*w;
            ycorr=corr(yi,overlap);
            ycorr=ycorr./(abs(max(ycorr)));
            A=ycorr(1:P);
            r=ycorr(2:(P+1));
            A=toeplitz(A);
            L=-inv(A)*mtlb_t(r);
            
            L=mtlb_t(L);
            
            LPCoeffs(1,1:length([1,L]))=[1,L];
            features(row,1:P+1)=LPCoeffs(1:P+1);
            row=row+1
        end
    end
    disp(file_name+'_lpcc')
    fprintfMat(file_name+'_lpcc',features);
endfunction

clear all;
clc;
directory='train_lpccdata/'
folders=findfiles(directory,'m*')
[nfolders,temp]=size(folders)
for i=3:nfolders
    current=findfiles(directory+folders(i),'*.wav')
    [nfiles,temp]=size(current)
    for j=1:nfiles
        extract_lpcc(directory+folders(i)+'/'+current(j),10,320,160);
        deletefile(directory+folders(i)+'/'+current(j));
    end
end
