
REM this run recurrent network language model experiments
REM baseline is class-based LSTM
REM words are clustered in frequency. Each class has equal probability mass. 
REM 

set WORKDIR=c:\dev\mycnn\exp\lm
set DATADIR=c:/data/ptbdata
set BINDIR=C:\dev\mycnn\msbuildx64\examples\Release

mkdir %WORKDIR%\logs
mkdir %WORKDIR%\models

call %BINDIR%\rnnlm2_cls_based.exe --lstm --layers 2 --seed 127 --parameters %WORKDIR%\models\lstm.l2.h200.mdl --hidden 200 -t %DATADIR%\ptb.trn -d %DATADIR%\ptb.dev --word2cls %DATADIR%/wrd2cls.txt --cls2size %DATADIR%/cls2wrd.txt > %WORKDIR%\logs\train.lstm.l2.hd200.log 2>&1

call %BINDIR%\rnnlm2_cls_based.exe --lstm --layers 2 --seed 127 --initialise %WORKDIR%\models\lstm.l2.h200.mdl --hidden 200 -t %DATADIR%\ptb.trn --test %DATADIR%\ptb.tst  --word2cls %DATADIR%/wrd2cls.txt --cls2size %DATADIR%/cls2wrd.txt  > %WORKDIR%\logs\tst.lstm.l2.hd200.log 2>&1

REM run with the following argument in a local machine
REM  C:\dev\cnn\msbuild\examples\Release\rnnlm2.exe --lstm --layers 2 --seed 127 --parameters c:/temp/models/lstm.l2.h200.mdl --hidden 200 -t c:/data/ptbdata/ptb.trn -d c:/data/ptbdata/ptb.dev
REM cuda
REM C:\dev\cnn2\msbuild\examples\Release\rnnlm2.exe --verbose --lstm --layers 2 --seed 127 --parameters c:/temp/models/lstm.l2.h200.mdl --hidden 200 -t c:/data/ptbdata/ptb.trn.1 -d c:/data/ptbdata/ptb.trn.1
REM C:\dev\cnn2\msbuild\examples\Release\rnnlm2.exe --verbose --lstm --layers 2 --seed 127 --parameters c:/temp/models/lstm.l2.h200.mdl.2 --hidden 200 -t c:/data/ptbdata/ptb.trn.1 -d c:/data/ptbdata/ptb.trn.1

goto exit

***TEST E = 4.90867 ppl=135.46 [completed in 87039.5 ms]
***TEST E = 4.69185 ppl=109.055 [completed in 85790.6 ms]
***TEST E = 4.58014 ppl=97.5277 [completed in 87515.1 ms]
***TEST E = 4.51791 ppl=91.6438 [completed in 56845.2 ms]
***TEST E = 4.46853 ppl=87.2282 [completed in 85471.8 ms]
***TEST E = 4.44234 ppl=84.9733 [completed in 85454.2 ms]
***TEST E = 4.42582 ppl=83.581 [completed in 85565.1 ms]
***TEST E = 4.4018 ppl=81.5974 [completed in 86267.7 ms]
***TEST E = 4.38446 ppl=80.1952 [completed in 84486.2 ms]
***TEST E = 4.37629 ppl=79.5421 [completed in 85029.8 ms]
***TEST E = 4.37922 ppl=79.7756 [completed in 74974.4 ms]
***TEST E = 4.3683 ppl=78.9095 [completed in 83343.6 ms]
***TEST E = 4.35678 ppl=78.0054 [completed in 84922.3 ms]
***TEST E = 4.36182 ppl=78.3999 [completed in 75443.5 ms]
***TEST E = 4.27959 ppl=72.2111 [completed in 85596.8 ms]
***TEST E = 4.34951 ppl=77.4407 [completed in 31130.7 ms]
***TEST E = 4.34161 ppl=76.8308 [completed in 77652 ms]
***TEST E = 4.28546 ppl=72.6362 [completed in 79076.9 ms]
***TEST E = 4.34992 ppl=77.4719 [completed in 78543.6 ms]
***TEST E = 4.27918 ppl=72.1816 [completed in 89235.5 ms]
***TEST E = 4.33762 ppl=76.525 [completed in 84048.5 ms]

:exit
