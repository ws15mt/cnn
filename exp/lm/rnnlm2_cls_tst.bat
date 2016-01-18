
REM this run recurrent network language model experiments
REM baseline is class-based LSTM
REM words are clustered in frequency. Each class has equal probability mass. 
REM 

set WORKDIR=c:\dev\mycnn\exp\lm
set DATADIR=c:/data/ptbdata
set BINDIR=C:\dev\mycnn\msbuildx64\examples\Release

mkdir %WORKDIR%\logs
mkdir %WORKDIR%\models

REM call %BINDIR%\rnnlm2_cls_based.exe --lstm --layers 2 --seed 127 --parameters %WORKDIR%\models\lstm.l2.h200.mdl --hidden 200 -t %DATADIR%\ptb.trn -d %DATADIR%\ptb.dev --word2cls %DATADIR%/wrd2cls.txt --cls2size %DATADIR%/cls2wrd.txt > %WORKDIR%\logs\train.lstm.l2.hd200.log 2>&1

call %BINDIR%\rnnlm2_cls_based.exe --lstm --layers 2 --seed 127 --initialise %WORKDIR%\models\lstm.l2.h200.mdl --hidden 200 -t %DATADIR%\ptb.trn --test %DATADIR%\ptb.tst  --word2cls %DATADIR%/wrd2cls.txt --cls2size %DATADIR%/cls2wrd.txt  > %WORKDIR%\logs\tst.lstm.l2.hd200.log 2>&1


REM ***DEV [epoch=0] E = 4.32119 ppl=75.2784

goto exit

mkdir d:\dev\mycnn\exp\lm\

mkdir d:\dev\mycnn\exp\lm\models

mkdir d:\dev\mycnn\exp\lm\logs

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --lstm --layers 2 --seed 127 --parameters d:\dev\mycnn\exp\lm\models\lstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn -d d:\data\ptbdata\ptb.dev  > d:\dev\mycnn\exp\lm\logs\train.lstm.l2.hd200.log 2>&1

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --dglstm --layers 2 --seed 127 --parameters d:\dev\mycnn\exp\lm\models\dglstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn -d d:\data\ptbdata\ptb.dev  > d:\dev\mycnn\exp\lm\logs\train.lstm.l2.hd200.log 2>&1

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --lstm --layers 2 --seed 127 --initialise d:\dev\mycnn\exp\lm\models\lstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn --test d:\data\ptbdata\ptb.tst  > d:\dev\mycnn\exp\lm\logs\tst.lstm.l2.hd200.log 2>&1

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --dglstm --layers 2 --seed 127 --initialise d:\dev\mycnn\exp\lm\models\dglstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn --test d:\data\ptbdata\ptb.tst  > d:\dev\mycnn\exp\lm\logs\tst.lstm.l2.hd200.log 2>&1


:exit
