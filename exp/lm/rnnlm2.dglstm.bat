
REM this run recurrent network language model experiments
REM baseline is LSTM
REM alternatively can run Depth-gated LSTM
REM 

set WORKDIR=\\gcr\scratch\b99\kaisheny\exp\lm
set DATADIR=\\gcr\scratch\b99\kaisheny\data\ptddata
set BINDIR=\\gcr\scratch\b99\kaisheny\bin\windows


mkdir %WORKDIR%\logs
mkdir %WORKDIR%\models


call %BINDIR%\rnnlm2.exe --dglstm --layers 2 --seed 127 --parameters %WORKDIR%\models\dglstm.l2.h200.mdl --hidden 200 -t %DATADIR%\ptb.trn -d %DATADIR%\ptb.dev  > %WORKDIR%\logs\train.dglstm.l2.hd200.log 2>&1

call %BINDIR%\rnnlm2.exe --dglstm --layers 2 --seed 127 --initialise %WORKDIR%\models\dglstm.l2.h200.mdl --hidden 200 -t %DATADIR%\ptb.trn --test %DATADIR%\ptb.tst  > %WORKDIR%\logs\tst.dglstm.l2.hd200.log 2>&1


goto exit

mkdir d:\dev\mycnn\exp\lm\

mkdir d:\dev\mycnn\exp\lm\models

mkdir d:\dev\mycnn\exp\lm\logs

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --lstm --layers 2 --seed 127 --parameters d:\dev\mycnn\exp\lm\models\lstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn -d d:\data\ptbdata\ptb.dev  > d:\dev\mycnn\exp\lm\logs\train.lstm.l2.hd200.log 2>&1

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --dglstm --layers 2 --seed 127 --parameters d:\dev\mycnn\exp\lm\models\dglstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn -d d:\data\ptbdata\ptb.dev  > d:\dev\mycnn\exp\lm\logs\train.lstm.l2.hd200.log 2>&1

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --lstm --layers 2 --seed 127 --initialise d:\dev\mycnn\exp\lm\models\lstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn --test d:\data\ptbdata\ptb.tst  > d:\dev\mycnn\exp\lm\logs\tst.lstm.l2.hd200.log 2>&1

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --dglstm --layers 2 --seed 127 --initialise d:\dev\mycnn\exp\lm\models\dglstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn --test d:\data\ptbdata\ptb.tst  > d:\dev\mycnn\exp\lm\logs\tst.lstm.l2.hd200.log 2>&1


:exit
