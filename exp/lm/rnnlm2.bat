
REM this run recurrent network language model experiments
REM baseline is LSTM
REM alternatively can run Depth-gated LSTM
REM 

set WORKDIR=.
set DATADIR=c:\data\ptbdata
set BINDIR=C:\dev\cnn\msbuildcuda\examples\Release
set BINDIR=C:\dev\cnn\msbuild\examples\Release

mkdir %WORKDIR%\logs
mkdir %WORKDIR%\models


call %BINDIR%\rnnlm2.exe --lstm --layers 2 --seed 127 --parameters %WORKDIR%\models\lstm.l2.h200.mdl --hidden 200 -t %DATADIR%\ptb.trn -d %DATADIR%\ptb.dev  > %WORKDIR%\logs\train.lstm.l2.hd200.log 2>&1
REM call %BINDIR%\rnnlm2.exe --checkgrad true --lstm --layers 2 --seed 127 --parameters %WORKDIR%\models\lstm.l2.h200.mdl --hidden 200 -t %DATADIR%\ptb.trn -d %DATADIR%\ptb.dev  > %WORKDIR%\logs\train.lstm.l2.hd200.log 2>&1


call %BINDIR%\rnnlm2.exe --lstm --layers 2 --seed 127 --initialise %WORKDIR%\models\lstm.l2.h200.mdl --hidden 200 -t %DATADIR%\ptb.trn --test %DATADIR%\ptb.tst  > %WORKDIR%\logs\tst.lstm.l2.hd200.log 2>&1

REM run with the following argument in a local machine
REM  C:\dev\cnn\msbuild\examples\Release\rnnlm2.exe --lstm --layers 2 --seed 127 --parameters c:/temp/models/lstm.l2.h200.mdl --hidden 200 -t c:/data/ptbdata/ptb.trn -d c:/data/ptbdata/ptb.dev
REM cuda
REM C:\dev\cnn2\msbuild\examples\Release\rnnlm2.exe --verbose --lstm --layers 2 --seed 127 --parameters c:/temp/models/lstm.l2.h200.mdl --hidden 200 -t c:/data/ptbdata/ptb.trn.1 -d c:/data/ptbdata/ptb.trn.1
REM C:\dev\cnn2\msbuild\examples\Release\rnnlm2.exe --verbose --lstm --layers 2 --seed 127 --parameters c:/temp/models/lstm.l2.h200.mdl.2 --hidden 200 -t c:/data/ptbdata/ptb.trn.1 -d c:/data/ptbdata/ptb.trn.1

goto exit

mkdir d:\dev\mycnn\exp\lm\

mkdir d:\dev\mycnn\exp\lm\models

mkdir d:\dev\mycnn\exp\lm\logs

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --lstm --layers 2 --seed 127 --parameters d:\dev\mycnn\exp\lm\models\lstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn -d d:\data\ptbdata\ptb.dev  > d:\dev\mycnn\exp\lm\logs\train.lstm.l2.hd200.log 2>&1

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --dglstm --layers 2 --seed 127 --parameters d:\dev\mycnn\exp\lm\models\dglstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn -d d:\data\ptbdata\ptb.dev  > d:\dev\mycnn\exp\lm\logs\train.lstm.l2.hd200.log 2>&1

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --lstm --layers 2 --seed 127 --initialise d:\dev\mycnn\exp\lm\models\lstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn --test d:\data\ptbdata\ptb.tst  > d:\dev\mycnn\exp\lm\logs\tst.lstm.l2.hd200.log 2>&1

D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --dglstm --layers 2 --seed 127 --initialise d:\dev\mycnn\exp\lm\models\dglstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn --test d:\data\ptbdata\ptb.tst  > d:\dev\mycnn\exp\lm\logs\tst.lstm.l2.hd200.log 2>&1


:exit
