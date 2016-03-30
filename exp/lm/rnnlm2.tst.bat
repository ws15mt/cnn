REM this run recurrent network language model experiments


REM baseline is LSTM
REM alternatively can run Depth-gated LSTM
REM


mkdir d:\dev\mycnn\exp\lm\models

mkdir d:\dev\mycnn\exp\lm\logs
 


D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --lstm --layers 2 --seed 127 --initialise d:\dev\mycnn\exp\lm\models\lstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn --test d:\data\ptbdata\ptb.tst  > d:\dev\mycnn\exp\lm\logs\tst.lstm.l2.hd200.log 2>&1




D:\dev\mycnn\msbuildcuda\examples\Release\rnnlm2.exe --dglstm --layers 2 --seed 127 --initialise d:\dev\mycnn\exp\lm\models\dglstm.l2.h200.mdl --hidden 200 -t d:\data\ptbdata\ptb.trn --test d:\data\ptbdata\ptb.tst  > d:\dev\mycnn\exp\lm\logs\tst.lstm.l2.hd200.log 2>&1



