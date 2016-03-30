REM use small set with 10k training  
REM use rmspropwithmomentum
set MAXEPOCH=40
set EACHEPOCH=1
set HDIM=%3
set ADIM=%HDIM%
REM set IDIM=50
set IDIM=%2
set NLAYER=2
set STG=s41
set RNN=gru
set ETA=%1
set SCALE=0.1
set CLIP=5.0
set TRAINER=rmspropwithmomentum
set NUTT=10
set CMDT=AWI_InputFeedingWithNNAttention
set TAG=%STG%.%CMDT%.nosplitdialogue.%RNN%.e%ETA%.s%SCALE%.%TRAINER%.h%HDIM%.i%IDIM%.l%NLAYER%.clip%CLIP%.bsize%NUTT%.sz10k

REM call //gcr/scratch/b99/kaisheny/exp/Dialogue/setenv.s03.consattentional.bat
call //speechstore5/transient/kaishengy/exp/Dialogue/setenv.s03.consattentional.bat

echo %BINPATH%\attention_with_intention.nn.exe --nparallel %NUTT% --nosplitdialogue --clip %CLIP% --%CMDT% --trainer %TRAINER% --scale %SCALE% --eta %ETA% --%RNN% --layers %NLAYER% --seed 127 --hidden %HDIM% --align %ADIM% --intentiondim %IDIM% --epochs %MAXEPOCH% --readdict %WORKPATH%/lib/all.dict --parameters %WORKPATH%/models/%TAG%.mdl -t %DATAPATH%/mswindowshelp.trn.dat.unk.10000 -d %DATAPATH%/mswindowshelp.dev.dat.unk.2000  > %WORKPATH%/logs/train.%TAG%.log 2>&1
%BINPATH%\attention_with_intention.nn.exe --nparallel %NUTT% --nosplitdialogue --clip %CLIP% --%CMDT% --trainer %TRAINER% --scale %SCALE% --eta %ETA% --%RNN% --layers %NLAYER% --seed 127 --hidden %HDIM% --align %ADIM% --intentiondim %IDIM% --epochs %MAXEPOCH% --readdict %WORKPATH%/lib/all.dict --parameters %WORKPATH%/models/%TAG%.mdl -t %DATAPATH%/mswindowshelp.trn.dat.unk.10000 -d %DATAPATH%/mswindowshelp.dev.dat.unk.2000  >> %WORKPATH%/logs/train.%TAG%.log 2>>&1

REM D:\dev\quartzin\exp\dialogue>D:\dev\mycnn\msbuild\conversation\dev\src\Release\attention_with_intention.exe --verbose --nparallel 10 --nosplitdialogue --clip 5.0 --AWI_InputFeedingWDropout --trainer rmspropwithmomentum --scale 0.1 --eta 0.1 --gru --layers 2 --seed 127 --hidden 20 --align 20 --intentiondim 5 --epochs 40 --readdict \\speechstore5\transient\kaishengy\exp\dialogue/lib/all.dict --parameters d:\dev\quartzin\exp\dialogue\models\s33_s02.AWI_LocalGeneralInputFeeding.nosplitdialogue.gru.e50.0.s0.1.rmspropwithmomentum.h200.i50.l2.clip5.0.bsize10.sz10k.mdl -t //speechstore5/transient/kaishengy/data/DialogueSamelanguage/mswindowshelp.trn.dat.unk.dbg -d //speechstore5/transient/kaishengy/data/DialogueSamelanguage/mswindowshelp.dev.dat.unk.dbg


goto exit

:test
echo %BINPATH%\attention_with_intention.nn.exe --nparallel %NUTT% --nosplitdialogue --clip %CLIP% --%CMDT% --trainer %TRAINER% --scale %SCALE% --eta %ETA% --%RNN% --layers %NLAYER% --seed 127 --hidden %HDIM% --align %ADIM% --intentiondim %IDIM% --epochs %MAXEPOCH% --readdict %WORKPATH%/lib/all.dict --initialise %WORKPATH%/models/%TAG%.mdl --testcorpus %DATAPATH%/mswindowshelp.tst.dat.unk.2000 --outputfile %WORKPATH%/logs/test.%TAG%.results > %WORKPATH%/logs/test.%TAG%.log 2>&1
%BINPATH%\attention_with_intention.nn.exe --nparallel %NUTT% --nosplitdialogue --clip %CLIP% --%CMDT% --trainer %TRAINER% --scale %SCALE% --eta %ETA% --%RNN% --layers %NLAYER% --seed 127 --hidden %HDIM% --align %ADIM% --intentiondim %IDIM% --epochs %MAXEPOCH% --readdict %WORKPATH%/lib/all.dict --initialise %WORKPATH%/models/%TAG%.mdl --testcorpus %DATAPATH%/mswindowshelp.tst.dat.unk.2000 --outputfile %WORKPATH%/logs/test.%TAG%.results >> %WORKPATH%/logs/test.%TAG%.log 2>>&1

REM --verbose --nparallel 10 --nosplitdialogue --clip 5.0 --AWI_GeneralInputFeeding --trainer rmspropwithmomentum --scale 0.1 --eta 50.0 --gru --layers 2 --seed 127 --hidden 200 --align 200 --intentiondim 50 --epochs 40 --readdict \\speechstore5\transient\kaishengy\exp\dialogue/lib/all.dict --initialise \\speechstore5\transient\kaishengy\exp\dialogue/models/s38_s02.AWI_GeneralInputFeeding.nosplitdialogue.gru.e50.0.s0.1.rmspropwithmomentum.h200.i50.l2.clip5.0.bsize10.sz10k.mdl --testcorpus //speechstore5/transient/kaishengy/data/DialogueSamelanguage/mswindowshelp.trn.dat.unk.1 --outputfile \\speechstore5\transient\kaishengy\exp\dialogue/logs/xxx.log

goto exit

:exit
