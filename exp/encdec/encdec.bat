
REM this run encoder decoder experiemnts 

set WORKDIR=.
set DATADIR=c:\data\ptbdata
set BINDIR=C:\dev\cnn2\msbuild\examples\Release

mkdir %WORKDIR%\logs
mkdir %WORKDIR%\models

goto train

:gradientcheck
REM check gradient
REM to check gradient, need to use double precision
call %BINDIR%\encdec.exe %DATADIR%\ptb.trn %DATADIR%\ptb.dev 1 > %WORKDIR%\logs\encdec.log 2>&1

goto exit

:train

REM not checking gradient
call %BINDIR%\encdec.exe %DATADIR%\ptb.trn %DATADIR%\ptb.dev 0 > %WORKDIR%\logs\encdec.log 2>&1

goto exit

:exit
