@echo off
md build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
cmake --build . --config Release --target ALL_BUILD
cd ..
powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://github.com/springkim/MMODCNN_SpringEdition/releases/download/data/opencv_world400.dll','build\Release\opencv_world400.dll')"
pause