@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: Resolve repo root + venv python (robust, independent of CWD)
:: bat nam o <repo>\preprocess\download\  -> repo root = ..\..
:: ============================================================
pushd "%~dp0..\.."
set "REPO_ROOT=%CD%"
popd
set "PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
set "SCRIPT_DIR=%~dp0"
:: mimic_main.py dung fallback `from config import` (script dir) + `from preprocess.download...` (repo root)
set "PYTHONPATH=%REPO_ROOT%"

:: --- CAU HINH DUONG DAN ---
set "REMOTE_BASE=dhint:CHEX-DATA/Mimic-CXR/files"
set "REMOTE_PROCESSED=dhint:CHEX-DATA/mimic_processed"
set "LOCAL_BASE=C:\DONT SKIP CLASSES\HCMUT\RESEARCH\CHEX\CHEX-DATA\MIMIC-CXR_TEMP"
set "OUTPUT_DIR=C:\DONT SKIP CLASSES\HCMUT\RESEARCH\CHEX\MIMIC-CXR"
set "MIMIC_METADATA_CSV=%REPO_ROOT%\data\mimic-cxr-2.0.0-metadata.csv"
set "CHECKPOINTS_DIR=%OUTPUT_DIR%\checkpoints"
set "PROCESSED_PACKS_FILE=%CHECKPOINTS_DIR%\processed_packs.json"

:: --- RCLONE TUNING (HIGH THROUGHPUT) ---
set "RCLONE_TRANSFERS=12"
set "RCLONE_CHECKERS=32"
set "RCLONE_DRIVE_CHUNK_SIZE=256M"
set "RCLONE_BUFFER_SIZE=128M"
set "RCLONE_MULTI_THREAD_STREAMS=8"
set "RCLONE_MULTI_THREAD_CUTOFF=10M"
set "RCLONE_RETRIES=6"
set "RCLONE_LOW_LEVEL_RETRIES=20"
set "RCLONE_RETRIES_SLEEP=5s"
set "RCLONE_TIMEOUT=10m"
set "RCLONE_CONTIMEOUT=30s"
set "RCLONE_STATS=1s"
set "RCLONE_LOG_LEVEL=ERROR"
set "RCLONE_LOG_DIR=%OUTPUT_DIR%\logs"
set "BATCH_LOG=%RCLONE_LOG_DIR%\mimic_batch.log"

:: Common rclone flags (tranh lap lai)
set "RCLONE_FLAGS=--transfers %RCLONE_TRANSFERS% --checkers %RCLONE_CHECKERS% --drive-chunk-size %RCLONE_DRIVE_CHUNK_SIZE% --buffer-size %RCLONE_BUFFER_SIZE% --multi-thread-streams %RCLONE_MULTI_THREAD_STREAMS% --multi-thread-cutoff %RCLONE_MULTI_THREAD_CUTOFF% --contimeout %RCLONE_CONTIMEOUT% --timeout %RCLONE_TIMEOUT% --retries %RCLONE_RETRIES% --low-level-retries %RCLONE_LOW_LEVEL_RETRIES% --retries-sleep %RCLONE_RETRIES_SLEEP% --drive-skip-gdocs --fast-list --drive-pacer-min-sleep 10ms"

if not exist "%RCLONE_LOG_DIR%" (
    mkdir "%RCLONE_LOG_DIR%"
)

where rclone >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Khong tim thay rclone trong PATH.
    exit /b 1
)

if not exist "%PYTHON%" (
    echo [ERROR] Khong tim thay venv python: %PYTHON%
    echo         Tao venv bang: python -m venv "%REPO_ROOT%\.venv"
    exit /b 1
)

if not exist "%MIMIC_METADATA_CSV%" (
    echo [ERROR] Khong tim thay file metadata CSV: %MIMIC_METADATA_CSV%
    exit /b 1
)

:: Vong lap chay tu Pack 10 den 19
for /L %%i in (10,1,19) do (
    set "PACK=p%%i"

    echo.
    echo [%TIME%] KIEM TRA xem !PACK! da hoan thanh chua...

    :: check_pack_processed.py tra exit code 1 neu da xu ly, 0 neu chua (tranh for/f quoting voi path co dau cach)
    "%PYTHON%" "%SCRIPT_DIR%check_pack_processed.py" "%PROCESSED_PACKS_FILE%" "!PACK!" >nul 2>nul
    if errorlevel 1 (
        echo [%TIME%] SKIP !PACK! vi da hoan thanh trong lan chay truoc
        echo [%TIME%] [INFO] Skip !PACK! because it is already completed>>"%BATCH_LOG%"
    ) else (
        echo.
        echo [%TIME%] DANG TAI !PACK! bang rclone...
        rclone copy "%REMOTE_BASE%/!PACK!" "%LOCAL_BASE%/!PACK!" --create-empty-src-dirs --progress --stats %RCLONE_STATS% --stats-log-level NOTICE --log-level %RCLONE_LOG_LEVEL% --log-file "%RCLONE_LOG_DIR%\rclone_!PACK!.log" %RCLONE_FLAGS%
        if errorlevel 1 (
            echo [ERROR] Tai !PACK! that bai, bo qua pack nay.
            echo [%TIME%] [ERROR] Download failed for !PACK!>>"%BATCH_LOG%"
        ) else (
            echo [%TIME%] DANG XU LY !PACK! bang Python...
            echo [%TIME%] [INFO] Start preprocessing !PACK!>>"%BATCH_LOG%"

            :: Xu ly + xoa local RAW p-folder (--delete-p-folders) sau khi thanh cong
            "%PYTHON%" "%SCRIPT_DIR%mimic_main.py" --source-dir "%LOCAL_BASE%" --output-dir "%OUTPUT_DIR%" --metadata-csv "%MIMIC_METADATA_CSV%" --p-folders !PACK! --pack-name !PACK! --delete-p-folders
            if errorlevel 1 (
                echo [ERROR] Preprocess !PACK! that bai. Khong upload, khong xoa local.
                echo [%TIME%] [ERROR] Preprocess failed for !PACK!>>"%BATCH_LOG%"
            ) else (
                echo [%TIME%] DA XU LY XONG !PACK!. DANG UPLOAD anh da xu ly len Drive...
                echo [%TIME%] [INFO] Finished preprocessing !PACK!, start upload>>"%BATCH_LOG%"

                :: Upload anh da xu ly: OUTPUT_DIR\images\pXX -> REMOTE_PROCESSED/pXX
                rclone copy "%OUTPUT_DIR%\images\!PACK!" "%REMOTE_PROCESSED%/!PACK!" --progress --stats %RCLONE_STATS% --stats-log-level NOTICE --log-level %RCLONE_LOG_LEVEL% --log-file "%RCLONE_LOG_DIR%\upload_!PACK!.log" %RCLONE_FLAGS%
                if errorlevel 1 (
                    echo [ERROR] Upload !PACK! that bai. GIU LAI local de thu lai sau.
                    echo [%TIME%] [ERROR] Upload failed for !PACK!, keep local>>"%BATCH_LOG%"
                ) else (
                    echo [%TIME%] UPLOAD XONG !PACK!. Xoa anh local da xu ly...
                    echo [%TIME%] [INFO] Uploaded !PACK!, deleting local processed images>>"%BATCH_LOG%"
                    rmdir /s /q "%OUTPUT_DIR%\images\!PACK!"
                    echo [%TIME%] [INFO] Deleted local processed images for !PACK!>>"%BATCH_LOG%"
                )
            )
        )
    )
)

echo.
echo ============================================================
echo [DONE] TAT CA CAC GOI DA DUOC TAI, XU LY VA UPLOAD XONG!
echo ============================================================
pause
