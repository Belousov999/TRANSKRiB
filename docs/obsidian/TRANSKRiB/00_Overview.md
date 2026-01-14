# TRANSKRiB — Overview

## Что делает пайплайн
INPUT (сырьё) → CONVERTOR (wav mono 16k) → TTGP (GPU transcribe) → !RESULT (txt) + WavARH (архив wav)

## Быстрый старт
- Запуск: D:\TRANSKRiB\RUN\START_TURBO.bat (или другие модели)
- Стоп:   D:\TRANSKRiB\RUN\STOP_ALL.bat

## Где логи
- Конвертор: D:\TRANSKRiB\CONVERTOR\ConvLog
- TTGP:      D:\TRANSKRiB\TTGP\TTGPlog

## Assets
- [[assets/TRANSKRiB_Guide_RU_EN.md]]
- [[assets/prompts_ru.txt]]
- [[assets/prompts_en.txt]]
- [[assets/commands_setup.ps1]]
- [[assets/requirements_ttgp_py311_cu118.txt]]
- [[assets/py311_cuda_memo_ru.txt]]