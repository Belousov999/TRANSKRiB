# Benchmarks (файл 18:34)

| Hardware | Affinity / Threads | Model | Time | Notes |
|---|---:|---|---:|---|
| i7-8700 CPU | 2C / 4T @ ~4.2GHz | LargeV3 | 154 min | слишком долго, только “терпеливый” режим |
| i7-8700 CPU | 2C / 4T @ ~4.2GHz | LargeV3-turbo | 8:35 | качество turbo не устраивает |
| i7-8700 CPU | 1C / 2T @ ~3.2GHz | LargeV3-turbo | 20:07 | качество turbo не устраивает |
| RTX 3090 GPU | CUDA | LargeV3 | 164 sec | базовый качественный режим |
| RTX 3090 GPU | CUDA | LargeV3-turbo | 34 sec | быстрее, но качество ниже |

## Сценарий “офисный CPU”
Днём: 2 ядра/4 потока (чтобы не мешать работе).  
Ночью: расширять affinity до 4–6 ядер (ускорение, работе не мешает).
