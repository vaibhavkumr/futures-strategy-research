#!/bin/bash
cd /c/Users/vaibh/tjr-bot
for SYM in BTCUSDT ETHUSDT SOLUSDT; do
  for Y in 2024 2025; do
    for M in 01 02 03 04 05 06 07 08 09 10 11 12; do
      F="${SYM}-fundingRate-${Y}-${M}"
      [ -f "funding/$F.csv" ] && continue
      curl -sf "https://data.binance.vision/data/futures/um/monthly/fundingRate/${SYM}/${F}.zip" -o "/tmp/$F.zip" 2>/dev/null \
        && unzip -oq "/tmp/$F.zip" -d funding/ 2>/dev/null && rm -f "/tmp/$F.zip"
    done
  done
done
echo FUNDDONE
