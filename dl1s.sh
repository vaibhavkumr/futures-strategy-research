#!/bin/bash
cd /c/Users/vaibh/tjr-bot
for SYM in BTCUSDT ETHUSDT; do
  for M in 2025-04 2025-05 2025-06; do
    F="${SYM}-1s-${M}"
    [ -f "binance1s/$F.csv" ] && continue
    curl -sf "https://data.binance.vision/data/spot/monthly/klines/${SYM}/1s/${F}.zip" -o "/tmp/$F.zip" \
      && unzip -oq "/tmp/$F.zip" -d binance1s/ && rm -f "/tmp/$F.zip" && echo "got $F"
  done
done
echo "DONE1S"
