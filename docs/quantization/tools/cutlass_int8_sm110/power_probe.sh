#!/bin/bash
# FP8 vs INT8 tensor-core throttling probe under the current nvpmodel power mode. No sudo needed to run; to compare modes run
#   sudo nvpmodel -m 0            # MAXN (mode 1 = 120W default). Optionally: sudo jetson_clocks  (locks max clocks)
#   ./power_probe.sh maxn         # -> results/power_probe_maxn.md
#   sudo nvpmodel -m 1            # restore
# Samples the GPU clock (devfreq gpu-gpc-0), EMC clock (bwmgr) and Tj every 50 ms while each kernel runs. If a tegrastats log is
# given as $2 (e.g. started with: sudo tegrastats --interval 200 --logfile /tmp/tegrastats.log), its VDD_GPU field is summarised.
cd "$(dirname "$0")"
TAG=${1:-$(nvpmodel -q 2>/dev/null | sed -n 's/NV Power Mode: //p' | tr -d ' ' | tr 'A-Z' 'a-z')}
OUT=results/power_probe_${TAG:-unknown}.md; SCR=$(mktemp -d); n=0
maxf=$(( $(cat /sys/class/devfreq/gpu-gpc-0/max_freq) / 1000000 ))
{
echo "# FP8 vs INT8 probe, power mode: $(nvpmodel -q 2>/dev/null | sed -n 's/NV Power Mode: //p'), GPU max_freq ${maxf} MHz, $(date '+%Y-%m-%d %H:%M')"
echo; echo "cfg3 (2SM 256x256x128 cluster2x1, swizzle auto) for both dtypes. warm = --flush=0 --nw=1 (L2-resident), hotcold = --flush=0 --nw=8 (activations hot, 8 rotating weights), cold = --flush=1. normal = per-tensor-quantized Gaussian operands."
echo; echo "| case | dtype | TFLOPS | median us | gpu clk med/min MHz | emc clk med/min MHz | Tj end C | GPU power mean/max W (tegrastats VDD_GPU) | TOPS per W |"; echo "|---|---|---:|---:|---:|---:|---:|---:|---:|"
} > $OUT
LOG="$2"
gpu_power() {  # mean/max VDD_GPU (mW) over log lines [$1, $2]
  [ -n "$LOG" ] && [ -f "$LOG" ] && [ "$2" -gt "$1" ] && sed -n "$(( $1 + 1 )),$2p" "$LOG" | grep -o "VDD_GPU [0-9]*mW" | awk '{gsub("mW","",$2); v=$2+0; s+=v; if(v>m)m=v; n++} END {if(n>0) printf "%.1f/%.1f", s/n/1000, m/1000; else print "n/a"}' || echo "n/a"; }
sample() { label="$1"; dt="$2"; shift 2; n=$((n+1)); o="$SCR/$n.out"; l0=$([ -f "$LOG" ] && wc -l < "$LOG" || echo 0); ./pertensor_gemm --dtype=$dt "$@" > "$o" 2>/dev/null & pid=$!; g=(); e=()
  while kill -0 $pid 2>/dev/null; do g+=($(( $(cat /sys/class/devfreq/gpu-gpc-0/cur_freq) / 1000000 ))); e+=($(( $(cat /sys/class/devfreq/bwmgr/cur_freq 2>/dev/null || echo 0) / 1000000 ))); sleep 0.05; done
  gm=$(printf '%s\n' "${g[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)+1]"/"a[1]}'); em=$(printf '%s\n' "${e[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)+1]"/"a[1]}')
  tj=$(cat /sys/class/thermal/thermal_zone0/temp); tf=$(grep -o "tflops=[0-9.]*" "$o" | cut -d= -f2); us=$(grep -o "median_us=[0-9.]*" "$o" | cut -d= -f2)
  l1=$([ -f "$LOG" ] && wc -l < "$LOG" || echo 0); pw=$(gpu_power $l0 $l1); eff=$(echo "$pw $tf" | awk '{split($1,a,"/"); if(a[1]>0 && $2>0) printf "%.1f", $2/a[1]; else print "n/a"}')
  line="| $label | $dt | ${tf:-ERR} | ${us:-ERR} | $gm | $em | $((tj/1000)).$(( (tj%1000)/100 )) | $pw | $eff |"; echo "$line" >> $OUT; echo "$line"; }
W="--cfg=3 --m=4096 --n=4096 --k=4096 --flush=0 --nw=1 --verify=0 --iters=3000"
for dt in fp8 int8; do sample "warm 4096^3 normal 3000it" $dt $W --dist=normal; done
for dt in fp8 int8; do sample "warm 4096^3 zeros 3000it" $dt $W --zeros=1; done
for dt in fp8 int8; do sample "warm 4096^3 normal 3000it (repeat)" $dt $W --dist=normal; done
for M in 901 1517 1802; do for dt in fp8 int8; do sample "hotcold 4096x4096 M=$M normal 200it" $dt --cfg=3 --m=$M --n=4096 --k=4096 --flush=0 --nw=8 --verify=0 --iters=200 --dist=normal; done; done
for M in 901 1517; do for dt in fp8 int8; do sample "hotcold 1024x4096 M=$M normal 200it" $dt --cfg=3 --m=$M --n=1024 --k=4096 --flush=0 --nw=8 --verify=0 --iters=200 --dist=normal; done; done
for dt in fp8 int8; do sample "cold 4096x4096 M=1517 normal 100it" $dt --cfg=3 --m=1517 --n=4096 --k=4096 --flush=1 --verify=0 --iters=100 --dist=normal; done
if [ -n "$LOG" ] && [ -f "$LOG" ]; then echo >> $OUT; echo "GPU power source: tegrastats VDD_GPU from $LOG (root tegrastats; the non-root reading is 0 mW). Idle before probe: $(head -3 "$LOG" | grep -o 'VDD_GPU [0-9]*mW' | tail -1)." >> $OUT; fi
echo "wrote $OUT"; rm -rf "$SCR"
