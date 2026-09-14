#!/bin/bash
# Kernel-dominated GPU power per case: each case runs >= 3 s of back-to-back kernels so the tegrastats VDD_GPU mean over the process
# lifetime is the kernel's power. Needs a root tegrastats log:  sudo tegrastats --interval 200 --logfile /tmp/tegrastats.log &
#   ./power_probe_long.sh <tag> /tmp/tegrastats.log     -> results/power_probe_<tag>_long.md
cd "$(dirname "$0")"
TAG=${1:-$(nvpmodel -q 2>/dev/null | sed -n 's/NV Power Mode: //p' | tr -d ' ' | tr 'A-Z' 'a-z')}; LOG=$2; OUT=results/power_probe_${TAG}_long.md
[ -f "$LOG" ] || { echo "need a tegrastats log as \$2"; exit 1; }
gpu_power() { sed -n "$(( $1 + 1 )),$2p" "$LOG" | grep -o "VDD_GPU [0-9]*mW" | awk '{gsub("mW","",$2); v=$2+0; s+=v; if(v>m)m=v; n++} END {if(n>0) printf "%.1f %.1f %d", s/n/1000, m/1000, n; else print "0 0 0"}'; }
{ echo "# Power mode: $(nvpmodel -q 2>/dev/null | sed -n 's/NV Power Mode: //p'), GPU max_freq $(( $(cat /sys/class/devfreq/gpu-gpc-0/max_freq)/1000000 )) MHz, $(date '+%Y-%m-%d %H:%M'). >= 3 s of back-to-back kernels per case; GPU power = tegrastats VDD_GPU mean over the process (kernel-dominated). Energy per GEMM = mean W x median us."
  echo; echo "| case (Gaussian operands) | dtype | TFLOPS | median us | GPU W mean | GPU W max | samples | mJ per GEMM | TOPS/W |"; echo "|---|---|---:|---:|---:|---:|---:|---:|---:|"; } > $OUT
sample() { label="$1"; bin="$2"; dt="$3"; shift 3; l0=$(wc -l < "$LOG"); o=$(mktemp); ./$bin --dtype=$dt "$@" > $o 2>/dev/null; l1=$(wc -l < "$LOG"); read mean max ns <<< "$(gpu_power $l0 $l1)"
  tf=$(grep -o "tflops=[0-9.]*" $o | head -1 | cut -d= -f2); us=$(grep -o "median_us=[0-9.]*" $o | head -1 | cut -d= -f2)
  awk -v l="$label" -v d=$dt -v tf=${tf:-0} -v us=${us:-0} -v m=$mean -v x=$max -v n=$ns 'BEGIN {printf "| %s | %s | %.1f | %.1f | %.1f | %.1f | %d | %.2f | %.2f |\n", l, d, tf, us, m, x, n, m*us/1000, (m>0?tf/m:0)}' | tee -a $OUT; rm -f $o; }
C="--cfg=3 --verify=0 --dist=normal"
for dt in fp8 int8; do sample "hotcold 4096x4096 M=901"  pertensor_gemm $dt $C --m=901  --n=4096 --k=4096 --flush=0 --nw=8 --iters=30000; done
for dt in fp8 int8; do sample "hotcold 4096x4096 M=1517" pertensor_gemm $dt $C --m=1517 --n=4096 --k=4096 --flush=0 --nw=8 --iters=20000; done
for dt in fp8 int8; do sample "hotcold 4096x4096 M=1802" pertensor_gemm $dt $C --m=1802 --n=4096 --k=4096 --flush=0 --nw=8 --iters=15000; done
for dt in fp8 int8; do sample "hotcold 1024x4096 M=901"  pertensor_gemm $dt $C --m=901  --n=1024 --k=4096 --flush=0 --nw=8 --iters=60000; done
for dt in fp8 int8; do sample "hotcold 1024x4096 M=1802" pertensor_gemm $dt $C --m=1802 --n=1024 --k=4096 --flush=0 --nw=8 --iters=40000; done
for dt in fp8 int8; do sample "cold(flush) 4096x4096 M=901" pertensor_gemm $dt $C --m=901 --n=4096 --k=4096 --flush=1 --iters=3000; done
for dt in fp8 int8; do sample "warm 4096^3" pertensor_gemm $dt $C --m=4096 --n=4096 --k=4096 --flush=0 --nw=1 --iters=6000; done
for dt in fp8 int8; do sample "warm 4096^3 zeros" pertensor_gemm $dt --cfg=3 --verify=0 --zeros=1 --m=4096 --n=4096 --k=4096 --flush=0 --nw=1 --iters=6000; done
sample "hotcold 4096x4096 M=1517 (cuBLASLt)" cublaslt_bench bf16 --m=1517 --n=4096 --k=4096 --flush=0 --nw=8 --verify=0 --iters=15000
sample "hotcold 4096x4096 M=1517 (cuBLASLt)" cublaslt_bench fp8  --m=1517 --n=4096 --k=4096 --flush=0 --nw=8 --verify=0 --iters=15000
echo "wrote $OUT"
