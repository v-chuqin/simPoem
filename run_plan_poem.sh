#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=168:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q GpuQ

set -e
set -x
export THEANO_FLAGS=device=gpu,floatX=float32#,optimizer=None

to_doc_script="./results_to_doc.py"
test_data_dir="../data/picked"
doc_dir="../data/docs/"
result_dir="../results/"

saveto="plan_poem.npz"
test_target="test_plan_poem"
model='nmt_plan_poem'

mode=$1
if [ "$mode"x = "x" ]; then
  mode='train'
fi

reload_iter=$2
if [ "$reload_iter"x = "x" ]; then
  reload_iter=-1
fi

if [ "$mode"x = "trainx" ]; then
  python run_nmt.py $model train \
    --train_source train_prev_line_seq \
    --train_target train_target_seq \
    --train_keyword train_keyword_seq \
    --val_source val_prev_line_seq \
    --val_target val_target_seq \
    --val_keyword val_keyword_seq \
    --argmax --saveto $saveto
else
  reload_iter=$2
  if [ "$reload_iter"x = "x" ]; then
    reload_iter=-1
  fi
  temp=$3
  if [ "$temp"x != "x" ]; then
    test_target=$temp
  fi
  if [ $reload_iter -ne -1 ]; then
    test_target=$test_target"-"$reload_iter
  fi
  python run_nmt.py $model test \
    --test_source test_imageids_with_keywords \
    --argmax --saveto $saveto --test_target $test_target \
    --reload --reload_iter $reload_iter

  $to_doc_script $test_data_dir $result_dir/$test_target \
    $doc_dir/$test_target".doc"
fi
