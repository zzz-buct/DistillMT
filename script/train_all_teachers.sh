#!/usr/bin/env bash

set -e

# Replace this path with your own environment path
ROOT_DIR="$HOME/autodl-tmp/DistillMT-main"
SCRIPT_DIR="$ROOT_DIR/script"

cd "$SCRIPT_DIR"

python main.py --project activemq --train_dataset activemq-5.0.0 --valid_dataset activemq-5.1.0 --test_dataset activemq-5.2.0 --train_mode T --device 0 --seed 1 --nhid 256 --nlayers 4 --lr 0.01 --backbone HGT --early_stop_metric val_loss --batch_size 128

python main.py --project camel --train_dataset camel-1.4.0 --valid_dataset camel-2.9.0 --test_dataset camel-2.10.0 --train_mode T --device 0 --seed 1 --nhid 256 --nlayers 4 --lr 0.01 --backbone HGT --early_stop_metric val_loss --batch_size 128

python main.py --project derby --train_dataset derby-10.2.1.6 --valid_dataset derby-10.3.1.4 --test_dataset derby-10.5.1.1 --train_mode T --device 0 --seed 1 --nhid 256 --nlayers 4 --lr 0.01 --backbone HGT --early_stop_metric val_loss --batch_size 128

python main.py --project groovy --train_dataset groovy-1_5_7 --valid_dataset groovy-1_6_BETA_1 --test_dataset groovy-1_6_BETA_2 --train_mode T --device 0 --seed 1 --nhid 256 --nlayers 4 --lr 0.01 --backbone HGT --early_stop_metric val_loss --batch_size 128

python main.py --project hbase --train_dataset hbase-0.94.0 --valid_dataset hbase-0.95.0 --test_dataset hbase-0.95.2 --train_mode T --device 0 --seed 1 --nhid 256 --nlayers 4 --lr 0.01 --backbone HGT --early_stop_metric val_loss --batch_size 128

python main.py --project hive --train_dataset hive-0.9.0 --valid_dataset hive-0.10.0 --test_dataset hive-0.12.0 --train_mode T --device 0 --seed 1 --nhid 256 --nlayers 4 --lr 0.01 --backbone HGT --early_stop_metric val_loss --batch_size 128

python main.py --project jruby --train_dataset jruby-1.1 --valid_dataset jruby-1.4.0 --test_dataset jruby-1.5.0 --train_mode T --device 0 --seed 1 --nhid 256 --nlayers 4 --lr 0.01 --backbone HGT --early_stop_metric val_loss --batch_size 128

python main.py --project lucene --train_dataset lucene-2.3.0 --valid_dataset lucene-2.9.0 --test_dataset lucene-3.0.0 --train_mode T --device 0 --seed 1 --nhid 256 --nlayers 4 --lr 0.01 --backbone HGT --early_stop_metric val_loss --batch_size 128

python main.py --project wicket --train_dataset wicket-1.3.0-incubating-beta-1 --valid_dataset wicket-1.3.0-beta2 --test_dataset wicket-1.5.3 --train_mode T --device 0 --seed 1 --nhid 256 --nlayers 4 --lr 0.01 --backbone HGT --early_stop_metric val_loss --batch_size 128
