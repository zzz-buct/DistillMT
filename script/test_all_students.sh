#!/usr/bin/env bash

set -e

# Replace this path with your own environment path
ROOT_DIR="$HOME/autodl-tmp/DistillMT-main"
SCRIPT_DIR="$ROOT_DIR/script"

cd "$SCRIPT_DIR"

python test.py --model_project activemq --target_project activemq --test_dataset activemq-5.2.0 --backbone HGT
python test.py --model_project activemq --target_project activemq --test_dataset activemq-5.3.0 --backbone HGT
python test.py --model_project activemq --target_project activemq --test_dataset activemq-5.8.0 --backbone HGT

python test.py --model_project camel --target_project camel --test_dataset camel-2.10.0 --backbone HGT
python test.py --model_project camel --target_project camel --test_dataset camel-2.11.0 --backbone HGT

python test.py --model_project derby --target_project derby --test_dataset derby-10.5.1.1 --backbone HGT

python test.py --model_project groovy --target_project groovy --test_dataset groovy-1_6_BETA_2 --backbone HGT

python test.py --model_project hbase --target_project hbase --test_dataset hbase-0.95.2 --backbone HGT

python test.py --model_project hive --target_project hive --test_dataset hive-0.12.0 --backbone HGT

python test.py --model_project jruby --target_project jruby --test_dataset jruby-1.5.0 --backbone HGT
python test.py --model_project jruby --target_project jruby --test_dataset jruby-1.7.0.preview1 --backbone HGT

python test.py --model_project lucene --target_project lucene --test_dataset lucene-3.0.0 --backbone HGT
python test.py --model_project lucene --target_project lucene --test_dataset lucene-3.1 --backbone HGT

python test.py --model_project wicket --target_project wicket --test_dataset wicket-1.5.3 --backbone HGT
