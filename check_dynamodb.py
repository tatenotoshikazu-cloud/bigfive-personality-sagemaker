# -*- coding: utf-8 -*-
"""
DynamoDBテーブル構造確認スクリプト
"""
import boto3
import json

# DynamoDB接続
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1')

# テーブルチェック
tables_to_check = ['recording-poc-records', 'recording-poc-meta', 'recording-poc-students']

for table_name in tables_to_check:
    print(f"\n{'='*60}")
    print(f"Table: {table_name}")
    print('='*60)

    table = dynamodb.Table(table_name)

    # 件数確認
    response = table.scan(Limit=1)
    print(f"Sample count: {response.get('Count', 0)}")
    print(f"ScannedCount: {response.get('ScannedCount', 0)}")

    # サンプルデータ表示
    if response['Items']:
        print("\nSample item:")
        sample = response['Items'][0]
        print(json.dumps(sample, indent=2, ensure_ascii=False, default=str))

        # キー確認
        print("\nAvailable keys:")
        for key in sample.keys():
            value = sample[key]
            value_type = type(value).__name__
            value_preview = str(value)[:100] if isinstance(value, str) else str(value)
            print(f"  - {key} ({value_type}): {value_preview}")
