# -*- coding: utf-8 -*-
"""
DynamoDBから文字起こしデータを全件取得
"""
import boto3
import json

# DynamoDB接続
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1')
table = dynamodb.Table('recording-poc-records')

# 全件スキャン
print("Fetching all records from DynamoDB...")
response = table.scan()
items = response['Items']

# ページネーション対応
while 'LastEvaluatedKey' in response:
    response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
    items.extend(response['Items'])

print(f"\nTotal records: {len(items)}")

# 文字起こしテキストがあるものだけフィルタ
transcripts_with_text = []
for item in items:
    transcript_text = item.get('transcript_text', '').strip()
    if transcript_text:  # 空でない
        transcripts_with_text.append({
            'uuid': item.get('uuid'),
            'student_number': item.get('student_number'),
            'recorded_at': item.get('recorded_at'),
            'transcript_text': transcript_text,
            'duration_sec': item.get('duration_sec'),
            'speaker_count': item.get('transcript_speaker_count', 0),
            'confidence': item.get('transcript_confidence', 0)
        })

print(f"Records with transcript text: {len(transcripts_with_text)}")

# 結果表示
if transcripts_with_text:
    print("\n" + "="*80)
    print("Transcripts found:")
    print("="*80)
    for i, item in enumerate(transcripts_with_text, 1):
        print(f"\n[{i}] UUID: {item['uuid']}")
        print(f"    Student: {item['student_number']}")
        print(f"    Recorded: {item['recorded_at']}")
        print(f"    Duration: {item['duration_sec']} sec")
        print(f"    Speaker count: {item['speaker_count']}")
        print(f"    Confidence: {item['confidence']}")
        print(f"    Text length: {len(item['transcript_text'])} chars")
        print(f"    Text preview: {item['transcript_text'][:200]}...")
else:
    print("\n⚠️  No transcripts with text found!")
    print("\nChecking transcribe status for all records...")
    status_summary = {}
    for item in items:
        status = item.get('transcribe_status', 'UNKNOWN')
        status_summary[status] = status_summary.get(status, 0) + 1

    print("\nTranscribe status summary:")
    for status, count in status_summary.items():
        print(f"  {status}: {count}")

# JSONファイルに保存
if transcripts_with_text:
    output_file = 'transcripts_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcripts_with_text, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Transcripts saved to: {output_file}")
